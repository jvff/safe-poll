//! Abstractions to help to avoid writing [`Future`] implementations that never wake up.
//!
//! This crate experiments with using Rust's type system to help avoid one common pitfall when
//! writing manual implementations of [`Future`]. Every implementation of [`Future::poll`] returns
//! [`Poll`]. [`Poll`] has two variants, [`Poll::Ready`] to represent when the asynchronous
//! computation has produced a value and [`Poll::Pending`] when the asynchronous computation isn't
//! ready to produce a value yet, and should be polled again later. Returning [`Poll::Pending`]
//! requires having the current task registered for a wakeup when it should be polled again.
//! Returning [`Poll::Pending`] without registering for a wakeup will mean that the current task
//! will never be polled again, and therefore it will never advance its computation.
//!
//! An example of this pitfall follows:
//!
//! ```
//! # use futures::Stream;
//! # use std::{future::Future, pin::Pin, task::{Context, Poll}};
//! #
//! struct MyFuture {
//!     inner_stream: Pin<Box<dyn Stream<Item = usize>>>,
//! }
//!
//! impl Future for MyFuture {
//!     type Output = usize;
//!
//!     fn poll(mut self: Pin<&mut Self>, context: &mut Context) -> Poll<Self::Output> {
//!         match self.as_mut().inner_stream.as_mut().poll_next(context) {
//!             Poll::Ready(None) => {
//!                 unreachable!("The inner stream will produce at least one positive integer");
//!             }
//!             Poll::Ready(Some(value)) if (value > 0) => Poll::Ready(value),
//!             _ => Poll::Pending,
//!         }
//!     }
//! }
//! ```
//!
//! In this example, the intention is pretty clear. The developer wants the future to only complete
//! when the inner stream of asynchronous computations results in a positive value. However, if the
//! inner stream produces a zero, although the intention is for `MyFuture` to always be polled
//! again if it returns [`Poll::Pending`], it will not be polled ever again if the `inner_stream`
//! produces a zero before producing a positive integer. If that happens, `MyFuture` will never
//! complete.
//!
//! Note that this is not a great example, because it could be replaced by simpler and safer code,
//! like:
//!
//! ```
//! use futures::{future, FutureExt, Stream, StreamExt};
//! use std::future::Future;
//!
//! fn make_my_future(stream: impl Stream<Item = usize> + Unpin) -> impl Future<Output = usize> {
//!     stream
//!         .filter(|&value| future::ready(value > 0))
//!         .into_future()
//!         .map(|(item, _rest_of_stream)| {
//!             item.expect("The inner stream will produce at least one positive integer")
//!         })
//! }
//! ```
//!
//! The purpose of this crate is to try to help avoiding the pitfall described above using the type
//! system. The crate introduces a new `SafePoll` type that's equivalent to [`Poll`], but the
//! `SafePoll::Pending` variant requires a special token to be created. That token is of a new
//! [`WakeupRegisteredToken`] type, which can only be created in an `unsafe` block. However, once a
//! token is created, it can be returned inside `SafePoll::Pending` and any outer types that are
//! asynchronous can just forward that token to outer layers without any `unsafe` blocks. This
//! means that an `unsafe` block is required when the type system can't guarantee that a wakeup was
//! registered, which is usually done once, deep inside the code where the low-level details of
//! wakeup registration are handled. The idea is that `unsafe` blocks make it easier for developers
//! to find where wakeups need to be properly registered, so that it's easier to verify and harder
//! to make mistakes.
//!
//! The goal is for the initial example to become something more correct, even if it's more
//! complex:
//!
//! ```ignore
//! # use futures::{Stream, StreamExt};
//! # use pin_project::pin_project;
//! # use pin_utils::pin_mut;
//! # use safe_poll::{AssumeSafe, SafeFuture, SafePoll};
//! # use std::{future::Future, pin::Pin, task::{Context, Poll}};
//! #
//! #[pin_project]
//! struct MyFuture {
//!     #[pin]
//!     inner_stream: Pin<Box<dyn Stream<Item = usize>>>,
//! }
//!
//! impl SafeFuture for MyFuture {
//!     type Output = usize;
//!
//!     fn safe_poll(mut self: Pin<&mut Self>, context: &mut Context) -> SafePoll<Self::Output> {
//!         let mut this = self.project();
//!         let mut inner_stream = this.inner_stream.as_mut();
//!
//!         loop {
//!             let next_item_future = AssumeSafe(inner_stream.next());
//!
//!             pin_mut!(next_item_future);
//!
//!             match next_item_future.safe_poll(context) {
//!                 SafePoll::Ready(None) => {
//!                     unreachable!("The inner stream will produce at least one positive integer");
//!                 }
//!                 SafePoll::Ready(Some(0)) => continue,
//!                 SafePoll::Ready(Some(value)) => return SafePoll::Ready(value),
//!                 SafePoll::Pending(token) => return SafePoll::Pending(token),
//!             }
//!         }
//!     }
//! }
//!
//! impl Future for MyFuture {
//!     type Output = <Self as SafeFuture>::Output;
//!
//!     fn poll(self: Pin<&mut Self>, context: &mut Context) -> Poll<Self::Output> {
//!         self.safe_poll(context).into()
//!     }
//! }
//! ```
//!
//! [`Future`]: std::future::Future
//! [`Future::poll`]: std::future::Future::poll
//! [`Poll`]: std::task::Poll
//! [`Poll::Ready`]: std::task::Poll::Ready
//! [`Poll::Pending`]: std::task::Poll::Pending

use std::marker::PhantomData;

/// A zero-sized marker type to indicate that the current task was correctly registered for waking
/// up to be polled again later.
///
/// This type only has an `unsafe` constructor, because it can only be constructed after the
/// current task was correctly registered to wake up and be polled again after returning. However,
/// it is possible to obtain the token without any `unsafe` blocks by extracting it from a
/// `SafePoll` returned from a poll to an inner type.
///
/// # Safety
///
/// Creating this type without correctly registering the current task for a wakeup might lead to a
/// situation where the task is never polled again.
#[derive(Debug)]
pub struct WakeupRegisteredToken {
    _inner: PhantomData<()>,
}

impl WakeupRegisteredToken {
    /// Create a new token indicating that the current task was registered for a wakeup to be
    /// polled again.
    ///
    /// This should only be called after properly registering the current task for a wakeup using
    /// the [`Context`](std::task::Context) provided in the poll method implementation.
    ///
    /// # Safety
    ///
    /// Calling this method without correctly registering the current task for a wakeup might lead
    /// to a situation where the task is never polled again.
    pub unsafe fn new() -> Self {
        WakeupRegisteredToken {
            _inner: PhantomData,
        }
    }
}
