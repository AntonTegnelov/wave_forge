//! Workers' threads the process waits for before it exits. Dropping a worker does not wait for its
//! thread, since a node drops it on the main thread and a thread still building a device takes
//! seconds; but a thread still tearing its device down when the process exits faults in the
//! driver (docs/architecture/overview.md, "Threads").

use std::ops::{Deref, DerefMut};
use std::sync::Mutex;
use std::thread::JoinHandle;
use wave_forge::Worker;
use wave_forge::stages::StageWorker;

/// The threads of dropped workers that may still be ending. It is the process's, because exiting
/// is: the hook that joins them, the extension's unloading, has no node to keep them on.
static ENDING: Mutex<Vec<JoinHandle<()>>> = Mutex::new(Vec::new());

/// Keeps `thread` to join before the process exits, forgetting those that have ended.
pub(crate) fn join_before_exit(thread: JoinHandle<()>) {
    let mut ending = ENDING
        .lock()
        .expect("nothing panics while holding the threads");
    ending.retain(|thread| !thread.is_finished());
    ending.push(thread);
}

/// Waits for every thread kept, as the extension is unloaded.
pub(crate) fn join_all() {
    let threads = std::mem::take(
        &mut *ENDING
            .lock()
            .expect("nothing panics while holding the threads"),
    );
    for thread in threads {
        // A thread that panicked was reported through its worker already.
        let _ = thread.join();
    }
}

/// A worker that stops its thread for the process to wait for when it is dropped.
pub(crate) trait Finish {
    fn finish(self) -> JoinHandle<()>;
}

impl Finish for Worker {
    fn finish(self) -> JoinHandle<()> {
        Worker::finish(self)
    }
}

impl Finish for StageWorker {
    fn finish(self) -> JoinHandle<()> {
        StageWorker::finish(self)
    }
}

/// A node's worker, whose thread is joined before the process exits once the worker is dropped.
pub(crate) struct Ending<W: Finish>(Option<W>);

impl<W: Finish> Ending<W> {
    pub(crate) fn new(worker: W) -> Self {
        Self(Some(worker))
    }
}

impl<W: Finish> Deref for Ending<W> {
    type Target = W;

    fn deref(&self) -> &W {
        self.0.as_ref().expect("taken only when dropped")
    }
}

impl<W: Finish> DerefMut for Ending<W> {
    fn deref_mut(&mut self) -> &mut W {
        self.0.as_mut().expect("taken only when dropped")
    }
}

impl<W: Finish> Drop for Ending<W> {
    fn drop(&mut self) {
        let worker = self.0.take().expect("taken only here");
        join_before_exit(worker.finish());
    }
}
