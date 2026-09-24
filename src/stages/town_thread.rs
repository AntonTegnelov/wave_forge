//! A town solver on a thread of its own, so a runtime generates every other stage while a town is
//! being solved, its kernels compiled included.

use super::runtime::SiteId;
use crate::towns::{Selector, Town, TownError, TownRequest, TownSolver};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError, channel};
use std::time::{Duration, Instant};

/// Which town a request is for: a Solve stage's index and a site, and a ticket that tells a
/// request from an earlier one for the same site that has gone stale since.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TownKey {
    pub(crate) stage: usize,
    pub(crate) site: SiteId,
    pub(crate) ticket: u64,
}

/// One town to solve, owned so it can cross to the thread.
pub(crate) struct Job {
    pub(crate) key: TownKey,
    pub(crate) rules: String,
    pub(crate) seed: u64,
    pub(crate) size: (u32, u32),
    pub(crate) bottom: Option<Selector>,
    pub(crate) top: Option<Selector>,
}

/// A town the thread solved, and the milliseconds solving it took.
pub(crate) struct Done {
    pub(crate) key: TownKey,
    pub(crate) town: Result<Town, TownError>,
    pub(crate) ms: f64,
}

/// The runtime's end of the town thread, which solves the jobs it is sent in order until the
/// runtime drops its end.
pub(crate) struct TownThread {
    jobs: Sender<Job>,
    done: Receiver<Done>,
}

impl TownThread {
    pub(crate) fn spawn(mut solver: Box<dyn TownSolver>) -> Self {
        let (jobs, taken) = channel::<Job>();
        let (sent, done) = channel::<Done>();
        // Not joined: dropping a GPU solver's device takes time, and a runtime is dropped on
        // whatever thread held it.
        std::thread::Builder::new()
            .name("wave forge towns".to_owned())
            .spawn(move || {
                for job in taken {
                    let started = Instant::now();
                    let town = solver.solve(&TownRequest {
                        rules: &job.rules,
                        seed: job.seed,
                        size: job.size,
                        bottom: job.bottom.as_ref(),
                        top: job.top.as_ref(),
                    });
                    let ms = started.elapsed().as_secs_f64() * 1000.0;
                    if sent
                        .send(Done {
                            key: job.key,
                            town,
                            ms,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
            })
            .expect("a thread");
        Self { jobs, done }
    }

    pub(crate) fn send(&self, job: Job) {
        // A thread that stopped is reported by `take`.
        let _ = self.jobs.send(job);
    }

    /// What the thread has finished, waiting up to `wait` for the first of it.
    ///
    /// # Errors
    /// [`Stopped`] if the thread has stopped, which only a panic in the solver does.
    pub(crate) fn take(&self, wait: Duration) -> Result<Vec<Done>, Stopped> {
        let mut done = Vec::new();
        if wait > Duration::ZERO {
            match self.done.recv_timeout(wait) {
                Ok(first) => done.push(first),
                Err(RecvTimeoutError::Timeout) => return Ok(done),
                Err(RecvTimeoutError::Disconnected) => return Err(Stopped),
            }
        }
        loop {
            match self.done.try_recv() {
                Ok(next) => done.push(next),
                Err(TryRecvError::Empty) => return Ok(done),
                Err(TryRecvError::Disconnected) if done.is_empty() => return Err(Stopped),
                Err(TryRecvError::Disconnected) => return Ok(done),
            }
        }
    }
}

/// The town thread stopped, which only a panic in the solver makes it do.
#[derive(Debug)]
pub(crate) struct Stopped;
