//! Generation on a thread of its own, for an engine that has to block somewhere.
//!
//! Some compute APIs cannot be asked whether work has finished without waiting for it. Godot's
//! `RenderingDevice` is one: `sync()` blocks, and the device belongs to the thread that made it. A
//! `Worker` owns the generator on such a thread, takes requests through a channel and hands back
//! the events, so the engine's own thread never waits.

use crate::generator::{ChunkEvent, GeneratorStats, WorldGenerator};
use crate::scheduler::FocusPoint;
use crate::{Chunk, ChunkCoord, Error, Solver};
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::thread::JoinHandle;

/// What the worker's thread is asked to do.
enum Command {
    Request(Vec<FocusPoint>),
    Evict { focus: Vec<FocusPoint>, margin: u32 },
    Import(Box<Chunk>),
    Chunk(ChunkCoord),
    Stop,
}

/// What the worker's thread reports.
enum Report {
    Events(Vec<ChunkEvent>),
    Chunk(Option<Box<Chunk>>),
    Stats(Box<GeneratorStats>),
    Failed(String),
}

/// A generator running on its own thread.
pub struct Worker {
    commands: Sender<Command>,
    reports: Receiver<Report>,
    thread: Option<JoinHandle<()>>,
    stats: GeneratorStats,
    failure: Option<String>,
}

impl Worker {
    /// Starts a thread and builds the generator on it with `build`.
    ///
    /// The generator is built inside the thread because a device may belong to the thread that
    /// created it, and then nothing about it can cross a thread boundary.
    ///
    /// # Errors
    /// Building runs on the worker's thread, so a failure there arrives at the first
    /// [`Worker::drain`] rather than here.
    ///
    /// # Panics
    /// If the operating system will not start a thread.
    pub fn spawn<S, B>(build: B) -> Self
    where
        S: Solver,
        B: FnOnce() -> Result<WorldGenerator<S>, Error> + Send + 'static,
    {
        let (commands, orders) = channel::<Command>();
        let (reports, results) = channel::<Report>();
        let thread = std::thread::Builder::new()
            .name("wave forge".to_owned())
            .spawn(move || run(build, &orders, &reports))
            .expect("a thread");
        Self {
            commands,
            reports: results,
            thread: Some(thread),
            stats: GeneratorStats::default(),
            failure: None,
        }
    }

    /// Asks for the chunks around these focus points.
    pub fn request(&self, focus: &[FocusPoint]) {
        self.send(Command::Request(focus.to_vec()));
    }

    /// Drops the chunks far from these focus points.
    pub fn evict_outside(&self, focus: &[FocusPoint], margin: u32) {
        self.send(Command::Evict {
            focus: focus.to_vec(),
            margin,
        });
    }

    /// Puts a chunk back.
    pub fn import(&self, chunk: Chunk) {
        self.send(Command::Import(Box::new(chunk)));
    }

    /// Takes the events the worker has produced, without blocking.
    pub fn drain(&mut self) -> Vec<ChunkEvent> {
        let mut events = Vec::new();
        loop {
            match self.reports.try_recv() {
                Ok(Report::Events(batch)) => events.extend(batch),
                Ok(Report::Stats(stats)) => self.stats = *stats,
                Ok(Report::Chunk(_)) => {}
                Ok(Report::Failed(reason)) => self.failure = Some(reason),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => return events,
            }
        }
    }

    /// A chunk's tiles, waiting for the worker to answer.
    pub fn chunk(&mut self, coord: ChunkCoord) -> Option<Chunk> {
        self.send(Command::Chunk(coord));
        while let Ok(report) = self.reports.recv() {
            match report {
                Report::Chunk(chunk) => return chunk.map(|chunk| *chunk),
                Report::Events(_) => {}
                Report::Stats(stats) => self.stats = *stats,
                Report::Failed(reason) => {
                    self.failure = Some(reason);
                    return None;
                }
            }
        }
        None
    }

    /// What generation has cost, as of the last report.
    #[must_use]
    pub const fn stats(&self) -> &GeneratorStats {
        &self.stats
    }

    /// Why the worker stopped, if it did. Generation happens on another thread, so a failure is
    /// reported here rather than returned from a call.
    #[must_use]
    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    fn send(&self, command: Command) {
        // A worker whose thread has stopped reports through `failure`; dropping a command is how a
        // caller that keeps asking stays harmless.
        let _ = self.commands.send(command);
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.send(Command::Stop);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The worker's thread: build, then take orders and generate until asked to stop.
fn run<S, B>(build: B, orders: &Receiver<Command>, reports: &Sender<Report>)
where
    S: Solver,
    B: FnOnce() -> Result<WorldGenerator<S>, Error>,
{
    let mut world = match build() {
        Ok(world) => world,
        Err(error) => {
            let _ = reports.send(Report::Failed(error.to_string()));
            return;
        }
    };
    loop {
        // Take every order that has arrived, then do a batch of work. A batch is short, so a
        // request never waits long behind one.
        loop {
            match orders.try_recv() {
                Ok(Command::Request(focus)) => world.request(&focus),
                Ok(Command::Evict { focus, margin }) => {
                    world.evict_outside(&focus, margin);
                }
                Ok(Command::Import(chunk)) => {
                    if let Err(error) = world.import(*chunk) {
                        let _ = reports.send(Report::Failed(error.to_string()));
                    }
                }
                Ok(Command::Chunk(coord)) => {
                    let chunk = world.chunk(coord).cloned().map(Box::new);
                    let _ = reports.send(Report::Chunk(chunk));
                }
                Ok(Command::Stop) => return,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        if let Err(error) = work(&mut world, reports) {
            let _ = reports.send(Report::Failed(error.to_string()));
            return;
        }
        if world.is_idle() {
            // Nothing to do until an order arrives, so wait for one rather than spinning.
            match orders.recv() {
                Ok(Command::Stop) | Err(_) => return,
                Ok(Command::Request(focus)) => world.request(&focus),
                Ok(Command::Evict { focus, margin }) => {
                    world.evict_outside(&focus, margin);
                }
                Ok(Command::Import(chunk)) => {
                    if let Err(error) = world.import(*chunk) {
                        let _ = reports.send(Report::Failed(error.to_string()));
                    }
                }
                Ok(Command::Chunk(coord)) => {
                    let chunk = world.chunk(coord).cloned().map(Box::new);
                    let _ = reports.send(Report::Chunk(chunk));
                }
            }
        }
    }
}

/// One batch of generation, reported as it finishes.
fn work<S: Solver>(world: &mut WorldGenerator<S>, reports: &Sender<Report>) -> Result<(), Error> {
    world.tick()?;
    let events = world.wait()?;
    if !events.is_empty() {
        let _ = reports.send(Report::Events(events));
        let _ = reports.send(Report::Stats(Box::new(*world.stats())));
    }
    Ok(())
}
