//! Generation on a thread of its own, for an engine that has to block somewhere.
//!
//! Some compute APIs cannot be asked whether work has finished without waiting for it. Godot's
//! `RenderingDevice` is one: `sync()` blocks, and the device belongs to the thread that made it. A
//! `Worker` owns the generator on such a thread, takes requests through a channel and hands back
//! the events, so the engine's own thread never waits.

use crate::generator::{ChunkEvent, GeneratorStats, WorldGenerator};
use crate::scheduler::FocusPoint;
use crate::{Chunk, ChunkCoord, Error, Solver};
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::thread::JoinHandle;

/// What the worker's thread is asked to do.
enum Command {
    Request(Vec<FocusPoint>),
    Evict { focus: Vec<FocusPoint>, margin: u32 },
    Import(Box<Chunk>),
    Stop,
}

/// What the worker's thread reports.
///
/// A batch's events come with the tiles they are about. Anything else would leave a caller with a
/// coordinate and no way to read it without a round trip to a thread that may be mid-dispatch.
enum Report {
    Batch {
        events: Vec<ChunkEvent>,
        chunks: Vec<Chunk>,
        stats: Box<GeneratorStats>,
    },
    Evicted(Vec<Chunk>),
    Failed(String),
}

/// A generator running on its own thread.
///
/// The worker keeps the chunks it has reported, so [`Worker::chunk`] reads tiles without waiting
/// for the generating thread. [`Worker::drain`] is what moves both forward: it takes the events and
/// the tiles that came with them.
pub struct Worker {
    commands: Sender<Command>,
    reports: Receiver<Report>,
    thread: Option<JoinHandle<()>>,
    chunks: HashMap<ChunkCoord, Chunk>,
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
            chunks: HashMap::new(),
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

    /// Takes the events the worker has produced, without blocking, and with them the tiles of
    /// every chunk they are about.
    pub fn drain(&mut self) -> Vec<ChunkEvent> {
        let mut events = Vec::new();
        loop {
            match self.reports.try_recv() {
                Ok(Report::Batch {
                    events: batch,
                    chunks,
                    stats,
                }) => {
                    events.extend(batch);
                    for chunk in chunks {
                        self.chunks.insert(chunk.coord, chunk);
                    }
                    self.stats = *stats;
                }
                Ok(Report::Evicted(chunks)) => {
                    for chunk in &chunks {
                        self.chunks.remove(&chunk.coord);
                    }
                    events.extend(
                        chunks
                            .into_iter()
                            .map(|chunk| ChunkEvent::Evicted(chunk.coord)),
                    );
                }
                Ok(Report::Failed(reason)) => self.failure = Some(reason),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => return events,
            }
        }
    }

    /// A chunk's tiles, as of the last [`Worker::drain`].
    #[must_use]
    pub fn chunk(&self, coord: ChunkCoord) -> Option<&Chunk> {
        self.chunks.get(&coord)
    }

    /// Every chunk the worker has reported and not evicted.
    pub fn chunks(&self) -> impl Iterator<Item = &Chunk> {
        self.chunks.values()
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
                    let dropped = world.evict_outside(&focus, margin);
                    if !dropped.is_empty() {
                        let _ = reports.send(Report::Evicted(dropped));
                    }
                }
                Ok(Command::Import(chunk)) => {
                    if let Err(error) = world.import(*chunk) {
                        let _ = reports.send(Report::Failed(error.to_string()));
                    }
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
                    let dropped = world.evict_outside(&focus, margin);
                    if !dropped.is_empty() {
                        let _ = reports.send(Report::Evicted(dropped));
                    }
                }
                Ok(Command::Import(chunk)) => {
                    if let Err(error) = world.import(*chunk) {
                        let _ = reports.send(Report::Failed(error.to_string()));
                    }
                }
            }
        }
    }
}

/// One batch of generation, reported with the tiles it produced.
fn work<S: Solver>(world: &mut WorldGenerator<S>, reports: &Sender<Report>) -> Result<(), Error> {
    world.tick()?;
    let events = world.wait()?;
    if events.is_empty() {
        return Ok(());
    }
    let chunks: Vec<Chunk> = events
        .iter()
        .filter_map(|event| match event {
            ChunkEvent::Updated(coord) => world.chunk(*coord).cloned(),
            ChunkEvent::Failed { .. } | ChunkEvent::Evicted(_) => None,
        })
        .collect();
    let _ = reports.send(Report::Batch {
        events,
        chunks,
        stats: Box::new(*world.stats()),
    });
    Ok(())
}
