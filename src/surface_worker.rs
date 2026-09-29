//! Volume surfaces built on a thread of their own, so an engine's thread only draws them.
//!
//! Meshing a chunk's surface ([`volume_mesh`]) takes about a millisecond for a chunk of 16 by 16
//! columns and 64 levels (docs/research/measurements.md, E53), which a frame cannot spare many of.
//! A [`SurfaceWorker`] takes a chunk and the nine volumes around it, shared rather than copied,
//! and hands the surface back once it is built. Each request carries a ticket, and only the
//! result of a chunk's newest request comes back, so an engine that asked again after an edit, or
//! cancelled after a volume was dropped, never draws a surface built from volumes it no longer
//! holds.

use crate::stages::Product;
use crate::volume_mesh::{VolumeMesh, volume_mesh};
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use wfc_core::ChunkCoord;

/// One chunk's surface to build: its ticket, the chunk, and the Volume stage's products around it.
struct Job {
    ticket: u64,
    chunk: ChunkCoord,
    around: [Arc<Product>; 9],
    voxel_size: [f32; 3],
}

/// Builds volume surfaces on a thread of its own.
pub struct SurfaceWorker {
    jobs: Sender<Job>,
    done: Mutex<Receiver<(u64, VolumeMesh)>>,
    /// The ticket of each chunk's newest request, which only its own result answers.
    asked: HashMap<ChunkCoord, u64>,
    tickets: u64,
}

impl SurfaceWorker {
    /// A worker with a thread of its own.
    ///
    /// # Panics
    /// If the thread cannot be started.
    #[must_use]
    pub fn spawn() -> Self {
        let (jobs, taken) = channel::<Job>();
        let (finished, done) = channel();
        // Not joined: the thread ends when the worker, and with it the job sender, is dropped.
        std::thread::Builder::new()
            .name("wave forge surfaces".to_owned())
            .spawn(move || {
                for job in taken {
                    let mesh = volume_mesh(
                        job.chunk,
                        |at| {
                            let (dx, dy) = (at.x - job.chunk.x + 1, at.y - job.chunk.y + 1);
                            match job.around[(dy * 3 + dx) as usize].as_ref() {
                                Product::Volume(volume) => Some(volume),
                                _ => unreachable!("a surface is built from a Volume stage"),
                            }
                        },
                        job.voxel_size,
                    )
                    .expect("every volume around the chunk is given");
                    if finished.send((job.ticket, mesh)).is_err() {
                        break;
                    }
                }
            })
            .expect("a thread");
        Self {
            jobs,
            done: Mutex::new(done),
            asked: HashMap::new(),
            tickets: 0,
        }
    }

    /// Asks for the surface of `chunk` from `around`, the Volume stage's products at the chunks
    /// around it row by row from its lower corner, `around[4]` being its own, with voxels
    /// `voxel_size` along the engine's x, y and z. It replaces any request for the chunk not yet
    /// answered.
    ///
    /// # Panics
    /// If the worker's thread has stopped, which only a panic in [`volume_mesh`] does.
    pub fn build(&mut self, chunk: ChunkCoord, around: [Arc<Product>; 9], voxel_size: [f32; 3]) {
        self.tickets += 1;
        self.asked.insert(chunk, self.tickets);
        self.jobs
            .send(Job {
                ticket: self.tickets,
                chunk,
                around,
                voxel_size,
            })
            .expect("the surface thread runs as long as its worker");
    }

    /// Forgets the request for `chunk`, if one is waiting, so its result never comes back.
    pub fn cancel(&mut self, chunk: ChunkCoord) {
        self.asked.remove(&chunk);
    }

    /// Whether a request for `chunk` is waiting for its surface.
    #[must_use]
    pub fn is_building(&self, chunk: ChunkCoord) -> bool {
        self.asked.contains_key(&chunk)
    }

    /// How many requests are waiting for their surfaces.
    #[must_use]
    pub fn building(&self) -> usize {
        self.asked.len()
    }

    /// The surfaces built since the last call that answer their chunk's newest request, in the
    /// order they were built.
    pub fn drain(&mut self) -> Vec<VolumeMesh> {
        let done = self.done.lock().expect("only the engine's thread drains");
        let mut out = Vec::new();
        while let Ok((ticket, mesh)) = done.try_recv() {
            if self.asked.get(&mesh.chunk) == Some(&ticket) {
                self.asked.remove(&mesh.chunk);
                out.push(mesh);
            }
        }
        out
    }
}
