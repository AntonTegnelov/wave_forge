//! A pack's stages generated on a thread of their own, for an engine whose frame must not wait.
//!
//! The runtime is built inside the thread, because a town solver may own a device that belongs to
//! the thread that made it. The thread takes requests between small steps of generation, so a new
//! request never waits long, and hands every product back shared, so the engine's thread reads it
//! without a copy and without asking.

use super::runtime::{Field, Point, Product, Runtime, Site, TownChunk};
use crate::ChunkCoord;
use crate::scheduler::FocusPoint;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};

/// Products generated between two looks for new requests: small, so a request waits little.
const STEP: usize = 8;

enum Order {
    Request {
        focus: Vec<FocusPoint>,
        targets: Vec<String>,
    },
    Stop,
}

enum Report {
    Generated(Vec<(String, ChunkCoord, Arc<Product>)>),
    Dropped(Vec<(String, ChunkCoord)>),
    Failed(String),
}

/// What changed about a stage's chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StageEvent {
    /// The product is ready to read.
    Generated { stage: String, chunk: ChunkCoord },
    /// The product is no longer needed and was dropped, so anything built from it can go too.
    Dropped { stage: String, chunk: ChunkCoord },
}

/// A runtime on its own thread. Dropping it asks the thread to stop and does not wait for it.
pub struct StageWorker {
    orders: Sender<Order>,
    reports: Receiver<Report>,
    products: HashMap<(String, ChunkCoord), Arc<Product>>,
    failure: Option<String>,
}

impl StageWorker {
    /// Starts a thread and builds the runtime on it with `build`; a failure to build arrives at the
    /// first [`StageWorker::drain`], as [`StageWorker::failure`].
    ///
    /// # Panics
    /// If the operating system will not start a thread.
    pub fn spawn<B>(build: B) -> Self
    where
        B: FnOnce() -> Result<Runtime, String> + Send + 'static,
    {
        let (orders, taken) = channel::<Order>();
        let (reports, received) = channel::<Report>();
        // Not joined: a thread still building (a device, kernels) takes seconds, and an engine
        // drops its worker on its main thread.
        std::thread::Builder::new()
            .name("wave forge stages".to_owned())
            .spawn(move || run(build, &taken, &reports))
            .expect("a thread");
        Self {
            orders,
            reports: received,
            products: HashMap::new(),
            failure: None,
        }
    }

    /// Asks for the `targets` stages around `focus`, replacing the previous request.
    pub fn request(&self, focus: &[FocusPoint], targets: &[&str]) {
        // A worker whose thread has stopped reports through `failure`.
        let _ = self.orders.send(Order::Request {
            focus: focus.to_vec(),
            targets: targets.iter().map(|target| (*target).to_owned()).collect(),
        });
    }

    /// Takes what the thread has produced or dropped since the last call, without blocking.
    pub fn drain(&mut self) -> Vec<StageEvent> {
        let mut events = Vec::new();
        loop {
            match self.reports.try_recv() {
                Ok(Report::Generated(products)) => {
                    for (stage, chunk, product) in products {
                        events.push(StageEvent::Generated {
                            stage: stage.clone(),
                            chunk,
                        });
                        self.products.insert((stage, chunk), product);
                    }
                }
                Ok(Report::Dropped(dropped)) => {
                    for (stage, chunk) in dropped {
                        self.products.remove(&(stage.clone(), chunk));
                        events.push(StageEvent::Dropped { stage, chunk });
                    }
                }
                Ok(Report::Failed(reason)) => self.failure = Some(reason),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => return events,
            }
        }
    }

    /// Why the thread stopped, if it did.
    #[must_use]
    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    /// What `stage` holds for `chunk`, as of the last [`StageWorker::drain`].
    #[must_use]
    pub fn product(&self, stage: &str, chunk: ChunkCoord) -> Option<&Product> {
        self.products
            .get(&(stage.to_owned(), chunk))
            .map(Arc::as_ref)
    }

    /// The field `stage` holds for `chunk`, if it is one and has arrived.
    #[must_use]
    pub fn field(&self, stage: &str, chunk: ChunkCoord) -> Option<&Field> {
        match self.product(stage, chunk)? {
            Product::Field(field) => Some(field),
            Product::Sites(_) | Product::Tiles(_) | Product::Points(_) => None,
        }
    }

    /// The sites `stage` holds for `chunk`, if it is a Sites stage and they have arrived.
    #[must_use]
    pub fn sites(&self, stage: &str, chunk: ChunkCoord) -> Option<&[Site]> {
        match self.product(stage, chunk)? {
            Product::Sites(sites) => Some(sites),
            Product::Field(_) | Product::Tiles(_) | Product::Points(_) => None,
        }
    }

    /// The town chunk `stage` holds for `chunk`, if it is a Solve stage's and lies in a site.
    #[must_use]
    pub fn tiles(&self, stage: &str, chunk: ChunkCoord) -> Option<&TownChunk> {
        match self.product(stage, chunk)? {
            Product::Tiles(town) => town.as_ref(),
            Product::Field(_) | Product::Sites(_) | Product::Points(_) => None,
        }
    }

    /// The points `stage` placed in `chunk`, if it is a Scatter stage and they have arrived.
    #[must_use]
    pub fn points(&self, stage: &str, chunk: ChunkCoord) -> Option<&[Point]> {
        match self.product(stage, chunk)? {
            Product::Points(points) => Some(points),
            Product::Field(_) | Product::Sites(_) | Product::Tiles(_) => None,
        }
    }
}

impl Drop for StageWorker {
    fn drop(&mut self) {
        let _ = self.orders.send(Order::Stop);
    }
}

/// The thread: build, then alternate between taking orders and a step of generation.
fn run<B>(build: B, orders: &Receiver<Order>, reports: &Sender<Report>)
where
    B: FnOnce() -> Result<Runtime, String>,
{
    let mut runtime = match build() {
        Ok(runtime) => runtime,
        Err(reason) => {
            let _ = reports.send(Report::Failed(reason));
            return;
        }
    };
    loop {
        let order = if runtime.is_idle() {
            // Nothing to do until an order arrives, so wait for one rather than spinning.
            match orders.recv() {
                Ok(order) => Some(order),
                Err(_) => return,
            }
        } else {
            match orders.try_recv() {
                Ok(order) => Some(order),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => return,
            }
        };
        match order {
            Some(Order::Stop) => return,
            Some(Order::Request { focus, targets }) => {
                let names: Vec<&str> = targets.iter().map(String::as_str).collect();
                match runtime.request(&focus, &names) {
                    Ok(dropped) if dropped.is_empty() => {}
                    Ok(dropped) => {
                        let _ = reports.send(Report::Dropped(dropped));
                    }
                    Err(error) => {
                        let _ = reports.send(Report::Failed(error.to_string()));
                        return;
                    }
                }
                // Take every order already waiting before generating for this one.
                continue;
            }
            None => {}
        }
        match runtime.step(STEP) {
            Ok(generated) if generated.is_empty() => {}
            Ok(generated) => {
                let products = generated
                    .into_iter()
                    .map(|(stage, chunk)| {
                        let product = Arc::clone(
                            runtime
                                .shared(&stage, chunk)
                                .expect("a product just generated is held"),
                        );
                        (stage, chunk, product)
                    })
                    .collect();
                let _ = reports.send(Report::Generated(products));
            }
            Err(error) => {
                let _ = reports.send(Report::Failed(error.to_string()));
                return;
            }
        }
    }
}
