//! A pack's stages generated on a thread of their own, for an engine whose frame must not wait.
//!
//! The runtime is built inside the thread, because a town solver may own a device that belongs to
//! the thread that made it. The thread takes requests between small steps of generation, so a new
//! request never waits long, and hands every product back shared, so the engine's thread reads it
//! without a copy and without asking.

use super::edits::Edits;
use super::facts::{Facts, RowId};
use super::regions::Curve;
use super::runtime::{Categories, Field, Point, Product, Runtime, Site, StageTiming, TownChunk};
use super::save::Save;
use crate::ChunkCoord;
use crate::scheduler::FocusPoint;
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::sync::{Arc, Mutex};

/// Products generated between two looks for new requests: small, so a request waits little.
const STEP: usize = 8;

enum Order {
    Request {
        focus: Vec<FocusPoint>,
        /// Each target and its own radius, if it has one.
        targets: Vec<(String, Option<u32>)>,
    },
    Facts(Facts),
    Edits(Edits),
    Save,
    Load(Box<Save>),
    Focus {
        table: String,
        id: RowId,
    },
    Stop,
}

enum Report {
    /// Products, and every stage's cost so far.
    Generated(
        Vec<(String, ChunkCoord, Arc<Product>)>,
        Vec<(String, StageTiming)>,
    ),
    Dropped(Vec<(String, ChunkCoord)>),
    Failed(String),
    Saved(Box<Save>),
}

/// What changed about a stage's chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StageEvent {
    /// The product is ready to read.
    Generated { stage: String, chunk: ChunkCoord },
    /// The product is no longer needed and was dropped, so anything built from it can go too.
    Dropped { stage: String, chunk: ChunkCoord },
    /// The save [`StageWorker::request_save`] asked for is ready to take
    /// ([`StageWorker::take_save`]).
    Saved,
}

/// A runtime on its own thread. Dropping it asks the thread to stop and does not wait for it.
pub struct StageWorker {
    orders: Sender<Order>,
    /// Behind a mutex only so the worker can live where Sync is required (a Bevy resource);
    /// `drain` takes `&mut self` and reaches it without locking.
    reports: Mutex<Receiver<Report>>,
    products: HashMap<(String, ChunkCoord), Arc<Product>>,
    failure: Option<String>,
    /// The save the thread last made, until it is taken.
    saved: Option<Save>,
    timings: Vec<(String, StageTiming)>,
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
            reports: Mutex::new(received),
            products: HashMap::new(),
            failure: None,
            saved: None,
            timings: Vec::new(),
        }
    }

    /// Asks for the `targets` stages around `focus`, replacing the previous request.
    pub fn request(&self, focus: &[FocusPoint], targets: &[&str]) {
        let targets: Vec<(&str, Option<u32>)> =
            targets.iter().map(|&target| (target, None)).collect();
        self.request_each(focus, &targets);
    }

    /// Asks for each of `targets` within its own radius of every focus point, or the focus
    /// point's radius for a target given none, as [`Runtime::request_each`] does.
    pub fn request_each(&self, focus: &[FocusPoint], targets: &[(&str, Option<u32>)]) {
        // A worker whose thread has stopped reports through `failure`.
        let _ = self.orders.send(Order::Request {
            focus: focus.to_vec(),
            targets: targets
                .iter()
                .map(|&(target, radius)| (target.to_owned(), radius))
                .collect(),
        });
    }

    /// Gives the runtime new facts, as [`Runtime::set_facts`] does: what they make stale arrives as
    /// drops, and is generated again. An error stops the thread and arrives as a failure.
    pub fn set_facts(&self, facts: Facts) {
        let _ = self.orders.send(Order::Facts(facts));
    }

    /// Gives the runtime the player's edits, as [`Runtime::set_edits`] does: what the change
    /// reaches arrives as drops, and is generated again with the edits. An error stops the thread
    /// and arrives as a failure.
    pub fn set_edits(&self, edits: Edits) {
        let _ = self.orders.send(Order::Edits(edits));
    }

    /// Asks the thread for a save of the world as [`Runtime::save`] makes it; it arrives as a
    /// [`StageEvent::Saved`] from [`StageWorker::drain`], to take with [`StageWorker::take_save`].
    pub fn request_save(&self) {
        let _ = self.orders.send(Order::Save);
    }

    /// The save the thread last made, once: `None` before it arrives or after it was taken.
    pub fn take_save(&mut self) -> Option<Save> {
        self.saved.take()
    }

    /// Brings the world back from `save`, as [`Runtime::load`] does: what it changes arrives as
    /// drops and is generated again. An error stops the thread and arrives as a failure.
    pub fn load(&self, save: Save) {
        let _ = self.orders.send(Order::Load(Box::new(save)));
    }

    /// Focuses the runtime on the row `id` of `table`, as [`Runtime::focus`] does, with what that
    /// makes stale arriving as drops. An error stops the thread and arrives as a failure.
    pub fn focus(&self, table: &str, id: RowId) {
        let _ = self.orders.send(Order::Focus {
            table: table.to_owned(),
            id,
        });
    }

    /// Takes what the thread has produced or dropped since the last call, without blocking.
    pub fn drain(&mut self) -> Vec<StageEvent> {
        let mut events = Vec::new();
        let reports = self
            .reports
            .get_mut()
            .expect("nothing panics while holding the reports");
        loop {
            match reports.try_recv() {
                Ok(Report::Generated(products, timings)) => {
                    self.timings = timings;
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
                Ok(Report::Saved(save)) => {
                    self.saved = Some(*save);
                    events.push(StageEvent::Saved);
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => return events,
            }
        }
    }

    /// What each stage has cost on the thread, as of the last [`StageWorker::drain`], in the
    /// order the pack lists them; empty until something is generated.
    #[must_use]
    pub fn timings(&self) -> &[(String, StageTiming)] {
        &self.timings
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
            Product::Sites(_)
            | Product::Tiles(_)
            | Product::Points(_)
            | Product::Categories(_)
            | Product::Curves(_) => None,
        }
    }

    /// The categories `stage` holds for `chunk`, if it is a Rules stage and they have arrived.
    #[must_use]
    pub fn categories(&self, stage: &str, chunk: ChunkCoord) -> Option<&Categories> {
        match self.product(stage, chunk)? {
            Product::Categories(categories) => Some(categories),
            Product::Field(_)
            | Product::Sites(_)
            | Product::Tiles(_)
            | Product::Points(_)
            | Product::Curves(_) => None,
        }
    }

    /// The curves `stage` holds for `chunk`, if it is a Region stage and they have arrived.
    #[must_use]
    pub fn curves(&self, stage: &str, chunk: ChunkCoord) -> Option<&[Curve]> {
        match self.product(stage, chunk)? {
            Product::Curves(curves) => Some(curves),
            Product::Field(_)
            | Product::Categories(_)
            | Product::Sites(_)
            | Product::Tiles(_)
            | Product::Points(_) => None,
        }
    }

    /// The sites `stage` holds for `chunk`, if it is a Sites stage and they have arrived.
    #[must_use]
    pub fn sites(&self, stage: &str, chunk: ChunkCoord) -> Option<&[Site]> {
        match self.product(stage, chunk)? {
            Product::Sites(sites) => Some(sites),
            Product::Field(_)
            | Product::Tiles(_)
            | Product::Points(_)
            | Product::Categories(_)
            | Product::Curves(_) => None,
        }
    }

    /// The town chunk `stage` holds for `chunk`, if it is a Solve stage's and lies in a site.
    #[must_use]
    pub fn tiles(&self, stage: &str, chunk: ChunkCoord) -> Option<&TownChunk> {
        match self.product(stage, chunk)? {
            Product::Tiles(town) => town.as_ref(),
            Product::Field(_)
            | Product::Sites(_)
            | Product::Points(_)
            | Product::Categories(_)
            | Product::Curves(_) => None,
        }
    }

    /// The points `stage` placed in `chunk`, if it is a Scatter stage and they have arrived.
    #[must_use]
    pub fn points(&self, stage: &str, chunk: ChunkCoord) -> Option<&[Point]> {
        match self.product(stage, chunk)? {
            Product::Points(points) => Some(points),
            Product::Field(_)
            | Product::Sites(_)
            | Product::Tiles(_)
            | Product::Categories(_)
            | Product::Curves(_) => None,
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
        if let Some(order) = order {
            let result = match order {
                Order::Stop => return,
                Order::Request { focus, targets } => {
                    let targets: Vec<(&str, Option<u32>)> = targets
                        .iter()
                        .map(|(target, radius)| (target.as_str(), *radius))
                        .collect();
                    runtime.request_each(&focus, &targets)
                }
                Order::Facts(facts) => runtime.set_facts(facts),
                Order::Edits(edits) => runtime.set_edits(&edits),
                Order::Save => {
                    let _ = reports.send(Report::Saved(Box::new(runtime.save())));
                    Ok(Vec::new())
                }
                Order::Load(save) => runtime.load(&save),
                Order::Focus { table, id } => runtime.focus(&table, id),
            };
            match result {
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
                let _ = reports.send(Report::Generated(products, runtime.timings()));
            }
            Err(error) => {
                let _ = reports.send(Report::Failed(error.to_string()));
                return;
            }
        }
    }
}
