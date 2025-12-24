use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, RwLock,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::utils::errors::DBError;

use super::Segment;

pub type SharedSegment = Arc<RwLock<Segment>>;

#[derive(Clone, Debug)]
pub struct SnapshotConfig {
    pub path: PathBuf,
    pub interval: Duration,
    pub max_ops: u64,
    pub check_every: Duration,
}

impl SnapshotConfig {
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        Self {
            path: path.into(),
            interval: Duration::from_secs(60),
            max_ops: 10_000,
            check_every: Duration::from_secs(1),
        }
    }
}

pub struct SnapshotterHandle {
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl SnapshotterHandle {
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl Drop for SnapshotterHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

pub fn start_background_snapshots(segment: SharedSegment, config: SnapshotConfig) -> SnapshotterHandle {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = stop.clone();
    let join = thread::spawn(move || {
        let mut last_snapshot = Instant::now();
        let mut last_ops = 0u64;
        let check_every = if config.check_every.is_zero() {
            Duration::from_secs(1)
        } else {
            config.check_every
        };

        loop {
            if stop_thread.load(Ordering::Relaxed) {
                break;
            }

            let mut snapshot = None;
            let ops_now;
            {
                let guard = match segment.read() {
                    Ok(guard) => guard,
                    Err(e) => {
                        log::warn!(target: "segment::snapshot", "snapshotter lock poisoned: {}", e);
                        break;
                    }
                };
                ops_now = guard.op_count();
                let time_due = !config.interval.is_zero() && last_snapshot.elapsed() >= config.interval;
                let ops_due = config.max_ops > 0 && ops_now.saturating_sub(last_ops) >= config.max_ops;
                if time_due || ops_due {
                    snapshot = Some(guard.build_snapshot());
                }
            }

            if let Some(snapshot) = snapshot {
                let res = Segment::persist_snapshot(&snapshot, &config.path);
                match res {
                    Ok(()) => {
                        last_snapshot = Instant::now();
                        last_ops = ops_now;
                    }
                    Err(DBError::SerializationError(err)) => {
                        log::warn!(
                            target: "segment::snapshot",
                            "snapshot serialization failed: {}",
                            err
                        );
                    }
                    Err(err) => {
                        log::warn!(target: "segment::snapshot", "snapshot write failed: {}", err);
                    }
                }
            }

            thread::sleep(check_every);
        }
    });

    SnapshotterHandle {
        stop,
        join: Some(join),
    }
}
