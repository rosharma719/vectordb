use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use std::path::PathBuf;

use rand::Rng;

use vectordb::payload_storage::filters::Filter;
use vectordb::segment::{start_background_snapshots, Segment, SnapshotConfig};
use vectordb::utils::errors::DBError;
use vectordb::utils::payload::{Payload, PayloadValue};
use vectordb::utils::types::{DistanceMetric, Vector};
use vectordb::vector::hnsw::HNSWIndex;

fn tmp_path(prefix: &str) -> PathBuf {
    let mut rng = rand::rng();
    std::env::temp_dir().join(format!("{prefix}_{}.bin", rng.random::<u64>()))
}

fn vecf(v: &[f32]) -> Vector {
    v.to_vec()
}

#[test]
fn hnsw_round_trip_preserves_results() -> Result<(), DBError> {
    let mut hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 16, 32, 8, 3);
    for i in 0..50u64 {
        hnsw.insert(i, vecf(&[i as f32, (i * 2) as f32, 1.0]))?;
    }

    let query = vecf(&[5.0, 10.0, 1.0]);
    let before: Vec<_> = hnsw.search(&query, 5)?.into_iter().map(|sp| sp.id).collect();

    let path = tmp_path("hnsw_snapshot");
    println!("[hnsw] saving to {:?}", path);
    hnsw.save_to_path(&path)?;
    let restored = HNSWIndex::load_from_path(&path)?;
    let after: Vec<_> = restored.search(&query, 5)?.into_iter().map(|sp| sp.id).collect();
    println!("[hnsw] before={:?} after={:?}", before, after);
    let _ = fs::remove_file(path);

    assert_eq!(before, after);
    Ok(())
}

#[test]
fn segment_round_trip_preserves_payload_filtered_results() -> Result<(), DBError> {
    let mut seg = Segment::new(HNSWIndex::new(DistanceMetric::Euclidean, 16, 32, 8, 2));
    for i in 0..20u64 {
        let mut payload = Payload(HashMap::new());
        let group = if i % 2 == 0 { "even" } else { "odd" };
        payload.set("group", PayloadValue::Str(group.to_string()));
        seg.insert_with_id(i, vecf(&[i as f32, 0.0]), Some(payload))?;
    }

    let filter = Filter::Match {
        key: "group".into(),
        value: PayloadValue::Str("even".into()),
    };
    let query = vecf(&[1.0, 0.0]);
    let mut before: Vec<_> = seg
        .search_with_filter(&query, 5, Some(&filter))?
        .into_iter()
        .map(|sp| sp.id)
        .collect();
    before.sort();

    let path = tmp_path("segment_snapshot");
    println!("[segment-small] saving to {:?}", path);
    seg.save_to_path(&path)?;
    let restored = Segment::load_from_path(&path)?;
    let mut after: Vec<_> = restored
        .search_with_filter(&query, 5, Some(&filter))?
        .into_iter()
        .map(|sp| sp.id)
        .collect();
    after.sort();
    println!("[segment-small] before={:?} after={:?}", before, after);
    let _ = fs::remove_file(path);

    assert_eq!(before, after);
    Ok(())
}

#[test]
fn segment_round_trip_large_dataset() -> Result<(), DBError> {
    // Build a 20k-point deterministic dataset with even/odd payloads.
    let dim = 4;
    let size = 20_000u64;
    let mut seg = Segment::new(HNSWIndex::new(DistanceMetric::Euclidean, 16, 64, 16, dim));
    seg.hnsw_mut().set_ef_construct(100);
    seg.hnsw_mut().set_ef_search(128);

    let mut payload = Payload(HashMap::new());
    for i in 0..size {
        payload.0.clear();
        let group = if i % 2 == 0 { "even" } else { "odd" };
        payload.set("group", PayloadValue::Str(group.to_string()));
        let vec = vec![
            i as f32,
            (i % 10) as f32,
            (i % 100) as f32,
            (i % 7) as f32,
        ];
        seg.insert_with_id(i, vec, Some(payload.clone()))?;
    }

    let filter = Filter::Match {
        key: "group".into(),
        value: PayloadValue::Str("even".into()),
    };
    let query = vecf(&[0.0, 0.0, 0.0, 0.0]);
    let mut before: Vec<_> = seg
        .search_with_filter(&query, 10, Some(&filter))?
        .into_iter()
        .map(|sp| sp.id)
        .collect();
    before.sort();

    let path = tmp_path("segment_snapshot_large");
    println!(
        "[segment-large] saving {} vectors (payloads={}) to {:?}",
        size,
        seg.payloads().len(),
        path
    );
    seg.save_to_path(&path)?;
    let restored = Segment::load_from_path(&path)?;
    let mut after: Vec<_> = restored
        .search_with_filter(&query, 10, Some(&filter))?
        .into_iter()
        .map(|sp| sp.id)
        .collect();
    after.sort();
    println!(
        "[segment-large] before={:?} after={:?} restored_payloads={}",
        &before[..before.len().min(10)],
        &after[..after.len().min(10)],
        restored.payloads().len()
    );
    let _ = fs::remove_file(path);

    assert_eq!(before, after);
    assert_eq!(restored.payloads().len(), size as usize);
    Ok(())
}

#[test]
fn segment_persist_append_and_reload() -> Result<(), DBError> {
    let dim = 4;
    let initial = 10_000u64;
    let extra = 2_000u64;

    let mut seg = Segment::new(HNSWIndex::new(DistanceMetric::Euclidean, 16, 64, 16, dim));
    seg.hnsw_mut().set_ef_construct(80);
    seg.hnsw_mut().set_ef_search(128);

    // Insert initial batch.
    for i in 0..initial {
        let vec = vec![i as f32, (i % 10) as f32, (i % 7) as f32, 1.0];
        seg.insert_with_id(i, vec, None)?;
    }

    // Save snapshot after initial insert.
    let path1 = tmp_path("segment_append_stage1");
    seg.save_to_path(&path1)?;
    println!("[append] saved first snapshot to {:?}", path1);

    // Reload and append more vectors.
    let mut seg = Segment::load_from_path(&path1)?;
    for i in initial..(initial + extra) {
        let vec = vec![i as f32, (i % 10) as f32, (i % 7) as f32, 1.0];
        seg.insert_with_id(i, vec, None)?;
    }

    // Save combined snapshot.
    let path2 = tmp_path("segment_append_stage2");
    seg.save_to_path(&path2)?;
    println!("[append] saved second snapshot to {:?}", path2);

    // Reload final snapshot and validate all IDs are present and searchable.
    let seg = Segment::load_from_path(&path2)?;
    assert_eq!(seg.hnsw().len(), (initial + extra) as usize);

    let sample_old = 1234u64;
    let sample_new = initial + 42;
    let query_old = vec![sample_old as f32, (sample_old % 10) as f32, (sample_old % 7) as f32, 1.0];
    let query_new = vec![sample_new as f32, (sample_new % 10) as f32, (sample_new % 7) as f32, 1.0];

    let res_old = seg.search(&query_old, 1)?.first().map(|sp| sp.id);
    let res_new = seg.search(&query_new, 1)?.first().map(|sp| sp.id);
    assert_eq!(res_old, Some(sample_old));
    assert_eq!(res_new, Some(sample_new));

    let _ = fs::remove_file(path1);
    let _ = fs::remove_file(path2);
    Ok(())
}

#[test]
fn hnsw_snapshot_checksum_detects_corruption() -> Result<(), DBError> {
    let mut hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 16, 32, 8, 3);
    for i in 0..10u64 {
        hnsw.insert(i, vecf(&[i as f32, (i * 2) as f32, 1.0]))?;
    }

    let path = tmp_path("hnsw_snapshot_corrupt");
    hnsw.save_to_path(&path)?;

    let mut bytes = fs::read(&path)?;
    assert!(!bytes.is_empty());
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    fs::write(&path, bytes)?;

    let res = HNSWIndex::load_from_path(&path);
    let _ = fs::remove_file(path);
    assert!(res.is_err());
    Ok(())
}

#[test]
fn segment_snapshot_checksum_detects_corruption() -> Result<(), DBError> {
    let mut seg = Segment::new(HNSWIndex::new(DistanceMetric::Euclidean, 16, 32, 8, 2));
    for i in 0..10u64 {
        seg.insert_with_id(i, vecf(&[i as f32, 0.0]), None)?;
    }

    let path = tmp_path("segment_snapshot_corrupt");
    seg.save_to_path(&path)?;

    let mut bytes = fs::read(&path)?;
    assert!(!bytes.is_empty());
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    fs::write(&path, bytes)?;

    let res = Segment::load_from_path(&path);
    let _ = fs::remove_file(path);
    assert!(res.is_err());
    Ok(())
}

#[test]
fn background_snapshot_writes_file() -> Result<(), DBError> {
    let path = tmp_path("segment_background_snapshot");
    let segment = Arc::new(RwLock::new(Segment::new(HNSWIndex::new(
        DistanceMetric::Euclidean,
        16,
        32,
        8,
        2,
    ))));

    let mut config = SnapshotConfig::new(&path);
    config.interval = Duration::from_millis(50);
    config.max_ops = 1;
    config.check_every = Duration::from_millis(10);

    let handle = start_background_snapshots(segment.clone(), config);

    {
        let mut guard = segment.write().unwrap();
        guard.insert_with_id(1, vecf(&[1.0, 0.0]), None)?;
    }

    let start = Instant::now();
    while !path.exists() && start.elapsed() < Duration::from_secs(2) {
        std::thread::sleep(Duration::from_millis(10));
    }

    handle.stop();

    assert!(path.exists(), "snapshot file was not created");
    let restored = Segment::load_from_path(&path)?;
    assert_eq!(restored.hnsw().len(), 1);
    let _ = fs::remove_file(path);
    Ok(())
}

#[test]
fn background_snapshot_handles_concurrent_writes() -> Result<(), DBError> {
    let path = tmp_path("segment_background_concurrent");
    let segment = Arc::new(RwLock::new(Segment::new(HNSWIndex::new(
        DistanceMetric::Euclidean,
        16,
        32,
        8,
        2,
    ))));

    let mut config = SnapshotConfig::new(&path);
    config.interval = Duration::from_millis(100);
    config.max_ops = 25;
    config.check_every = Duration::from_millis(10);
    let handle = start_background_snapshots(segment.clone(), config);

    let writer = {
        let segment = segment.clone();
        std::thread::spawn(move || {
            for i in 0..200u64 {
                let mut guard = segment.write().unwrap();
                let _ = guard.insert_with_id(i, vecf(&[i as f32, 0.0]), None);
            }
        })
    };

    writer.join().expect("writer thread panicked");
    let start = Instant::now();
    while !path.exists() && start.elapsed() < Duration::from_secs(2) {
        std::thread::sleep(Duration::from_millis(10));
    }

    handle.stop();
    assert!(path.exists(), "snapshot file was not created");

    let restored = Segment::load_from_path(&path)?;
    let final_len = segment.read().unwrap().hnsw().len();
    assert!(restored.hnsw().len() > 0);
    assert!(restored.hnsw().len() <= final_len);

    let _ = fs::remove_file(path);
    Ok(())
}

#[test]
fn background_snapshot_updates_over_time() -> Result<(), DBError> {
    let path = tmp_path("segment_background_updates");
    let segment = Arc::new(RwLock::new(Segment::new(HNSWIndex::new(
        DistanceMetric::Euclidean,
        16,
        32,
        8,
        2,
    ))));

    let mut config = SnapshotConfig::new(&path);
    config.interval = Duration::from_millis(200);
    config.max_ops = 5;
    config.check_every = Duration::from_millis(10);
    let handle = start_background_snapshots(segment.clone(), config);

    {
        let mut guard = segment.write().unwrap();
        for i in 0..10u64 {
            guard.insert_with_id(i, vecf(&[i as f32, 0.0]), None)?;
        }
    }

    let start = Instant::now();
    while !path.exists() && start.elapsed() < Duration::from_secs(2) {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(path.exists(), "snapshot file was not created");
    let first_size = fs::metadata(&path)?.len();

    {
        let mut guard = segment.write().unwrap();
        for i in 10..40u64 {
            guard.insert_with_id(i, vecf(&[i as f32, 0.0]), None)?;
        }
    }

    let start = Instant::now();
    let mut grew = false;
    while start.elapsed() < Duration::from_secs(3) {
        let size = fs::metadata(&path)?.len();
        if size > first_size {
            grew = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    handle.stop();
    assert!(grew, "snapshot file size did not grow after more inserts");
    let _ = fs::remove_file(path);
    Ok(())
}

#[test]
fn background_snapshot_stop_is_clean() -> Result<(), DBError> {
    let path = tmp_path("segment_background_stop");
    let segment = Arc::new(RwLock::new(Segment::new(HNSWIndex::new(
        DistanceMetric::Euclidean,
        16,
        32,
        8,
        2,
    ))));

    let mut config = SnapshotConfig::new(&path);
    config.interval = Duration::from_millis(50);
    config.max_ops = 1;
    config.check_every = Duration::from_millis(10);
    let handle = start_background_snapshots(segment.clone(), config);

    {
        let mut guard = segment.write().unwrap();
        guard.insert_with_id(1, vecf(&[1.0, 0.0]), None)?;
    }

    let start = Instant::now();
    while !path.exists() && start.elapsed() < Duration::from_secs(2) {
        std::thread::sleep(Duration::from_millis(10));
    }

    handle.stop();
    assert!(path.exists(), "snapshot file was not created");
    Segment::load_from_path(&path)?;
    let _ = fs::remove_file(path);
    Ok(())
}

#[test]
fn background_snapshot_respects_deletes() -> Result<(), DBError> {
    let path = tmp_path("segment_background_deletes");
    let segment = Arc::new(RwLock::new(Segment::new(HNSWIndex::new(
        DistanceMetric::Euclidean,
        16,
        32,
        8,
        2,
    ))));

    let mut config = SnapshotConfig::new(&path);
    config.interval = Duration::from_millis(100);
    config.max_ops = 1;
    config.check_every = Duration::from_millis(10);
    let handle = start_background_snapshots(segment.clone(), config);

    {
        let mut guard = segment.write().unwrap();
        for i in 0..10u64 {
            guard.insert_with_id(i, vecf(&[i as f32, 0.0]), None)?;
        }
        guard.delete(2)?;
        guard.delete(7)?;
    }

    let start = Instant::now();
    while !path.exists() && start.elapsed() < Duration::from_secs(2) {
        std::thread::sleep(Duration::from_millis(10));
    }

    handle.stop();
    let restored = Segment::load_from_path(&path)?;
    assert!(restored.get_vector(2).is_none());
    assert!(restored.get_vector(7).is_none());
    assert!(restored.get_vector(1).is_some());

    let _ = fs::remove_file(path);
    Ok(())
}

#[test]
fn wal_replay_applies_post_snapshot_ops() -> Result<(), DBError> {
    let snapshot_path = tmp_path("segment_wal_snapshot");
    let wal_path = Segment::wal_path_for_snapshot(&snapshot_path);

    let mut seg = Segment::new(HNSWIndex::new(DistanceMetric::Euclidean, 16, 32, 8, 2));
    seg.enable_wal(&wal_path)?;

    let mut payload = Payload(HashMap::new());
    payload.set("tag", PayloadValue::Str("v1".to_string()));
    seg.insert_with_id(1, vecf(&[1.0, 0.0]), Some(payload.clone()))?;

    seg.save_to_path(&snapshot_path)?;

    payload.set("tag", PayloadValue::Str("v2".to_string()));
    seg.update_payload(1, payload.clone())?;
    seg.insert_with_id(2, vecf(&[2.0, 0.0]), None)?;
    seg.delete(2)?;

    let restored = Segment::load_from_path(&snapshot_path)?;
    assert!(restored.get_vector(2).is_none());
    assert_eq!(restored.get_payload(1), Some(&payload));

    let _ = fs::remove_file(snapshot_path);
    let _ = fs::remove_file(wal_path);
    Ok(())
}

#[test]
fn wal_checkpoint_truncates_log() -> Result<(), DBError> {
    let snapshot_path = tmp_path("segment_wal_checkpoint");
    let wal_path = Segment::wal_path_for_snapshot(&snapshot_path);

    let mut seg = Segment::new(HNSWIndex::new(DistanceMetric::Euclidean, 16, 32, 8, 2));
    seg.enable_wal(&wal_path)?;
    seg.insert_with_id(1, vecf(&[1.0, 0.0]), None)?;
    assert!(fs::metadata(&wal_path)?.len() > 0);

    seg.save_to_path_and_checkpoint(&snapshot_path)?;
    let size = fs::metadata(&wal_path)?.len();
    assert!(size > 0, "wal header should remain after truncate");

    let mut seg2 = Segment::new(HNSWIndex::new(DistanceMetric::Euclidean, 16, 32, 8, 2));
    seg2.enable_wal(&wal_path)?;
    seg2.insert_with_id(2, vecf(&[2.0, 0.0]), None)?;

    let restored = Segment::load_from_path(&snapshot_path)?;
    assert!(restored.get_vector(1).is_some());
    assert!(restored.get_vector(2).is_some());

    let _ = fs::remove_file(snapshot_path);
    let _ = fs::remove_file(wal_path);
    Ok(())
}

#[test]
fn wal_auto_replay_can_be_disabled() -> Result<(), DBError> {
    let snapshot_path = tmp_path("segment_wal_no_replay");
    let wal_path = Segment::wal_path_for_snapshot(&snapshot_path);

    let mut seg = Segment::new(HNSWIndex::new(DistanceMetric::Euclidean, 16, 32, 8, 2));
    seg.enable_wal(&wal_path)?;
    seg.insert_with_id(1, vecf(&[1.0, 0.0]), None)?;
    seg.save_to_path(&snapshot_path)?;
    seg.insert_with_id(2, vecf(&[2.0, 0.0]), None)?;

    unsafe {
        std::env::set_var("VECTORDB_WAL_AUTO_REPLAY", "0");
    }
    let restored = Segment::load_from_path(&snapshot_path)?;
    unsafe {
        std::env::remove_var("VECTORDB_WAL_AUTO_REPLAY");
    }

    assert!(restored.get_vector(1).is_some());
    assert!(restored.get_vector(2).is_none());

    let _ = fs::remove_file(snapshot_path);
    let _ = fs::remove_file(wal_path);
    Ok(())
}

#[test]
fn wal_corruption_is_detected() -> Result<(), DBError> {
    let snapshot_path = tmp_path("segment_wal_corrupt_snapshot");
    let wal_path = Segment::wal_path_for_snapshot(&snapshot_path);

    let mut seg = Segment::new(HNSWIndex::new(DistanceMetric::Euclidean, 16, 32, 8, 2));
    seg.enable_wal(&wal_path)?;
    seg.insert_with_id(1, vecf(&[1.0, 0.0]), None)?;
    seg.save_to_path(&snapshot_path)?;
    seg.insert_with_id(2, vecf(&[2.0, 0.0]), None)?;

    let mut bytes = fs::read(&wal_path)?;
    assert!(bytes.len() > 12, "wal too small to corrupt");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    fs::write(&wal_path, bytes)?;

    let res = Segment::load_from_path(&snapshot_path);
    assert!(res.is_err());

    let _ = fs::remove_file(snapshot_path);
    let _ = fs::remove_file(wal_path);
    Ok(())
}

#[test]
fn wal_and_background_snapshot_together() -> Result<(), DBError> {
    let snapshot_path = tmp_path("segment_wal_background_snapshot");
    let wal_path = Segment::wal_path_for_snapshot(&snapshot_path);

    let segment = Arc::new(RwLock::new(Segment::new(HNSWIndex::new(
        DistanceMetric::Euclidean,
        16,
        32,
        8,
        2,
    ))));
    {
        let mut guard = segment.write().unwrap();
        guard.enable_wal(&wal_path)?;
    }

    let mut config = SnapshotConfig::new(&snapshot_path);
    config.interval = Duration::from_millis(100);
    config.max_ops = 10;
    config.check_every = Duration::from_millis(10);
    let handle = start_background_snapshots(segment.clone(), config);

    {
        let mut guard = segment.write().unwrap();
        for i in 0..50u64 {
            guard.insert_with_id(i, vecf(&[i as f32, 0.0]), None)?;
        }
    }

    let start = Instant::now();
    while !snapshot_path.exists() && start.elapsed() < Duration::from_secs(2) {
        std::thread::sleep(Duration::from_millis(10));
    }
    handle.stop();

    let restored = Segment::load_from_path(&snapshot_path)?;
    let len = restored.hnsw().len();
    assert!(len > 0);

    let _ = fs::remove_file(snapshot_path);
    let _ = fs::remove_file(wal_path);
    Ok(())
}

#[test]
fn wal_with_deletes_and_purge_replay() -> Result<(), DBError> {
    let snapshot_path = tmp_path("segment_wal_purge_snapshot");
    let wal_path = Segment::wal_path_for_snapshot(&snapshot_path);

    let mut seg = Segment::new(HNSWIndex::new(DistanceMetric::Euclidean, 16, 32, 8, 2));
    seg.enable_wal(&wal_path)?;

    for i in 0..500u64 {
        seg.insert_with_id(i, vecf(&[i as f32, 0.0]), None)?;
    }
    seg.save_to_path(&snapshot_path)?;

    for i in 0..200u64 {
        seg.delete(i)?;
    }

    let restored = Segment::load_from_path(&snapshot_path)?;
    assert!(restored.get_vector(0).is_none());
    assert!(restored.get_vector(199).is_none());
    assert!(restored.get_vector(300).is_some());

    let _ = fs::remove_file(snapshot_path);
    let _ = fs::remove_file(wal_path);
    Ok(())
}
