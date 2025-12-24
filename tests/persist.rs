use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use rand::Rng;

use vectordb::payload_storage::filters::Filter;
use vectordb::segment::Segment;
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
