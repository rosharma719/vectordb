use std::env;
use std::fs;
use std::path::Path;

use vectordb::segment::Segment;

fn main() {
    let path = match env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: snapshot_info <path>");
            std::process::exit(2);
        }
    };
    let path_ref = Path::new(&path);
    let meta = fs::metadata(path_ref).ok();
    let size = meta.map(|m| m.len()).unwrap_or(0);

    let segment = match Segment::load_from_path(path_ref) {
        Ok(seg) => seg,
        Err(err) => {
            eprintln!("failed to load snapshot: {err}");
            std::process::exit(1);
        }
    };

    println!("snapshot_path={}", path_ref.display());
    println!("file_size_bytes={}", size);
    println!("points={}", segment.hnsw().len());
    println!("payloads={}", segment.payloads().len());
}
