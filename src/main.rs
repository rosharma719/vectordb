mod utils;
mod vector;
pub mod payload_storage;
pub mod segment;

mod tests {
    pub mod inverted_index;
    pub mod utilsvector;
    pub mod payload;
    pub mod hnsw;
    pub mod segment;

    pub use segment::run_segment_tests; // ✅ Import the function for clarity
}

fn main() {
    tests::utilsvector::run_utilsvector_tests();
    tests::payload::run_payload_tests();
    tests::inverted_index::run_inverted_index_tests();
    tests::hnsw::run_hnsw_tests();
    tests::run_segment_tests(); // ✅ Fixed function call
}
