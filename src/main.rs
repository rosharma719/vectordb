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
    pub mod filters; // ✅ Add this line

    pub use segment::run_segment_tests;
    pub use filters::run_filter_tests; // ✅ Add this line
}

fn main() {
    tests::utilsvector::run_utilsvector_tests();
    tests::payload::run_payload_tests();
    tests::inverted_index::run_inverted_index_tests();
    tests::hnsw::run_hnsw_tests();
    tests::run_segment_tests();
    tests::run_filter_tests(); // ✅ Add this line
}
