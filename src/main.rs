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
    pub mod filters;
    pub mod in_place;

    pub use segment::run_segment_tests;
    pub use filters::run_filter_tests;
    pub use in_place::run_in_place_tests;
}

fn main() {
    tests::utilsvector::run_utilsvector_tests();
    tests::payload::run_payload_tests();
    tests::inverted_index::run_inverted_index_tests();
    tests::hnsw::run_hnsw_tests();
    tests::run_segment_tests();
    tests::run_filter_tests();
    tests::run_in_place_tests();
}
