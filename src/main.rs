mod utils;
mod vector;
pub mod payload_storage;

mod tests {
    pub mod inverted_index;
    pub mod utilsvector;
    pub mod payload;
}

fn main() {
    tests::utilsvector::run_utilsvector_tests();
    tests::payload::run_payload_tests();
    tests::inverted_index::run_inverted_index_tests();
}
