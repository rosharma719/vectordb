mod utils;
mod vector;

mod tests;

use tests::utilsvector::run_utilsvector_tests;

fn main() {
    println!("Running internal tests from main...");
    run_utilsvector_tests();
}
