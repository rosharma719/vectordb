#!/usr/bin/env bash
set -euo pipefail

path="${1:-data/nytimes-256-angular/index.bin}"
cargo run --quiet --bin snapshot_info -- "${path}"
