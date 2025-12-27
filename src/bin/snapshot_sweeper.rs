use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::to_writer;

use vectordb::analysis::{AnalyzerConfig, analyze_snapshot};

const DEFAULT_OUTPUT: &str = "nyt_analysis_runner.jsonl";
const DEFAULT_BASE: &str = "data/nytimes-256-angular/base.npy";
const DEFAULT_QUERIES: &str = "data/nytimes-256-angular/queries.npy";
const DEFAULT_SNAPSHOT_DIR: &str = "data/nytimes-256-angular";
const DEFAULT_SNAPSHOT_NAMES: &[&str] = &[
    "index_m16_ef100.bin",
    "index_m16_ef100_a13.bin",
    "index_m16_m0_16_efc100.bin",
    "index_m16_m0_16_efc100_cap128.bin",
    "index_m16_m0_16_efc200.bin",
    "index_m16_m0_16_efc400_levelscale0.5.bin",
    "index_m16_m0_32_efc100.bin",
    "index_m16_m0_32_efc400.bin",
    "index_m32_ef200.bin",
    "index_m32_m0_32_efc100.bin",
    "index_m48_ef200_a12_l24.bin",
    "index_m48_m0_48_efc100.bin",
];

fn main() -> Result<()> {
    let cfg = SweeperConfig::from_args()?;
    let writer = File::create(&cfg.output)
        .with_context(|| format!("failed to create {}", cfg.output.display()))?;
    let mut writer = BufWriter::new(writer);

    for snapshot in &cfg.snapshots {
        let analysis = AnalyzerConfig::from_paths(
            snapshot.clone(),
            Some(cfg.base.clone()),
            Some(cfg.queries.clone()),
            cfg.top_k,
            cfg.num_queries,
            cfg.sample_size,
            cfg.neighbor_scan_cap,
        );
        let stats = analyze_snapshot(analysis)?;
        to_writer(&mut writer, &stats)?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

struct SweeperConfig {
    snapshots: Vec<PathBuf>,
    output: PathBuf,
    base: PathBuf,
    queries: PathBuf,
    top_k: usize,
    num_queries: usize,
    sample_size: usize,
    neighbor_scan_cap: usize,
}

impl SweeperConfig {
    fn from_args() -> Result<Self> {
        let mut snapshots = Vec::new();
        let mut manifest = None;
        let mut output = PathBuf::from(DEFAULT_OUTPUT);
        let mut base = PathBuf::from(DEFAULT_BASE);
        let mut queries = PathBuf::from(DEFAULT_QUERIES);
        let mut top_k = 10;
        let mut num_queries = 100;
        let mut sample_size = 500;
        let mut neighbor_scan_cap = 128;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--snapshot" => {
                    let path =
                        PathBuf::from(args.next().context("expected path after --snapshot")?);
                    snapshots.push(path);
                }
                "--snapshots" => {
                    let list = args
                        .next()
                        .context("expected comma-separated list after --snapshots")?;
                    for entry in list.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                        snapshots.push(PathBuf::from(entry));
                    }
                }
                "--manifest" => {
                    manifest = Some(PathBuf::from(
                        args.next().context("expected path after --manifest")?,
                    ));
                }
                "--output" => {
                    output = PathBuf::from(args.next().context("expected path after --output")?);
                }
                "--base" => {
                    base = PathBuf::from(args.next().context("expected path after --base")?);
                }
                "--queries" => {
                    queries = PathBuf::from(args.next().context("expected path after --queries")?);
                }
                "--top-k" => {
                    top_k = args
                        .next()
                        .context("expected usize after --top-k")?
                        .parse()
                        .context("failed to parse top-k")?;
                }
                "--num-queries" => {
                    num_queries = args
                        .next()
                        .context("expected usize after --num-queries")?
                        .parse()
                        .context("failed to parse num-queries")?;
                }
                "--sample-size" => {
                    sample_size = args
                        .next()
                        .context("expected usize after --sample-size")?
                        .parse()
                        .context("failed to parse sample-size")?;
                }
                "--neighbor-scan-cap" => {
                    neighbor_scan_cap = args
                        .next()
                        .context("expected usize after --neighbor-scan-cap")?
                        .parse()
                        .context("failed to parse neighbor scan cap")?;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    bail!("unknown option {}", other);
                }
            }
        }

        if let Some(path) = manifest {
            let file = File::open(&path)
                .with_context(|| format!("failed to open manifest {}", path.display()))?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    continue;
                }
                snapshots.push(PathBuf::from(trimmed));
            }
        }

        if snapshots.is_empty() {
            snapshots = DEFAULT_SNAPSHOT_NAMES
                .iter()
                .map(|name| Path::new(DEFAULT_SNAPSHOT_DIR).join(name))
                .collect();
        }

        Ok(SweeperConfig {
            snapshots,
            output,
            base,
            queries,
            top_k,
            num_queries,
            sample_size,
            neighbor_scan_cap,
        })
    }
}

fn print_help() {
    println!(
        "\
Usage:
  cargo run --bin snapshot_sweeper -- [options]

Options:
  --snapshot <path>       Process a single snapshot (can be repeated)
  --snapshots <list>      Comma-separated list of snapshot paths
  --manifest <path>       File containing newline snapshot paths (comments with #)
  --base <path>           Base `.npy` (default data/nytimes-256-angular/base.npy)
  --queries <path>        Query `.npy` (default data/nytimes-256-angular/queries.npy)
  --top-k <usize>         Number of top candidates per query (default 10)
  --num-queries <usize>   Number of queries to run (default 100)
  --sample-size <usize>   Dataset sample size for distance stats (default 500)
  --neighbor-scan-cap <usize> Cap used when computing level 0 tail ratios (default 128)
  --output <path>         JSONL output file (default nyt_analysis_runner.jsonl)
  --help, -h              Print this help message
"
    );
}
