#!/usr/bin/env bash
set -euo pipefail

# Sweep early-exit patience (0-5) and search expansion multiplier (1-3)
# for the NYTimes recall-from-snapshot test, writing full output to a log.

SNAPSHOT_PATH=${1:-data/nytimes-256-angular/index.bin}
LOG_FILE=${LOG_FILE:-logs/nyt_patience_mult_sweep.log}
EF_LIST="32,64,128,256,512"

if [[ ! -f "${SNAPSHOT_PATH}" ]]; then
  echo "Snapshot not found at ${SNAPSHOT_PATH}. Provide the path as the first arg." >&2
  exit 1
fi

mkdir -p "$(dirname "${LOG_FILE}")"

timestamp() {
  date -u "+%Y-%m-%dT%H:%M:%SZ"
}

run_config() {
  local patience=$1
  local mult=$2
  local started
  started=$(timestamp)
  echo "=== RUN patience=${patience} mult=${mult} ef_search_list=${EF_LIST} snapshot=${SNAPSHOT_PATH} start=${started}" | tee -a "${LOG_FILE}"

  # Pipe through tee for both console and log; capture exit via PIPESTATUS.
  set +e
  VECTORDB_NYT_PERSIST_PATH="${SNAPSHOT_PATH}" \
  VECTORDB_EARLY_EXIT_PATIENCE="${patience}" \
  VECTORDB_DISABLE_EARLY_EXIT=0 \
  VECTORDB_SEARCH_EXPANSION_MULT="${mult}" \
  VECTORDB_EF_SEARCH_LIST="${EF_LIST}" \
  cargo test --release nytimes_recall_from_snapshot -- --ignored --nocapture | tee -a "${LOG_FILE}"
  status=${PIPESTATUS[0]}
  set -e

  local ended
  ended=$(timestamp)
  if [[ ${status} -ne 0 ]]; then
    echo "=== FAIL patience=${patience} mult=${mult} status=${status} end=${ended}" | tee -a "${LOG_FILE}"
    exit ${status}
  else
    echo "=== DONE patience=${patience} mult=${mult} end=${ended}" | tee -a "${LOG_FILE}"
  fi
}

echo "===== NYT patience/mult sweep start $(timestamp) snapshot=${SNAPSHOT_PATH} log=${LOG_FILE} =====" | tee -a "${LOG_FILE}"
for patience in 0 1 2 3 4 5; do
  for mult in 1 2 3; do
    run_config "${patience}" "${mult}"
  done
done
echo "===== NYT patience/mult sweep end $(timestamp) =====" | tee -a "${LOG_FILE}"
