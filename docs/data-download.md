# Dataset Downloads (CLI-Friendly)

## H&M (2048-D Cosine, Filtered)
```bash
mkdir -p data/hnm
curl -L https://storage.googleapis.com/ann-filtered-benchmark/datasets/hnm.tgz \
  | tar -xz -C data/hnm --strip-components=1
ls data/hnm
# Expect: vectors.npy, payloads.jsonl, tests.jsonl
```

## NYTimes (256-D Angular)
Requires an `HF_TOKEN` with read access to `open-vdb/nytimes-256-angular`.
```bash
export HF_TOKEN=your_hf_token
python - <<'PY'
import os, json, numpy as np, pathlib
from datasets import load_dataset

token = os.environ["HF_TOKEN"]
train = load_dataset("open-vdb/nytimes-256-angular", name="train", split="train", token=token)
test = load_dataset("open-vdb/nytimes-256-angular", name="test", split="test", token=token)
nbrs = load_dataset("open-vdb/nytimes-256-angular", name="neighbors", split="neighbors", token=token)

def first_list_col(ds):
    for name in ds.column_names:
        if isinstance(ds[0][name], (list, tuple)):
            return name
    raise RuntimeError(f"no list-like column in {ds.column_names}")

emb_col = first_list_col(train)
q_col = first_list_col(test)
nbr_col = first_list_col(nbrs)

out = pathlib.Path("data/nytimes-256-angular"); out.mkdir(parents=True, exist_ok=True)
np.save(out/"base.npy", np.stack(train[emb_col]).astype("float32"))
np.save(out/"queries.npy", np.stack(test[q_col]).astype("float32"))
neighbors_list = nbrs[nbr_col].to_pylist() if hasattr(nbrs[nbr_col], "to_pylist") else list(nbrs[nbr_col])
with open(out/"ground_truth.json","w") as f:
    json.dump(neighbors_list, f)
print("wrote", out, "cols:", emb_col, q_col, nbr_col)
PY
```
