# Experiment Matrix

This release exposes the paper-facing configurations through
`configs/release_experiments.json` and `test/run_release_experiment.py`.

The launcher maps one benchmark and one method preset to a reproducible command.
For larger sweeps, invoke the same command with the desired benchmark and method
preset combinations.

## Required Environment

Set OpenAI-compatible endpoints and model names with environment variables:

```bash
export SELECTOR_URL="http://host:port/v1"
export SELECTOR_MODEL="selector-model-name"
export REASONING_URL="http://host:port/v1"
export REASONING_MODEL="reasoning-model-name"
export SUPERVISOR_URL="http://host:port/v1"
export SUPERVISOR_MODEL="auditor-model-name"
export EMBEDDING_URL="http://host:port/v1"
export EMBEDDING_MODEL="embedding-model-name"

# Optional, defaults to EMPTY for local endpoints without auth.
export SELECTOR_KEY="EMPTY"
export REASONING_KEY="EMPTY"
export SUPERVISOR_KEY="EMPTY"
export EMBEDDING_KEY="EMPTY"
```

For optional pool overrides or externally hosted embedding caches, also set:

```bash
export AGENTDROPOUT_METRIC_POOL_FILE="/path/to/pool.json"
export AGENTDROPOUT_EMBEDDING_CACHE_FILE="/path/to/pool_embeddings.jsonl"
```

LiveCodeBench data is not bundled in this release. Provide a local jsonl file:

```bash
export AGENTDROPOUT_LIVECODE_FILE="/path/to/livecode.jsonl"
```

## Examples

List available benchmarks and method presets:

```bash
python test/run_release_experiment.py --list
```

Dry-run the main math configuration:

```bash
python test/run_release_experiment.py \
  --benchmark gsm8k \
  --method adv2_math_main \
  --model-profile math_8b \
  --dry-run
```

Dry-run mode prints endpoint placeholders when endpoint environment variables
are unset.

Run a small smoke subset:

```bash
python test/run_release_experiment.py \
  --benchmark gsm8k \
  --method adv2_math_main \
  --model-profile math_8b \
  --limit 2
```

Legacy `test/run-*.sh` scripts are thin wrappers over this launcher:

```bash
bash test/run-gsm8k.sh --method adv2_math_main --model-profile math_8b --limit 2
```

## Method Presets

| Preset | Paper role | Main arguments |
| --- | --- | --- |
| `autogen_baseline` | Dynamic-MAS / AutoGen baseline | `--baseline_only` |
| `adv2_math_main` | Main math row | `--retrieval_mode rerank --retrieve_p 20 --select_q 5 --batch_audit_metrics --pass_rate 0.6 --retries_times 3` |
| `adv2_math_iter2` | Table 4 iteration ablation | Same as main, `--retries_times 2` |
| `adv2_math_iter4` | Table 4 iteration ablation | Same as main, `--retries_times 4` |
| `adv2_math_top3` | Table 4 retrieved-indicator ablation | Same as main, `--select_q 3` |
| `adv2_math_top7` | Table 4 retrieved-indicator ablation | Same as main, `--select_q 7` |
| `adv2_math_pass_2of5` | Table 4 pass-threshold ablation | Same as main, `--pass_rate 0.4` |
| `adv2_math_pass_5of5` | Table 4 pass-threshold ablation | Same as main, `--pass_rate 1.0` |
| `adv2_math_nondedup_pool` | Table 4 pool-deduplication ablation | Same as main, with an externally supplied non-deduplicated pool |
| `adv2_math_random_1to5` | Table 4 retrieval-control ablation | `--retrieval_mode random --random_k_min 1 --random_k_max 5` |
| `adv2_math_no_indicator_pool` | Table 4 no-pool control | Uses the low-level universal audit switch |
| `adv2_code_main` | Main code row | `--retrieval_mode direct --direct_k 3 --batch_audit_metrics --pass_rate 1.0` |

The no-pool control activates a low-level universal audit path used only for
the Table 4 control row.

## Indicator Pool Notes

The math and code indicator-pool JSON files are bundled in this
repository under `test/metrics_pool/`. Precomputed embedding caches can exceed
GitHub's single-file size limit, so they are optional release artifacts. Either
generate them locally or host them outside the repository and pass them to the
launcher with `AGENTDROPOUT_EMBEDDING_CACHE_FILE`.

For example, to generate the mixed code embedding cache:

```bash
python test/metrics_pool/two_pool/embed_metrics-trigger.py \
  --input_file test/metrics_pool/code_mixed/deduplicated_metrics_pool.json \
  --output_cache_file test/metrics_pool/code_mixed/deduplicated_embeddings-trigger.jsonl
```

For the math non-deduplication ablation:

```bash
python test/metrics_pool/two_pool/embed_metrics-trigger.py \
  --input_file test/metrics_pool/two_pool/mixed_metrics_two_pool.json \
  --output_cache_file test/metrics_pool/two_pool/mixed_embeddings_cache_two_pool.jsonl
```

Training-time scripts in `train/` collect raw trajectories for building a pool
and are controlled by environment variables. After collection, run:

```bash
cd train
python Extraction-deduplication-embedding.py
```

The extraction script produces deduplicated indicator records and embedding
caches that can be supplied to the test-time launcher.
