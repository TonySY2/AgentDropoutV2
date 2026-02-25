# AgentDropoutV2

This repository anonymously releases the codes and data for the paper:  
**AgentDropoutV2: Optimizing Information Flow in Multi-Agent Systems via Test-Time Rectify-or-Reject Pruning**

## **🔗 Quick Links**

- **[About AgentDropoutV2](#about)**
- **[File Structure](#structure)**
- **[Requirements](#requirements)**
- **[Quick Start](#start)**

## **🛡️ About AgentDropoutV2**<a name="about"></a>

**AgentDropoutV2** is a test-time framework designed to dynamically optimize information flow in Multi-Agent Systems (MAS) without expensive retraining. 

It acts as an active firewall during MAS execution:
1.  **Intercept**: It intercepts agent outputs before they are broadcast.
2.  **Rectify**: A retrieval-augmented rectifier scrutinizes the output using a **Failure-Driven Indicator Pool** (constructed from historical error patterns). It provides targeted feedback for iterative self-correction.
3.  **Reject**: If the output remains flawed after maximum retries, it is pruned to prevent error propagation.
4.  **Fallback**: A safeguard mechanism preserves structural integrity if too many agents are pruned.


<p align="center">
  <img src="image/readme/main-picture.png" alt="Main Picture">
</p>
<p align="center"><strong>The Framework of AgentDropout</strong></p>

## **📜 File Structure**<a name="structure"></a>

The repository is organized into two main components: `train` (for offline indicator pool construction) and `test` (for online inference).



## **🛠️ Requirements**<a name="requirements"></a>

This project can be reproduced with a single Python environment:

```bash
conda create -n myenv python=3.10.18
conda activate myenv
pip install -r requirements.txt
```


## **🚀 Quick Start**<a name="start"></a>

### **1. Indicator Pool Construction (Optional)**
*If you want to build your own indicator pool from scratch (The `train` folder).*

1.  **Collect Failure Trajectories**: Run the baseline MAS on training sets (e.g., MATH train set) to collect execution logs.
    ```bash
    cd train
    bash run-math-train.sh
    ```
2.  **Mine & Deduplicate Indicators**: Use the extraction script to distill error patterns from logs and remove redundancy.
    ```bash
    python Extraction-deduplication-embedding.py 
    ```

### **2. Test-Time Inference**
*Run the proposed AgentDropoutV2 on benchmarks (The `test` folder).*

> **Note**: You need to deploy the reasoning model (e.g., Qwen3-8B) and embedding model (e.g., Qwen3-Embedding) using vLLM servers before running the scripts.

⚠️ **IMPORTANT**: Due to file size limits, the pre-computed embedding cache file (`metrics_embeddings_trigger.jsonl`) is NOT included in this repository. 
**You must generate it locally before running any inference.**

1.  **Generate Embeddings**:
    Run the embedding script to encode the indicator pool (`deduped-mixed_metrics_two_pool.json`) into vectors.
    ```bash
    cd test/metrics_pool/two_pool
    python embed_metrics-trigger.py
    ```
    This will generate the `deduped-mixed_two_pool-trigger.jsonl` file required for retrieval.

2.  **Run Inference**:
    Once the embedding file is ready, you can launch the experiments.

    **To run evaluation on MATH-500:**
    ```bash
    cd ../..  # Back to ./test directory
    bash run-math500.sh
    ```

    **To run evaluation on AQuA:**
    ```bash
    bash run-aqua.sh
    ```

**Configuration Arguments**:
Most shell scripts accept the following key arguments:

*   **Model Configuration**
    *   `--reasoning_model`: The model name used by participant agents (e.g., Qwen3-8B).
    *   `--supervisor_model`: The model name used by the Supervisor/Rectifier (e.g., GPT-4o or Qwen3-8B).
    *   `--embedding_model`: The model name used for embedding retrieval.

*   **Pool & Cache**
    *   `--metric_pool_file`: Path to the JSON file containing the adversarial indicator pool.
    *   `--embedding_cache_file`: Path to the pre-computed embedding cache (`.jsonl`) for retrieval.

*   **Workflow Control**
    *   `--max_turns`: Maximum turns for the chat group. *Note: The actual maximum number of agent speaking rounds is `max_turns - 1`.*
    *   `--retries_times`: Maximum number of self-correction retries allowed for an agent if errors are detected.

*   **Ablation & Modes**
    *   `--baseline_only`: If set, runs the standard MAS baseline without the Rectify-or-Reject mechanism.
    *   `--use_simple_audit`: If set, uses a fixed, static metric for auditing instead of retrieving from the pool.
    *   `--direct_k`: The number of indicators to retrieve based on semantic similarity (default: 5).
    *   `--random_k`: If set to $x > 0$, retrieves $x$ random indicators instead of semantic matching (used for ablation studies).
