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


## **📜 File Structure**<a name="structure"></a>

The repository is organized into two main components: `train` (for offline indicator pool construction) and `test` (for online inference).

| Directory | Contents |
| :--- | :--- |
| `./train` | **Phase 1: Pool Construction**. Codes for sampling failure trajectories and mining adversarial indicators. |
| &nbsp;&nbsp;`./train/experiments` | Scripts to run MAS and collect raw failure logs. |
| &nbsp;&nbsp;`./train/Extraction...py` | The core script for Indicator Extraction, Deduplication, and Embedding. |
| `./test` | **Phase 2: Inference**. Codes for the test-time Rectify-or-Reject mechanism. |
| &nbsp;&nbsp;`./test/AgentDropout` | Core framework logic (Supervisor, MathSolver, Grader). |
| &nbsp;&nbsp;`./test/metrics_pool` | The pre-built **Adversarial Indicator Pool** and embedding scripts. |
| &nbsp;&nbsp;`./test/experiments` | Runners for specific datasets (MATH, AQuA, GSM8K, etc.). |
| &nbsp;&nbsp;`./test/run-*.sh` | Shell scripts to launch experiments easily. |
| `./test/project_datasets` | Evaluation datasets (MATH-500, AQuA, AIME, etc.). |



## **🛠️ System Architecture & Requirements**

The framework adopts a decoupled **Client-Server architecture** to ensure modularity and inference efficiency. To replicate our experiments, we recommend setting up two separate environments.

### **1. Inference Server Environment**
This environment hosts the LLM service. We utilized **NVIDIA A800 GPUs** for deployment.
*   **Core Engine:** `vLLM == 0.14.1`.
*   **Base Framework:** `PyTorch == 2.9.1`.

To set up the server environment, please use:
```bash
conda create -n vllm_env python=3.10
conda activate vllm_env
pip install -r requirements_vllm.txt
```

### **2. Agent Client Environment**
This environment handles the core multi-agent logic, evaluation pipelines, and retrieval-augmented generation (RAG).
*   **Language:** `Python == 3.10`
*   **Key Dependencies:**
    *   `openai == 2.15.0` (For API communication)
    *   `sentence-transformers == 5.2.2` (For retrieval augmentation)
    *   `json_repair == 0.55.1` (For robust JSON parsing)
    *   `numpy`, `tqdm`, `pyautogen` (As specified in dependencies)

To set up the client/test environment, please use:
```bash
conda create -n agent_env python=3.10
conda activate agent_env
pip install -r requirements_test.txt
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