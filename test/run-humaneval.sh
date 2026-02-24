#!/bin/bash


reasoning_api_list=(
    ####
)


supervisor_api_list=(
    ####
)


embedding_api_list=(
    ####
)






INPUT_DATASET="project_datasets/humaneval/humaneval-py_id.jsonl"

# ===========================================

node_num=${#reasoning_api_list[@]}


output_dir=results-humaneval/debug-1
tmp_dir=${output_dir}/tmp_humaneval
log_dir=${output_dir}/logs
result_dir=${output_dir}/results

mkdir -p ${output_dir}
mkdir -p ${tmp_dir}
mkdir -p ${log_dir}
mkdir -p ${result_dir}

echo "Starting HumanEval job with ${node_num} nodes..."
echo "Metrics File: ${METRIC_POOL_FILE}"
echo "Input Dataset: ${INPUT_DATASET}"


python -u get_tasks.py \
    --in_file ${INPUT_DATASET} \
    --out_dir ${result_dir} \
    --node_num ${node_num} \
    --tmp_dir ${tmp_dir}


for ((i=0; i<$node_num; i++)); do
    
 
    SCRIPT_PATH="experiments/humaneval/run_humaneval.py"
    
    nohup python -u ${SCRIPT_PATH} \
        --in_file ${tmp_dir}/part_${i}.jsonl \
        --out_file ${result_dir}/part_${i}.json \
        --log_file ${log_dir}/node_${i}_detailed.log \
        --reasoning_url ${reasoning_api_list[$i]} \
        --reasoning_model "####" \
        --supervisor_url ${supervisor_api_list[$i]} \
        --supervisor_model "####" \
        --embedding_url ${embedding_api_list[$i]} \
        --embedding_model "####" \
        --metric_pool_file "####" \
        --embedding_cache_file "####" \
        --max_turns 7 \
        --pass_rate 1.0 \
        --retries_times 3 \
        --use_simple_audit \
        > ${log_dir}/node_${i}.log 2>&1 & disown


    PID=$!
    
    echo "  - ✅ Success! Worker for node ${i} started in background."
    echo "  - Process ID (PID): ${PID}"
    echo "  - Stdout Log: ${log_dir}/node_${i}.log"
    echo "  - Detailed Log: ${log_dir}/node_${i}_detailed.log"
    echo "----------------------------------------"
done

echo "[+] All workers have been launched."
echo "👉 Monitor logs: tail -f ${log_dir}/node_0_detailed.log"