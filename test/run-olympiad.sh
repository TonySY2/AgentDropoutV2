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

#two-pool(deduped)
METRIC_POOL_FILE="metrics_pool/two_pool/deduped-mixed_metrics_two_pool.json"
EMBEDDING_CACHE_FILE="metrics_pool/two_pool/deduped-mixed_two_pool-trigger.jsonl"


INPUT_DATASET="project_datasets/olympiad/test.jsonl" 

# ===========================================

node_num=${#reasoning_api_list[@]}


output_dir=results-olympiad/debug-1
tmp_dir=${output_dir}/tmp_data
log_dir=${output_dir}/logs
result_dir=${output_dir}/results

mkdir -p ${output_dir}
mkdir -p ${tmp_dir}
mkdir -p ${log_dir}
mkdir -p ${result_dir}

echo ">>> 🚀 Starting OlympiadBench Run..."


python -u get_tasks.py \
    --in_file ${INPUT_DATASET} \
    --out_dir ${result_dir} \
    --node_num ${node_num} \
    --tmp_dir ${tmp_dir}


for ((i=0; i<$node_num; i++)); do
    

    SCRIPT_PATH="experiments/olympiad/run_olympiad.py"
    

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
        --metric_pool_file "${METRIC_POOL_FILE}" \
        --embedding_cache_file "${EMBEDDING_CACHE_FILE}" \
        --max_turns 7 \
        --pass_rate 1.0 \
        --direct_k 5 \
        > ${log_dir}/node_${i}.log 2>&1 & 
        #--disable_log_buffer \
        # force_direct_search
        # retrieve_p
        # select_q
        # direct_k
        #--baseline_only \
        #--use_simple_audit \
        #        --use_simple_audit \


    PID=$!
    
    echo "  - ✅ Worker ${i} started. PID: ${PID}"
    echo "    Log: ${log_dir}/node_${i}.log"
done

