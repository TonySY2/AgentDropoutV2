#!/bin/bash


reasoning_api_list=(
    ####
)



embedding_api_list=(
    ####
)

node_num=${#reasoning_api_list[@]}


output_dir=results-math/debug
tmp_dir=${output_dir}/tmp_math
log_dir=${output_dir}/logs
result_dir=${output_dir}/results

mkdir -p ${output_dir}
mkdir -p ${tmp_dir}
mkdir -p ${log_dir}
mkdir -p ${result_dir}


python -u get_tasks.py \
    --in_file project_datasets/MATH/processed_math/train.jsonl \
    --out_dir ${result_dir} \
    --node_num ${node_num} \
    --tmp_dir ${tmp_dir}


for ((i=0; i<$node_num; i++)); do
    LOG_FILE="${log_dir}/node_${i}_full.log"

    SCRIPT_NAME="experiments/math/run_math.py" 
    
    nohup python -u ${SCRIPT_NAME} \
        --in_file ${tmp_dir}/part_${i}.jsonl \
        --out_file ${result_dir}/part_${i}.json \
        --log_file ${LOG_FILE} \
        --selector_url "###" \
        --selector_model "gpt-4.1-mini" \
        --selector_key "####" \
        --reasoning_url ${reasoning_api_list[$i]} \
        --reasoning_model "####" \
        --reasoning_key "####" \
        --supervisor_url "###" \
        --supervisor_model "gpt-4o" \
        --supervisor_api_key "####" \
        --embedding_url ${embedding_api_list[$i]} \
        --embedding_model "####" \
        --embedding_key "####" \
        --max_turns 7 \
        --limit 2000 \
        > ${log_dir}/node_${i}.log 2>&1 & disown


    PID=$!
    
    echo "  - ✅ Success! Worker for node ${i} started."
    echo "  - Process ID (PID): ${PID}"
    echo "  - Stdout Log: ${log_dir}/node_${i}.log"
    echo "  - Detailed Log: ${LOG_FILE}"
    echo "----------------------------------------"
done

echo "[+] All workers have been launched."
echo "👉 Monitor logs: tail -f ${log_dir}/node_0_full.log"