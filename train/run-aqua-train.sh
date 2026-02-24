#!/bin/bash


reasoning_api_list=(
    ####
)

embedding_api_list=(
    ####
)

INPUT_DATASET="project_datasets/aqua/train.jsonl"

# ===========================================

node_num=${#reasoning_api_list[@]}


output_dir=results-aqua/debug
tmp_dir=${output_dir}/tmp_aqua
log_dir=${output_dir}/logs
result_dir=${output_dir}/results

mkdir -p ${output_dir}
mkdir -p ${tmp_dir}
mkdir -p ${log_dir}
mkdir -p ${result_dir}


python -u get_tasks.py \
    --in_file ${INPUT_DATASET} \
    --out_dir ${result_dir} \
    --node_num ${node_num} \
    --tmp_dir ${tmp_dir}


for ((i=0; i<$node_num; i++)); do
    
 
    DETAILED_LOG="${log_dir}/node_${i}_full.log"
    
    nohup python -u experiments/aqua/run_aqua.py \
        --in_file ${tmp_dir}/part_${i}.jsonl \
        --out_file ${result_dir}/part_${i}.json \
        --log_file ${DETAILED_LOG} \
        --reasoning_url ${reasoning_api_list[$i]} \
        --reasoning_model "####" \
        --embedding_url ${embedding_api_list[$i]} \
        --embedding_model "####" \
        --max_turns 7 \
        --num_samples 2000 \
        > ${log_dir}/node_${i}.log 2>&1 & disown

 
    PID=$!
    
    echo "  - ✅ Success! Worker for node ${i} started in background."
    echo "  - Process ID (PID): ${PID}"
    echo "  - Stdout Log: ${log_dir}/node_${i}.log"
    echo "  - Detailed Log: ${DETAILED_LOG}"
    echo "----------------------------------------"
  
done

echo "[+] All workers have been launched."
echo "👉 Monitor logs: tail -f ${log_dir}/node_0_full.log"