#!/usr/bin/bash

# LL3.1 70B TP8 FP8
size=70
tp_size=8

size=8
tp_size=1

model_cfg=/code/tensorrt_llm/run_readme/trtllm_modelconfig/llama-3.1-${size}b/config_tp${tp_size}_fp8.json
engine_dir=trtllm_engine_trtllm_llama3.1_${size}b_gpus${tp_size}_${tp_size}_fp8
if [[ $size = "8" ]]; then
    tokenizer=meta-llama/Meta-Llama-3.1-8B-Instruct
elif [[ $size == "70" ]]; then
    tokenizer=meta-llama/Meta-Llama-3.1-70B-Instruct
elif [[ $size == "405" ]]; then
    tokenizer=meta-llama/Meta-Llama-3.1-405B-Instruct
fi

if [ $tp_size -gt 1 ]; then
    MGPU="mpirun -n $tp_size --allow-run-as-root --oversubscribe " 
fi

echo "tokenizer: $tokenizer"

## engine build
#
#trtllm-build --model_config $model_cfg \
#--output_dir $engine_dir \
#--workers $tp_size \
#--max_num_tokens 4096 \
#--max_input_len 64000 \
#--max_seq_len 65000 \
#--use_paged_context_fmha enable 

# dataset preparation
dataset=trtllm_dataset
mkdir -p $dataset

req="1 2 4 8 16 32 64 128 256"
isl="128 2048"
osl="1 128 2048"

#for o in $osl; do
#    for i in $isl; do
#        for r in $req; do
#          python /app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py \
#          --output ${dataset}/Meta-Llama-3.1-${size}B-Instruct_tokens-fixed-lengths_${r}_${i}_${o}.json \
#          --tokenizer $tokenizer \
#           token-norm-dist \
#           --num-requests $r \
#           --input-mean $i --input-stdev 0 \
#           --output-mean $o --output-stdev 0
#           done
#        done
#done

# latency measurement by gptManagerBenchmark 

for o in $osl; do
    for i in $isl; do
        for r in $req; do
  	    echo "r: $r, i: $i, o: $o"
            $MGPU /app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark \
            --engine_dir $engine_dir \
            --request_rate -1 \
            --static_emulated_batch_size $r \
            --static_emulated_timeout 100 \
            --dataset ${dataset}/Meta-Llama-3.1-${size}B-Instruct_tokens-fixed-lengths_${r}_${i}_${o}.json 
           done
        done
done
