#!/usr/bin/bash

# LL3.1 70B TP8 FP8

model_cfg=/code/tensorrt_llm/run_readme/trtllm_modelconfig/llama-3.1-70b/config_tp8_fp8.json
engine_dir=trtllm_engine_trtllm_llama3.1_70b_gpus8_8_fp8
tokenizer=meta-llama/Meta-Llama-3.1-70B-Instruct
tp_size=8

# engine build

trtllm-build --model_config $model_cfg \
--output_dir $engine_dir \
--workers $tp_size \
--max_num_tokens 4096 \
--max_input_len 64000 \
--max_seq_len 65000 \
--use_paged_context_fmha enable 

# dataset preparation

req="1 2 4 8 16 32 64 128 256"
isl="128 2048"
osl="1 128 2048"

for o in $osl; do
    for i in $isl; do
        for r in $req; do
          python /app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py \
          --output Meta-Llama-3.1-70B-Instruct_tokens-fixed-lengths_${r}_${i}_${o}.json \
          --tokenizer $tokenizer \
           token-norm-dist \
           --num-requests $r \
           --input-mean $i --input-stdev 0 \
           --output-mean $o --output-stdev 0
           done
        done
done

# latency measurement by gptManagerBenchmark 

for o in $osl; do
    for i in $isl; do
	for r in $req; do
  	    echo "r: $r, i: $i, o: $o"
	    mpirun -n 8 --allow-run-as-root --oversubscribe /app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark \
            --engine_dir $engine_dir \
            --request_rate -1 \
            --static_emulated_batch_size $r \
            --static_emulated_timeout 100 \
            --dataset Meta-Llama-3.1-70B-Instruct_tokens-fixed-lengths_${r}_${i}_${o}.json 
           done
        done
done
