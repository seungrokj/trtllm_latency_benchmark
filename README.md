# trtllm_latency_benchmark

## This env is based on tensorrt-llm:0.12.0.dev2024073000 and clone the specific branch from the git repo. 


```sh
apt-get update && apt-get -y install git git-lfs
git lfs install

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout a681853d3803ee5893307e812530b5e7004bb6e1
git submodule update --init --recursive
git lfs pull
```

## Build & install the trtllm by this instruction.  

https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html

## Run benchmarks

Same for LL3.1 8B/70B/405B

```sh
# LL3.1 70B TP8 FP8
./trtllm_benchmark_latency.sh
```

## Analyze the latency

This is the latency of the specific batch (request) / input / output case.
*[BENCHMARK] total_latency(ms) 2239.22*

```sh
r: 1, i: 128, o: 128
[BENCHMARK] num_samples 1
[BENCHMARK] num_error_samples 0

[BENCHMARK] num_samples 1
[BENCHMARK] total_latency(ms) 2239.22
[BENCHMARK] seq_throughput(seq/sec) 0.45
[BENCHMARK] token_throughput(token/sec) 57.16

[BENCHMARK] avg_sequence_latency(ms) 2239.20
[BENCHMARK] max_sequence_latency(ms) 2239.20
[BENCHMARK] min_sequence_latency(ms) 2239.20
[BENCHMARK] p99_sequence_latency(ms) 2239.20
[BENCHMARK] p90_sequence_latency(ms) 2239.20
[BENCHMARK] p50_sequence_latency(ms) 2239.20
```
