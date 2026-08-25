# GenAI Benchmarking Helm Chart (single node)

Designed as a helm chart to work standalone on a K8s cluster, this helm chart can deploy any [vLLM compatible model](https://docs.vllm.ai/en/latest/models/supported_models.html), or [SGLang compatible model](https://docs.sglang.io/supported_models/text_generation/generative_models.html) and benchmark it on a kubernetes cluster using [Perf Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/perf_analyzer.html).

## Setting the `values.yaml` for the benchmark

The benchmark can be configured through various parameters in the `values.yaml` file. Here's a detailed explanation of each parameter:

### Server Configuration

- `SERVER_TYPE`: Currently "vLLM" or "SGLang"
- `SERVER_IMAGE`: Server image name (e.g., "vllm/vllm-openai:v0.8.5", or "sglang/sglang:latest") in Docker Hub repository
- `SERVER_ARGS`: Any additional arguments to pass to the run command of the container. *Note: For vLLM and SGLang `--model` is already added based on `MODEL_NAME`*
- `SERVER_PORT`: The port the server will listen on (default: "8000")
- `SERVER_HEALTH_CHECK_ENDPOINT`: The endpoint to check if the server is ready (e.g., "/health" for vLLM/SGLang)
- `SERVER_ENDPOINT`: The endpoint for chat completions (e.g., "v1/chat/completions"). This is the endpoint that will be used for the actual inference requests.

### Model Configuration

- `MODEL_NAME`: The full model name as it appears in HF or NGC (e.g., "meta/llama-3.1-8b-instruct")
- `MODEL_NAME_CLEANED`: A cleaned version of the model name without organization prefix (e.g., "llama-3.1-8b-instruct")
- `MODEL_TOKENIZER`: The Hugging Face tokenizer path (e.g., "meta-llama/Llama-3.1-8B-Instruct")
- `HF_TOKEN`: Your Hugging Face API token (required for models that need authentication) *Note: Some models/tokenizers require a `HF_TOKEN` to be set and the related account have approval from the upstream HF registry to pull the related artifacts. For example `meta-llama/Llama-3.1-8B-Instruct`*

### Benchmark Parameters

- `MIN_REQUESTS`: Minimum number of requests to run for each test (e.g., "20")
- `USE_CASES`: Space-separated list of use cases to test in format "name:options"
  - Both LLMs and embedding models can be benchmarked. LLM follow the "chat:input/output" format while embedddings follow "embeddings:batch_size"
  - Example: "chat:128/128 chat:4096/512" or "embeddings:512"
- `CONCURRENCY_RANGE`: Space-separated list of concurrent users to test (e.g., "1 25 50 100")
- `REQUEST_MULTIPLIER`: Number of requests to run per concurrency level (e.g., "5")
- `NUM_GPUS`: Number of GPUs to use for inference (e.g., "1", "2", "4", "8") *Note: ensure the instance type has equal or greater the number of GPUs otherwise the task will fail to deploy.*

### Results Configuration

- `RESULTS_PATH`: Directory where benchmark results will be stored (default: "results")

## Running Locally

Note, that this helm chart has been tested on a K8s cluster with GB200 GPUs. It may also work on other GPU types (supported by vLLM server) but we haven't tested these scenarios.

First create a secret with your NGC API key (required to successfully start server)

```
microk8s kubectl create secret docker-registry <image-pull-secret-name> --docker-server=nvcr.io --docker-username=\$oauthtoken --docker-password=<NGC-API-KEY> 
```

Next deploy the sample helm chart with:

```
microk8s helm install <release-name> llm-benchmark-chart/ --set ngcImagePullSecretName=<image-pull-secret-name>
```

This by default will use the `values.yaml` file inside the chart. If you want to easily switch between multiple values you can add a `-f <override_file_name>.yaml`.

For example, to deploy vLLM helm chart

```
microk8s helm install vllm-benchmark llm-benchmark-chart/ --set ngcImagePullSecretName=<image-pull-secret-name> -f vllm-llm-server-values.yaml
```

or

```
microk8s helm install sg-benchmark llm-benchmark-chart/ --set ngcImagePullSecretName=<image-pull-secret-name> -f sg-llm-server-values.yaml
```

## Running in NVCF

### Upload chart

First create Helm Chart in the NGC Web UI

```
helm package llm-benchmark-chart
ngc registry chart push <org-id>/llm-benchmark-chart:0.1.1
```

See here for more info: https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#managing-helm-charts-using-helm-cli

### Run task

```shell
ngc cf task create --name benchmark --helm-chart <org-name>/llm-benchmark-chart:<version> --result-handling-strategy NONE   --configuration '{
    "RESULTS_PATH": "",
    "HF_TOKEN": "<token>",
    "NUM_GPUS": <number-of-gpus>,
  }' \
--gpu-specification <GPU>:<InstanceName>:<ClusterName>
```

Example:

```shell
ngc cf task create --name benchmark --helm-chart qdrlnbkss8u1/llm-benchmark-chart:0.1.1 --result-handling-strategy NONE    --configuration '{
    "RESULTS_PATH": "",
    "HF_TOKEN": "<token>",
    "NUM_GPUS": 1,
  }' \
--gpu-specification H100:GPU.H100_1x
```
