# Dynamo + vLLM aggregated KVBM

Operator-independent proof-of-concept stack matching Dynamo's `agg_kvbm.sh` process shape: a Dynamo
frontend and one aggregated `dynamo.vllm` worker share a Pod, with the vLLM `DynamoConnector` enabling
KVBM. A pinned etcd sidecar provides KVBM registration/discovery; no Dynamo or Grove operator is needed.

The stack is intentionally small. It is suitable for validating KVBM wiring and collecting an initial
`llm-perf` sweep, not as a production HA topology. Recipe-specific cache sizing and vLLM flags live in
the cell's `recipe.yaml`.
