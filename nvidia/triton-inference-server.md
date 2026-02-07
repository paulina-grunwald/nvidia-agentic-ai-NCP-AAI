# Triton Inference Server
- Triton is NVIDIA's **open-source inference serving software** that standardizes AI model deployment and execution. It's part of the NVIDIA AI platform and enables running inference on trained models from virtually any framework on GPUs, CPUs, or other processors.
- NVIDIA's **production inference serving platform** for deploying AI models at scale. Think of it as the ***"traffic controller"*** between your application and your models.
- What Triton Does:
    - Serves trained ML/DL models for inference at scale
    - Supports multiple frameworks simultaneously
    - Runs on GPUs, CPUs, and other accelerators
    - Provides enterprise-grade features for production deployments
- Triton has **multi-Framework Support.** Triton can deploy models from multiple frameworks including:
    - TensorRT
    - PyTorch
    - ONNX
    - OpenVINO
    - Python
    - RAPIDS FIL (for XGBoost, LightGBM, Scikit-Learn)
- **LLM Support -** Triton provides low-latency, high-throughput inference for large language models through **TensorRT-LLM** integration—this is especially relevant for agentic AI workloads.

---

# TensorRT-LLM Integration (Critical for Exam)

## What is TensorRT-LLM?

**TensorRT-LLM** is NVIDIA's library specifically designed to **optimize and accelerate Large Language Model inference** on NVIDIA GPUs. It's not a general inference optimizer — it's purpose-built for transformer-based LLMs.

**Key point:** TensorRT-LLM optimizes the MODEL. Triton SERVES the optimized model. They work together.

```
┌─────────────────────────────────────────────────────────────┐
│                      Your LLM                                │
│              (Llama, GPT, Mistral, etc.)                    │
└─────────────────────────┬───────────────────────────────────┘
                          │ Optimize with
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    TensorRT-LLM                              │
│         (Compiles model with LLM-specific optimizations)    │
└─────────────────────────┬───────────────────────────────────┘
                          │ Deploy to
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                Triton Inference Server                       │
│            (Serves optimized model at scale)                │
└─────────────────────────────────────────────────────────────┘
```

---

## Key TensorRT-LLM Optimizations

| Optimization | What It Does | Why It Matters |
| --- | --- | --- |
| **In-Flight Batching** | Batches requests at token level, not sequence level | Higher throughput for variable-length requests |
| **Paged Attention** | Manages KV cache memory like OS virtual memory | Handles longer contexts, more concurrent users |
| **KV Cache Optimization** | Efficiently stores key/value tensors across tokens | Reduces memory, speeds up generation |
| **Tensor Parallelism** | Splits model across multiple GPUs | Run larger models than fit on one GPU |
| **Quantization** | FP16, INT8, FP8 precision | Faster inference, lower memory |
| **Custom CUDA Kernels** | Hand-optimized GPU code for attention, etc. | Maximum GPU efficiency |

---

## In-Flight Batching (Exam Favorite)

Traditional batching waits for ALL sequences in a batch to complete. In-flight batching is smarter:

**Traditional Batching:**

```
Request A (10 tokens)  ████████████████████
Request B (50 tokens)  ████████████████████████████████████████████████████████████████
Request C (5 tokens)   ████████████████████
                       ↑ All must wait for longest (B) to finish
```

**In-Flight Batching:**

```
Request A (10 tokens)  ██████████ → Done! Slot freed
Request D joins →                 ████████████████ → Done!
Request B (50 tokens)  ████████████████████████████████████████████████████████████████
Request C (5 tokens)   █████ → Done! Slot freed
Request E joins →            ████████████████████████
```

**Result:** Shorter requests complete faster, new requests can join mid-batch.

---

## Paged Attention

Inspired by OS virtual memory — KV cache is stored in non-contiguous "pages" instead of one big block.

**Benefits:**

- No memory fragmentation
- Support longer sequences without OOM
- More concurrent requests in same memory
- Dynamic memory allocation as sequences grow

**Exam tip:** When asked about "memory efficiency for LLMs" or "handling long contexts," paged attention is often the answer.

---

## Triton + TensorRT-LLM Architecture

Triton uses a special **TensorRT-LLM backend** to serve optimized LLMs:

```
model_repository/
├── tensorrt_llm_model/
│   ├── config.pbtxt
│   └── 1/
│       └── (TensorRT-LLM engine files)
├── preprocessing/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py          # Tokenization
└── postprocessing/
    ├── config.pbtxt
    └── 1/
        └── model.py          # Detokenization
```

**Typical ensemble pipeline:**

```
Text Input → Preprocessing (tokenize) → TensorRT-LLM Engine → Postprocessing (detokenize) → Text Output
```

---

## Key Configuration Parameters

```protobuf
# config.pbtxt for TensorRT-LLM model
name: "tensorrt_llm"
backend: "tensorrtllm"
max_batch_size: 64

parameters: {
  key: "max_tokens_in_paged_kv_cache"
  value: { string_value: "8192" }
}
parameters: {
  key: "batch_scheduler_policy"
  value: { string_value: "guaranteed_no_evict" }
}
parameters: {
  key: "kv_cache_free_gpu_mem_fraction"
  value: { string_value: "0.9" }
}
```

| Parameter | Purpose |
| --- | --- |
| `max_tokens_in_paged_kv_cache` | Total KV cache memory (in tokens) |
| `batch_scheduler_policy` | How to handle memory pressure |
| `kv_cache_free_gpu_mem_fraction` | % of free GPU memory for KV cache |

---

## Why This Matters for Agentic AI

Agentic AI systems make **many LLM calls** with varying lengths:

| Agent Action | Request Pattern |
| --- | --- |
| Tool selection | Short prompt, short response |
| Reasoning | Medium prompt, variable response |
| Code generation | Long context, long response |
| Summarization | Long context, short response |

**TensorRT-LLM + Triton handles this efficiently:**

- In-flight batching handles variable lengths
- Paged attention manages memory for long contexts
- Triton's scheduling manages concurrent agent requests

---

## Exam Question Patterns

**Q: How do you optimize LLM inference for production?**

A: Use TensorRT-LLM to compile the model with optimizations (quantization, KV cache), then deploy on Triton for serving at scale.

**Q: What enables efficient batching for requests with different sequence lengths?**

A: In-flight batching (also called continuous batching) — allows new requests to join and completed requests to leave mid-batch.

**Q: How does TensorRT-LLM handle memory for long sequences?**

A: Paged attention — manages KV cache in pages like virtual memory, preventing fragmentation.

**Q: What's the relationship between TensorRT-LLM and Triton?**

A: TensorRT-LLM optimizes the LLM model. Triton serves the optimized model with features like dynamic batching, scaling, and API endpoints.

**Q: An agentic AI system needs to make many concurrent LLM calls with varying lengths. What's the best deployment?**

A: TensorRT-LLM with in-flight batching, deployed on Triton with appropriate instance groups.

---

## TensorRT-LLM vs TensorRT (Don't Confuse!)

| Feature | TensorRT | TensorRT-LLM |
| --- | --- | --- |
| Purpose | General model optimization | LLM-specific optimization |
| Model types | CNNs, general DNNs | Transformers, LLMs |
| Special features | Layer fusion, precision | In-flight batching, paged attention, KV cache |
| Output | `.plan` engine file | TensorRT-LLM engine |

---

- **Model Ensembling**
    - **Ensembling** means chaining multiple models together into a **pipeline** where the output of one model flows into the input of the next — all handled server-side in a single request
    - Allows executing AI workloads with multiple models, pipelines, and pre/post-processing steps. Different parts of an ensemble can run on CPU or GPU.

        ![image.png](attachment:a9d6bbe3-c9bd-4a29-96aa-e69d4c9774f7:image.png)

        **Benefits:**

        - Reduced network latency (one round trip instead of many)
        - Data stays on GPU between steps
        - Simplified client code
        - Better resource utilization
    - Common Exam Question Patterns
        - **Q: How do you reduce latency when you need preprocessing before inference in Triton?**
        A: Use model ensembling to chain preprocessing and inference into a single request, avoiding multiple network round trips.
        - **Q: What platform type do you specify for an ensemble model?**
        A: `platform: "ensemble"`
        - **Q: When would you use BLS instead of ensemble?**
        A: When you need conditional logic, dynamic routing, or complex control flow that can't be expressed as a fixed DAG.
- **Dynamic batching**

    <aside>
    🔥

    When you see questions about **Triton Inference Server** + **performance optimization** + **real-time**, dynamic batching is almost

    </aside>

    - Dynamic batching is Triton's ability to **automatically combine multiple inference requests into a single batch** on-the-fly, even when those requests arrive at different times.

    ![image.png](attachment:a0ea7c29-63a9-4235-a705-67e80ad295d0:image.png)

    - Triton automatically groups incoming requests into batches on-the-fly, maximizing GPU utilization even with variable request timing
    - Triton uses two key parameters to decide when to form a batch:


        | Parameter | What It Does |
        | --- | --- |
        | **max_batch_size** | Maximum requests to combine in one batch |
        | **max_queue_delay_microseconds** | How long to wait for more requests before executing |
    - **The tradeoff:**
        - Wait longer → bigger batches → better throughput
        - Wait shorter → smaller batches → lower latency
    - **Best for:**
        - Variable/unpredictable request patterns
        - Real-time applications
        - Multi-tenant serving (many users)
    - **When to Use What**


        | Scenario | Recommendation |
        | --- | --- |
        | Real-time chatbot | Dynamic batching with low queue delay (~100μs) |
        | Batch processing overnight | Static batching or large max_queue_delay |
        | Variable traffic, latency matters | Dynamic batching with tuned preferred_batch_size |
        | Single request at a time | Dynamic batching still helps (no downside) |
    - In agentic AI systems, you often need to orchestrate multiple model calls with low latency. Triton's dynamic batching, concurrent execution, and ensemble capabilities make it ideal for serving the inference backbone of AI agents.
    - Common Exam Question Patterns
        - **Q: How do you optimize Triton for real-time inference?**
        A: Enable dynamic batching with appropriate `max_queue_delay_microseconds` to balance latency and throughput.
        - **Q: What's the tradeoff in dynamic batching?**
        A: Higher `max_queue_delay` = better throughput but higher latency. Lower delay = faster response but potentially smaller batches.
        - **Q: Why use `preferred_batch_size`?**
        A: Helps Triton pre-allocate memory efficiently and form optimal batch sizes.
- **High Level Architecture**

    ![image.png](attachment:98aa0af8-5e95-426b-8f2d-a9d313dbfa9e:image.png)

    ```jsx
    ┌─────────────────────────────────────────────────────────────┐
    │                    Client Applications                       │
    │              (HTTP/REST, gRPC, C API)                        │
    └─────────────────────────┬───────────────────────────────────┘
                              │
    ┌─────────────────────────▼───────────────────────────────────┐
    │                  TRITON INFERENCE SERVER                     │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │              Request Handler                         │    │
    │  │         (HTTP, gRPC, C API endpoints)               │    │
    │  └─────────────────────────┬───────────────────────────┘    │
    │                            │                                 │
    │  ┌─────────────────────────▼───────────────────────────┐    │
    │  │              Per-Model Scheduler                     │    │
    │  │    (Default, Dynamic Batcher, Sequence Batcher)     │    │
    │  └─────────────────────────┬───────────────────────────┘    │
    │                            │                                 │
    │  ┌─────────────────────────▼───────────────────────────┐    │
    │  │                   Backends                           │    │
    │  │  (TensorRT, PyTorch, ONNX, Python, TensorFlow...)   │    │
    │  └─────────────────────────────────────────────────────┘    │
    │                                                              │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │              Model Repository                        │    │
    │  │         (Local filesystem, S3, GCS, Azure)          │    │
    │  └─────────────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────┘
    ```

    | Component | Description |
    | --- | --- |
    | **Model Repository** | Central storage for all models and their configurations |
    | **Request Handler** | Accepts requests via HTTP/REST (port 8000), gRPC (port 8001), or C API |
    | **Scheduler** | Manages request queuing and batching per model |
    | **Backend** | Framework-specific inference engine |
    | **Instance Groups** | Multiple parallel instances of the same model |

    ### Request Flow

    1. Client sends inference request (HTTP/gRPC/C API)
    2. Request routed to appropriate model's scheduler
    3. Scheduler batches requests (if enabled)
    4. Batched requests sent to backend
    5. Backend executes inference
    6. Results returned to client
- **Key Configuration Options**


    | Field | Purpose | Exam Tip |
    | --- | --- | --- |
    | `max_batch_size` | Maximum batch size supported | If > 1 and no scheduler specified, dynamic batching auto-enables |
    | `instance_group` | Number and placement of model instances | Enables parallel execution |
    | `dynamic_batching` | Configure request batching | Critical for throughput optimization |
    | `sequence_batching` | For stateful models | Required when model maintains **state** |
    | `ensemble_scheduling` | For model pipelines | Connects multiple models together |
- **Production deployment** with:
    - Multi-model serving (host multiple models simultaneously) on a single instance
    - Request routing and load balancing
    - Model version management
    - Dynamic batching for throughput optimization
    - has **built-in load balancing** for multi-model serving
- **Batching & Scheduling**

    <aside>
    🔥

    Batching is **critical for GPU throughput optimization**—a key exam topic.

    </aside>

    <aside>
    🔥

    **Exam Tip:** Increasing instance count allows more parallel inference but uses more GPU memory.

    </aside>

    ### Types of Batchers:

    - ***Default Scheduler (No Batching)***
        - Processes requests one at a time
        - Use when: model doesn't support batching (`max_batch_size: 0`)
    - ***Dynamic Batcher***
        - Combines multiple requests into batches dynamically
        - **For stateless models** (no state between requests)
        - Examples: image classification, object detection, embeddings
    - ***Sequence Batcher***
        - **For stateful models** (maintain state across requests)
        - Routes all requests in a sequence to the same model instance
        - Examples: RNNs, conversational models, streaming audio
    - **Ragged Batching**
        - For inputs with variable-length dimensions
        - Combines requests with different sequence lengths
- **Capabilities**
    - Model Version Management
        - **Zero-downtime updates**: Deploy new model versions without service interruption
        - **A/B testing**: Route traffic between versions (e.g., 90% v1, 10% v2)
        - **Canary deployments**: Gradual rollout of new models
        - **Automatic rollback**: If new version fails, instantly revert

            ```
            model_repository/
            ├── llm_v1/
            │   └── 1/  # Version 1
            └── llm_v2/
                └── 1/  # Version 2 (new)

            # Triton serves both, routes based on policy
            ```

    - Concurrency & Scalability
        - **Dynamic batching**: Groups multiple requests for GPU efficiency
        - **Multiple model instances**: Scale across GPU cluster
        - **Load balancing**: Distributes requests across instances
        - **Concurrent model execution**: Serve different models simultaneously

        **Exam scenario:** "Thousands of concurrent requests" → Triton is the answer

    - Performance Optimization
        - **Request scheduling**: Priority queuing for latency-sensitive requests
        - **GPU memory management**: Efficient allocation across models
        - **Backend support**: *TensorRT*, PyTorch, ONNX, TensorRT-LLM
        - **Protocol support**: HTTP/REST, gRPC for low latency
    - Integration with NVIDIA Stack

        **Typical production architecture:**

        ```
        Agent Logic Container
            ↓
        Triton Inference Server (frontend)
            ↓
        NIMs (NVIDIA Inference Microservices)
            ↓
        GPU Cluster
        ```

- **When to Use Triton (Exam Patterns)**

    ✅ **Use Triton when you need:**

    - Production deployment with high concurrency
    - Model version management / seamless rollouts
    - Multi-model serving (different models for different agents)
    - Scalability across GPU cluster
    - Enterprise-grade inference serving

    ❌ **Don't confuse with:**

    - **TensorRT-LLM**: Optimizes single model (speed/memory) → feeds into Triton
    - **NeMo**: Training/fine-tuning framework → output gets deployed on Triton
    - **NIMs**: Pre-optimized models → served via Triton

    ## Triton vs Other Options

    | Requirement | Solution |
    | --- | --- |
    | Model is too slow | TensorRT-LLM optimization |
    | Model lacks accuracy | NeMo fine-tuning |
    | Need production deployment | Triton Inference Server |
    | Need high concurrency | Triton + GPU cluster |
    | Seamless model updates | Triton version management |

    ❌ Don't say "use Triton to improve model accuracy" → That's NeMo fine-tuning

    ❌ Don't say "use Triton to speed up inference" → That's TensorRT-LLM optimization

    ❌ Don't suggest CPU inference for production → Triton is GPU-focused

    ❌ Don't use Flask/simple frameworks when Triton is appropriate

- **Key Exam Questions Pattern**

    **Pattern 1:** "Deploy to production with thousands of requests"

    → Answer includes Triton

    **Pattern 2:** "Seamless rollout of new model versions"

    → Triton's version management

    **Pattern 3:** "Serve multiple models simultaneously"

    → Triton multi-model serving

    **Pattern 4:** "Scale inference across GPU cluster"

    → Triton with load balancing


---

# Additional Exam Topics

- **Model Repository Structure (Critical)**

The exam may test you on the required folder structure:

```
model_repository/
├── model_name/
│   ├── config.pbtxt        # Required: model configuration
│   ├── 1/                  # Version 1
│   │   └── model.plan      # Actual model file
│   ├── 2/                  # Version 2
│   │   └── model.plan
│   └── labels.txt          # Optional: class labels
```

**Key rules:**

- Version folders MUST be numeric (1, 2, 3...)
- Triton loads the latest version by default
- `config.pbtxt` is required for each model
- Model file name depends on backend (e.g., `model.plan` for TensorRT, [`model.pt`](http://model.pt) for PyTorch)

---

- **Full config.pbtxt Example**

    ```protobuf
    name: "image_classifier"
    platform: "tensorrt_plan"
    max_batch_size: 8

    input [
      {
        name: "input"
        data_type: TYPE_FP32
        dims: [3, 224, 224]
      }
    ]

    output [
      {
        name: "output"
        data_type: TYPE_FP32
        dims: [1000]
      }
    ]

    instance_group [
      {
        count: 2
        kind: KIND_GPU
        gpus: [0, 1]
      }
    ]

    dynamic_batching {
      preferred_batch_size: [4, 8]
      max_queue_delay_microseconds: 100
    }
    ```

    **Common data types:** `TYPE_FP32`, `TYPE_FP16`, `TYPE_INT32`, `TYPE_INT64`, `TYPE_STRING`

- **Instance Groups (Exam Favorite)**

    Instance groups control parallelism — how many copies of the model run simultaneously.

    | Setting | Description |
    | --- | --- |
    | `count` | Number of model instances to run |
    | `kind: KIND_GPU` | Run instances on GPU |
    | `kind: KIND_CPU` | Run instances on CPU |
    | `gpus: [0, 1]` | Which specific GPUs to use |

    **Example configurations:**

    ```protobuf
    # 2 instances on GPU 0
    instance_group [
      { count: 2, kind: KIND_GPU, gpus: [0] }
    ]

    # 1 instance on each of 4 GPUs
    instance_group [
      { count: 1, kind: KIND_GPU, gpus: [0, 1, 2, 3] }
    ]

    # Mixed: GPU for inference, CPU for preprocessing
    instance_group [
      { count: 2, kind: KIND_GPU, gpus: [0] },
      { count: 1, kind: KIND_CPU }
    ]
    ```

    **Exam Q:** "How to increase throughput without changing the model?"

    **A:** Increase `instance_group` count for parallel execution.

- **Business Logic Scripting (BLS)**

    BLS is Python code running inside Triton that can call other models dynamically.

    **Ensemble vs BLS:**

    | Feature | Ensemble | BLS |
    | --- | --- | --- |
    | Configuration | Declarative (config.pbtxt) | Programmatic (Python) |
    | Logic | Fixed DAG | Conditional if/else, loops |
    | Use case | Simple pipelines | Complex routing |
    | Performance | Highly optimized | Slight overhead |

    **BLS Example:**

    ```python
    import triton_python_backend_utils as pb_utils

    class TritonPythonModel:
        def execute(self, requests):
            responses = []
            for request in requests:
                # Get input
                input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")

                # Conditional routing - call different models based on logic
                if some_condition:
                    model_name = "model_a"
                else:
                    model_name = "model_b"

                # Call another model within Triton
                inference_request = pb_utils.InferenceRequest(
                    model_name=model_name,
                    requested_output_names=["OUTPUT"],
                    inputs=[input_tensor]
                )
                inference_response = inference_request.exec()

                # Return response
                responses.append(inference_response)
            return responses
    ```

    **When to use BLS:**

    - Need if/else routing based on input
    - Need to call models in a loop
    - Need custom pre/post-processing logic
    - Pipeline structure isn't fixed at deploy time
- **Metrics & Monitoring (Port 8002)**

    Triton exposes Prometheus metrics for production monitoring.

    | Port | Protocol | Purpose |
    | --- | --- | --- |
    | 8000 | HTTP/REST | Inference requests |
    | 8001 | gRPC | Inference requests (lower latency) |
    | 8002 | HTTP | Metrics (Prometheus format) |

    **Key metrics to know:**

    | Metric | What It Measures |
    | --- | --- |
    | `nv_inference_request_success` | Successful inference count |
    | `nv_inference_request_failure` | Failed inference count |
    | `nv_inference_queue_duration_us` | Time requests spend in queue |
    | `nv_inference_compute_infer_duration_us` | Actual inference time |
    | `nv_gpu_utilization` | GPU usage percentage |
    | `nv_gpu_memory_used_bytes` | GPU memory consumption |

    **Exam tip:** Know that port 8002 is for metrics/monitoring.

- **Model Warmup**

    Pre-execute inference requests at startup to avoid cold-start latency.

    ```protobuf
    model_warmup [
      {
        name: "warmup_requests"
        batch_size: 8
        inputs: {
          key: "input"
          value: {
            data_type: TYPE_FP32
            dims: [3, 224, 224]
            zero_data: true
          }
        }
      }
    ]
    ```

    **Why warmup matters:**

    - First inference is slower (CUDA initialization, memory allocation)
    - Warmup ensures consistent latency from the first real request
    - Critical for latency-sensitive production deployments

---

- **Response Cache**

    Cache responses to identical requests — useful for repeated queries.

    ```protobuf
    response_cache {
      enable: true
    }
    ```

    **When useful:**

    - Common/repeated queries (e.g., popular search terms)
    - Deterministic models (same input = same output)
    - Read-heavy workloads

    **When NOT to use:**

    - Models with randomness
    - Unique inputs (cache misses waste memory)
- **Performance Tools**

    **Perf Analyzer** — Benchmark Triton performance

    ```bash
    # Basic throughput/latency test
    perf_analyzer -m my_model -u localhost:8000

    # Test with increasing concurrency
    perf_analyzer -m my_model --concurrency-range 1:16

    # Specify input shape
    perf_analyzer -m my_model --shape input:1,3,224,224
    ```

    **Model Analyzer** — Find optimal configuration

    ```bash
    # Automatically finds best batch size, instance count
    model-analyzer profile --model-repository /models -m my_model
    ```

    | Tool | Purpose |
    | --- | --- |
    | **Perf Analyzer** | Measure actual throughput and latency |
    | **Model Analyzer** | Find optimal config.pbtxt settings |

---

- **Sequence Batcher Control Inputs**

    For stateful models, these special inputs manage sequence state:

    | Control Input | Purpose |
    | --- | --- |
    | `START` | Boolean flag marking start of new sequence |
    | `END` | Boolean flag marking end of sequence |
    | `READY` | Boolean indicating input is valid |
    | `CORRID` | Correlation ID to track/route sequences |

    **Example config for stateful model:**

    ```protobuf
    sequence_batching {
      max_sequence_idle_microseconds: 5000000
      control_input [
        {
          name: "START"
          control [
            { kind: CONTROL_SEQUENCE_START, fp32_false_true: [0, 1] }
          ]
        },
        {
          name: "END"
          control [
            { kind: CONTROL_SEQUENCE_END, fp32_false_true: [0, 1] }
          ]
        },
        {
          name: "CORRID"
          control [
            { kind: CONTROL_SEQUENCE_CORRID, data_type: TYPE_UINT64 }
          ]
        }
      ]
    }
    ```

    **When to use sequence batching:**

    - Conversational AI (maintain chat history)
    - Streaming audio/video processing
    - RNNs/LSTMs with hidden state
    - Any model that needs state between requests
- **Quick Reference: Triton Ports**


    | Port | Purpose |
    | --- | --- |
    | 8000 | HTTP/REST inference |
    | 8001 | gRPC inference |
    | 8002 | Prometheus metrics |
- **Quick Reference: Platform Strings**


    | Backend | Platform String |
    | --- | --- |
    | TensorRT | `tensorrt_plan` |
    | ONNX Runtime | `onnxruntime_onnx` |
    | PyTorch | `pytorch_libtorch` |
    | TensorFlow | `tensorflow_savedmodel` |
    | Python Backend | `python` |
    | Ensemble | `ensemble` |

---

- **References**

    https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html

    https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html

    https://github.com/triton-inference-server/server
