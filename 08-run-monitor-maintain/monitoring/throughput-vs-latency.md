# Throughput vs latency

# Core Definitions

| Metric | Definition | Unit |
| --- | --- | --- |
| **Latency** | Time for ONE request to complete (start to finish) | Milliseconds (ms) |
| **Throughput** | Number of requests processed per unit time | Requests/second |

**Key insight:** These are often **inversely related** — optimizing for one can hurt the other.

---

# The Fundamental Tradeoff

```
                Latency                         Throughput
                   ▲                               ▲
                   │      /                        │           ────────
                   │     /                         │      ────/
                   │    /                          │   ──/
                   │   /                           │  /
                   │  /                            │ /
                   │ /                             │/
                   └──────────► Batch Size         └──────────► Batch Size

As batch size ↑, latency ↑ (BAD)         As batch size ↑, throughput ↑ (GOOD)
```

---

# Triton Configuration Parameters

| Parameter | What It Controls | Effect |
| --- | --- | --- |
| `max_batch_size` | Maximum requests combined in one batch | Higher = better throughput, worse latency |
| `max_queue_delay_microseconds` | How long to wait for more requests | Higher = fuller batches, longer wait |
| `preferred_batch_size` | Target batch sizes for memory optimization | Helps Triton pre-allocate efficiently |

**Example configurations:**

```
# Low-latency config (real-time chatbot)
dynamic_batching {
  max_queue_delay_microseconds: 100    # Wait max 0.1ms
  preferred_batch_size: [1, 2, 4]
}

# High-throughput config (batch processing)
dynamic_batching {
  max_queue_delay_microseconds: 500000  # Wait up to 500ms
  preferred_batch_size: [16, 32, 64]
}
```

---

# How Queue Delay Affects Latency

```
max_queue_delay: 500ms (HIGH)
──────────────────────────────────────────────────
Request arrives
    │
    ▼
┌─────────────────────────────┐
│  Wait in queue: ~400ms      │ ← Waiting for batch to fill!
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  GPU inference: ~700ms      │
└─────────────────────────────┘
    │
    ▼
Total latency: 400 + 700 = 1100ms 😢

max_queue_delay: 50ms (LOW)
──────────────────────────────────────────────────
Request arrives
    │
    ▼
┌─────────────────────────────┐
│  Wait in queue: ~30ms       │ ← Quick!
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  GPU inference: ~700ms      │
└─────────────────────────────┘
    │
    ▼
Total latency: 30 + 700 = 730ms ✓
```

---

# Solutions by Problem Type

| Problem | Solution | Why |
| --- | --- | --- |
| "Latency too high" | Mixed precision (FP16/INT8) | Reduces computation time |
| "Latency too high" | Reduce queue delay | Less waiting in queue |
| "Latency too high" | Reduce batch size | Smaller batches process faster |
| "Throughput too low" | Increase batch size | More requests per GPU cycle |
| "Throughput too low" | Increase queue delay | Fuller batches, better efficiency |
| "Throughput too low" | Add more GPU instances | More parallel capacity |

---

# P99 vs Average Latency

| Metric | Meaning |
| --- | --- |
| **Average latency** | Mean time across all requests |
| **P99 latency** | 99% of requests are faster than this |

**Exam pattern:**

```
Average: 800ms
P99: 4500ms  ← Huge gap!

Diagnosis: Queue buildup during traffic spikes
Fix: Reduce max_queue_delay or add instances
```

**"High variance in latency" or "P99 much higher than average"** → **Queueing/batching issues**

---

# CPU vs GPU for Inference

| Aspect | CPU | GPU |
| --- | --- | --- |
| Cores | 8-64 | 10,000+ |
| Matrix math | Slow | Optimized (Tensor Cores) |
| Memory bandwidth | ~50-100 GB/s | ~2000+ GB/s |
| LLM inference | Very slow | Fast |

**Example (Llama 70B):**

```
GPU (H100): ~500ms
CPU:        ~30,000-60,000ms (60-120x slower!)
```

**Exam tip:** "Switch to CPU for lower latency" is almost ALWAYS wrong for LLMs.

---

# Multiple Workloads = Multiple Configurations

When you have different requirements, deploy **separate model configurations**:

```
model_repository/
├── llm_realtime/           ← For user-facing chat
│   └── config.pbtxt
│       • max_batch_size: 8
│       • max_queue_delay: 100μs
│
└── llm_batch/              ← For overnight processing
    └── config.pbtxt
        • max_batch_size: 64
        • max_queue_delay: 500000μs
```

---

# Interpreting Metrics

| Observation | Likely Cause | Fix |
| --- | --- | --- |
| High GPU util + High latency | Over-batching (queue delay too high) | Reduce queue delay |
| Low GPU util + High latency | Bottleneck elsewhere (CPU, memory, network) | Profile to find bottleneck |
| High GPU util + Low latency | ✅ Perfect configuration! | — |
| Low GPU util + Low latency | Under-utilizing GPU | Can increase batch size |

---

# Exam Keywords Cheat Sheet

**Latency-focused (don't increase batch size):**

- "real-time"
- "latency is high"
- "response time"
- "user-facing"
- "<100ms requirement"
- "interactive"

**Throughput-focused (can increase batch size):**

- "process many requests"
- "maximize throughput"
- "batch processing"
- "offline processing"
- "overnight job"
- "bulk processing"

---

# Quick Decision Table

| Scenario | Batch Size | Queue Delay | Priority |
| --- | --- | --- | --- |
| Real-time chatbot | Small/Moderate | Low | LATENCY |
| Batch processing overnight | High | High | THROUGHPUT |
| Variable traffic | Moderate | Moderate | BALANCE |
| Multiple use cases | Two separate configs | Different settings | BOTH |

---

# Common Exam Traps

| Trap | Reality |
| --- | --- |
| "95% GPU = overloaded" | 95% GPU utilization is GOOD, not bad |
| "Increase batch for latency" | Batch size helps throughput, HURTS latency |
| "CPU for predictable latency" | CPU is 60-120x SLOWER than GPU for LLMs |
| "High throughput = low latency" | They're often inversely related |
| "Add GPUs to fix latency" | Check batching config first — often the real issue |
