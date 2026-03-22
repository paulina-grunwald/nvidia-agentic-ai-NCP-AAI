# Multimodal Model Integration (Objective 2.2)

Integrating generative and multimodal models (text, vision, audio) into agent systems.

---

## What Are Multimodal Models?

A multimodal model can process and/or generate more than one type of data. Instead of just text in, text out, these models handle combinations of text, images, audio, and video.

Examples:
- **Text + Image in, Text out**: "Describe what's in this photo" (GPT-4V, LLaVA)
- **Text in, Image out**: "Generate an image of a sunset over mountains" (SDXL, DALL-E)
- **Audio in, Text out**: "Transcribe this audio clip" (Whisper, NVIDIA Riva ASR)
- **Text in, Audio out**: "Read this text aloud" (NVIDIA Riva TTS)

For agents, multimodal capability means the agent can see images, hear audio, generate visuals, and speak, not just read and write text.

---

## Model Types by Modality

### Text Models (LLMs)

The backbone of most agents. Handle reasoning, planning, tool calling.

| Model | Provider | Key Feature |
|---|---|---|
| Llama 3.x / Nemotron | Meta / NVIDIA | Open-weight, NIM-optimized |
| Mixtral | Mistral | Mixture of experts, efficient |
| GPT-4 | OpenAI | Strong reasoning, function calling |

NVIDIA NIM provides optimized inference for these models as microservices.

### Vision Models

Two categories: understanding (image → text) and generation (text → image).

**Vision-Language Models (VLMs)** - understand images:

| Model | What It Does | Use Case |
|---|---|---|
| LLaVA | Image + text → text | Visual Q&A, image description |
| NVIDIA VILA | Vision-language model | Enterprise visual understanding |
| Phi-3-vision | Lightweight VLM | Edge deployment visual tasks |
| Florence-2 | Vision foundation model | Object detection, captioning, grounding |

**Image Generation Models**:

| Model | What It Does | Use Case |
|---|---|---|
| Stable Diffusion XL | Text → image | Creative content, prototyping |
| SDXL Turbo | Fast text → image | Real-time generation |

**How agents use vision models**:
```
User uploads photo of damaged product
   ↓
Agent uses VLM: "Describe the damage in this image"
   ↓
VLM returns: "Cracked screen, top-left corner, approximately 3cm crack"
   ↓
Agent uses text LLM to file warranty claim with damage description
```

### Audio Models

**Speech-to-Text (ASR - Automatic Speech Recognition)**:

| Model/Service | Provider | Key Feature |
|---|---|---|
| NVIDIA Riva ASR | NVIDIA | Low-latency streaming, GPU-optimized |
| Whisper | OpenAI | Multilingual, robust |
| Parakeet | NVIDIA | CTC/RNNT models for ASR |

**Text-to-Speech (TTS)**:

| Model/Service | Provider | Key Feature |
|---|---|---|
| NVIDIA Riva TTS | NVIDIA | Natural voices, streaming, GPU-optimized |
| Bark | Suno | Multilingual, emotional speech |

**NVIDIA Riva** is the key platform here. It provides:
- Streaming ASR with low latency (important for real-time voice agents)
- Custom vocabulary support (domain-specific terms)
- Speaker diarization (who said what)
- Natural TTS with multiple voices
- Runs on NVIDIA GPUs, deploys as NIM microservices

### Embedding Models

Convert data into vector representations for retrieval:

| Model | Modality | Use Case |
|---|---|---|
| NV-Embed | Text → vectors | Text retrieval, RAG |
| CLIP | Text + Image → vectors | Cross-modal search (find images with text queries) |
| NV-DINOv2 | Image → vectors | Visual similarity search |

CLIP is important for agents: it lets you search images using text queries and vice versa. An agent could "search our product catalog for items similar to this photo."

---

## Multimodal Agent Pipelines

### Pattern 1: Vision-Augmented Agent

Agent can see images and reason about them.

```
User: "What's wrong with this circuit board?" + [image]
   ↓
1. Image sent to VLM (LLaVA / NVIDIA VILA via NIM)
2. VLM returns description: "Burnt capacitor near C3, solder bridge between R5-R6"
3. Text LLM reasons about the description
4. Agent searches knowledge base for repair procedures
5. Agent responds with diagnosis and repair steps
```

Implementation with LangChain:
```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Multimodal LLM that accepts images
llm = ChatNVIDIA(model="nvidia/vila")

# Create message with image
from langchain_core.messages import HumanMessage

message = HumanMessage(
    content=[
        {"type": "text", "text": "What defects do you see in this circuit board?"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
    ]
)

response = llm.invoke([message])
```

### Pattern 2: Voice-Enabled Agent

Full voice-in, voice-out agent pipeline.

```
User speaks: "What's the status of my order?"
   ↓
1. Riva ASR converts speech → text
2. Text sent to agent (LLM + tools)
3. Agent queries order database
4. Agent generates text response
5. Riva TTS converts text → speech
6. User hears: "Your order #4521 shipped yesterday, expected delivery Friday"
```

```
[Microphone] → [Riva ASR] → [Agent LLM] → [Riva TTS] → [Speaker]
                                  ↓↑
                            [Tools / APIs]
```

Key considerations:
- **Latency**: Voice interactions need to feel natural (<500ms response time ideal)
- **Streaming**: ASR should stream partial results, TTS should stream audio chunks
- **Turn-taking**: Need to detect when user stops speaking (endpointing)
- **Interruption**: User should be able to interrupt agent mid-speech (barge-in)

### Pattern 3: Multimodal RAG

Retrieve and reason over documents that contain both text and images (PDFs, slides, web pages).

```
User: "Based on last quarter's report, what caused the revenue dip?"
   ↓
1. Retrieve relevant pages from PDF (text + charts/graphs)
2. Send text chunks to text LLM
3. Send chart/graph images to VLM
4. VLM extracts data from charts: "Revenue dropped 15% in August"
5. Text LLM combines text and chart insights
6. Agent provides comprehensive answer
```

This is what the blueprint reading "Building Multimodal AI RAG With LlamaIndex, NVIDIA NIM, and Milvus" covers:

```python
# LlamaIndex multimodal RAG with NVIDIA NIM
from llama_index.multi_modal_llms.nvidia import NVIDIAMultiModal
from llama_index.vector_stores.milvus import MilvusVectorStore

# Text embeddings
text_embed = NVIDIAEmbedding(model="nvidia/nv-embedqa-e5-v5")

# Image embeddings using CLIP
image_embed = NVIDIAEmbedding(model="nvidia/clip")

# Multimodal LLM for reasoning over text + images
mm_llm = NVIDIAMultiModal(model="nvidia/vila")

# Store both text and image embeddings in Milvus
vector_store = MilvusVectorStore(dim=1024, collection_name="multimodal_docs")
```

### Pattern 4: Document Understanding Agent

Agent that processes mixed documents (PDFs with text, tables, images):

```
Upload: Company financial report (PDF)
   ↓
1. Extract text → text chunks
2. Extract tables → structured data
3. Extract charts/images → VLM description
4. All chunks embedded and stored in vector DB
5. Agent can now answer questions using all modalities
```

---

## NVIDIA Multimodal NIMs

NVIDIA provides ready-to-deploy NIMs for different modalities:

| NIM Category | Models | What It Does |
|---|---|---|
| **LLM** | Llama 3.x, Nemotron, Mixtral | Text reasoning and generation |
| **VLM** | VILA, LLaVA | Image understanding |
| **Embedding** | NV-Embed, NV-RerankQA | Text/image vectorization |
| **ASR** | Riva ASR, Parakeet | Speech to text |
| **TTS** | Riva TTS | Text to speech |
| **Image Gen** | SDXL | Text to image |

All NIMs:
- Deploy as Docker containers
- Expose OpenAI-compatible API endpoints
- Run on NVIDIA GPUs with TensorRT-LLM optimization
- Can be composed into pipelines

### Composing NIMs into a Multimodal Agent

```
┌─────────────────────────────────────────────────────┐
│                   Agent Orchestrator                 │
│               (LangGraph / AutoGen)                  │
├──────┬──────────┬──────────┬──────────┬─────────────┤
│ LLM  │   VLM    │   ASR    │   TTS    │  Embedding  │
│ NIM  │   NIM    │   NIM    │   NIM    │    NIM      │
│      │          │          │          │             │
│Llama │  VILA    │  Riva    │  Riva    │  NV-Embed   │
│  3   │          │  ASR     │  TTS     │             │
└──────┴──────────┴──────────┴──────────┴─────────────┘
              All running on NVIDIA GPUs
```

Each NIM is independent, so you can:
- Scale each modality separately (more ASR instances during peak voice traffic)
- Swap models without changing the agent (upgrade VILA to a newer version)
- Mix and match (use Riva ASR + third-party TTS if needed)

---

## Multimodal Input Processing

When an agent receives multimodal input, it needs to route each modality to the right model:

```python
def process_input(user_input):
    results = {}

    if user_input.has_text:
        results["text"] = user_input.text

    if user_input.has_image:
        # Send to VLM for understanding
        results["image_description"] = vlm.describe(user_input.image)

    if user_input.has_audio:
        # Send to ASR for transcription
        results["transcription"] = asr.transcribe(user_input.audio)

    # Combine all modality results into a unified text context
    combined_context = format_multimodal_context(results)

    # Send to main LLM agent for reasoning
    response = agent_llm.invoke(combined_context)
    return response
```

The key insight: most agent reasoning still happens in a text LLM. Other modalities get converted into text representations that the LLM can reason about. The VLM describes images as text. ASR converts speech to text. Then the text LLM handles the actual planning and reasoning.

---

## Design Considerations

From the blueprint reading "Design Considerations of Advanced Agentic AI for Real-World Applications":

1. **Latency budget**: Each modality adds processing time. Budget your latency:
   - ASR: 100-500ms for streaming transcription
   - VLM: 500ms-2s for image understanding
   - LLM reasoning: 1-5s
   - TTS: 200-500ms for first audio chunk
   - Total voice pipeline: aim for <2s end-to-end

2. **Error propagation**: If ASR misheard something, the whole pipeline gets bad input. Build in:
   - Confidence scores from ASR (low confidence → ask user to repeat)
   - VLM uncertainty detection (blurry image → ask for better photo)
   - Validation steps before acting on multimodal input

3. **Context management**: Multimodal inputs use more tokens. An image description might be 200+ tokens. Budget context window carefully.

4. **Fallback strategies**: If VLM is unavailable, can the agent still work with text-only? Design graceful degradation.

---

## Exam-Style Questions

**Q1: An agent needs to answer questions about PDF reports that contain text, tables, and charts. Which approach handles all three?**
- A) Text-only RAG with PDF text extraction
- B) Multimodal RAG with text extraction + VLM for charts/images
- C) OCR on the entire document
- D) Convert everything to images and use VLM only

**Answer: B** - Multimodal RAG processes text as text (efficient) and uses VLM specifically for visual elements like charts and images that text extraction would miss.

**Q2: In a voice-enabled agent pipeline, what is the correct order of components?**
- A) TTS → LLM → ASR
- B) ASR → LLM → TTS
- C) LLM → ASR → TTS
- D) ASR → TTS → LLM

**Answer: B** - ASR converts user speech to text, LLM processes and generates text response, TTS converts response to speech.

**Q3: Which NVIDIA platform provides low-latency streaming speech-to-text and text-to-speech for voice agents?**
- A) NVIDIA Triton
- B) NVIDIA NeMo
- C) NVIDIA Riva
- D) NVIDIA TensorRT

**Answer: C** - NVIDIA Riva provides GPU-optimized streaming ASR and TTS specifically designed for conversational AI.

**Q4: In a multimodal agent, how does image understanding typically integrate with the main reasoning LLM?**
- A) The LLM directly processes raw image pixels
- B) A VLM converts the image to a text description, which the LLM reasons about
- C) Images are stored in the vector database as-is
- D) The agent ignores images and only uses text

**Answer: B** - VLMs convert images to text descriptions. The main LLM then reasons over these text descriptions alongside other text context. Most agent reasoning still happens in text space.

**Q5: A company wants to build an agent that can search their product catalog using either text queries OR photos of products. Which embedding model enables this?**
- A) NV-Embed (text only)
- B) CLIP (cross-modal text + image embeddings)
- C) Whisper (audio embeddings)
- D) LLaVA (vision-language model)

**Answer: B** - CLIP produces embeddings in a shared space for both text and images, enabling cross-modal search (text query finds matching images, image query finds matching text descriptions).

---

## Key Takeaways for NVIDIA Certification

1. Multimodal agents combine text, vision, and audio models in a single pipeline
2. Most agent reasoning stays in the text LLM. Other modalities get converted to text (VLM describes images, ASR transcribes audio)
3. NVIDIA Riva handles ASR + TTS for voice agents (streaming, low-latency, GPU-optimized)
4. NVIDIA NIMs available for LLM, VLM, embedding, ASR, TTS, and image generation
5. Multimodal RAG (text + image retrieval) handles real-world documents with charts and diagrams
6. CLIP enables cross-modal search (search images with text, search text with images)
7. Latency budget is critical for multimodal pipelines - each modality adds processing time
8. LlamaIndex + NVIDIA NIM + Milvus is the reference stack for multimodal RAG (blueprint reading)
