# NeMo Curator: Data Preparation for LLM Training

## What is NeMo Curator?

NeMo Curator is NVIDIA's GPU-accelerated data preparation toolkit designed specifically for preparing large-scale training datasets for language models. Think of it as the foundational step before you even touch model training. It sits at the start of the pipeline and ensures your raw data is clean, deduplicated, and ready to produce high-quality models.

**Key point:** NeMo Curator is strictly for data preparation. It's NOT used for inference (that's TensorRT-LLM or NIM), and it's NOT for agent orchestration (that's NIM or other frameworks). It's the janitor of your data pipeline.

## Core Capabilities You'll See on the Exam

### Deduplication
NeMo Curator can identify and remove duplicate or near-duplicate documents at massive scale. It handles both exact deduplication (byte-for-byte matches) and fuzzy deduplication (semantically similar documents using techniques like MinHash). This matters because training on duplicates wastes compute and can lead to overfitting.

### Quality Filtering
Not all text on the internet is training-worthy. NeMo Curator filters out low-quality documents using heuristics and learned models. This keeps your training corpus clean and focused on useful content.

### PII (Personally Identifiable Information) Redaction
Before training on web-scraped data, you need to protect privacy. NeMo Curator automatically detects and redacts PII like email addresses, phone numbers, credit card numbers, and social security numbers. The exam loves asking about this.

### Language Identification
When you're pulling data from the wild, you get text in every language imaginable. NeMo Curator classifies documents by language so you can filter to the languages you actually need for your model.

### Text Extraction
Real-world data comes in messy formats: PDFs, HTML, Word documents, etc. NeMo Curator extracts clean text from these varied sources so you have consistent, usable content.

## Exam Keyword Triggers

When you see these phrases on the exam, immediately think NeMo Curator:

- "Data preparation for LLM training"
- "Deduplication of training datasets"
- "Quality filtering of documents"
- "Removing PII from training data"
- "Language identification at scale"
- "Text extraction from multiple formats"

## Common Exam Traps

**Trap 1:** Confusing NeMo Curator with AIQ Toolkit. AIQ is for observability and quality monitoring of models in production, NOT for preprocessing raw training data. NeMo Curator is the preprocessor.

**Trap 2:** Thinking NeMo Curator handles model training. It doesn't. It only cleans data. Once your data is ready, you hand it to the NeMo Framework for actual training.

## Your Pipeline Context

Here's where NeMo Curator fits in the overall NVIDIA stack:

```
Raw Data (web scrapes, documents, etc.)
    ↓
NeMo Curator (clean & deduplicate)
    ↓
NeMo Framework (train/fine-tune your LLM)
    ↓
TensorRT-LLM (optimize for inference)
    ↓
NIM (deploy inference in production)
```

## Key Takeaway

NeMo Curator is your go-to answer whenever the exam asks about preparing training data at scale. It's GPU-accelerated, handles the messy work of deduplication and quality filtering, and gets your data ready for the next stage of the pipeline.

---

## Exam-Style Practice Questions

**Question 1:** You're preparing a massive dataset of web-scraped documents for training a multilingual LLM. You need to remove documents that contain email addresses and phone numbers, identify documents by language, and eliminate near-duplicate articles. Which component should you use?

A) TensorRT-LLM
B) NeMo Curator
C) NeMo Retriever
D) NIM

**Answer:** B) NeMo Curator. This describes the exact use case NeMo Curator was built for: data cleaning (PII redaction), quality assurance (language ID), and deduplication. TensorRT-LLM optimizes models for inference, NeMo Retriever handles retrieval augmentation, and NIM deploys models.

---

**Question 2:** Your organization wants to remove semantically similar documents from your training dataset to prevent the model from overfitting on redundant content. Which NeMo Curator capability addresses this?

A) Language Identification
B) Exact Deduplication
C) Fuzzy Deduplication
D) PII Redaction

**Answer:** C) Fuzzy Deduplication. Exact deduplication only catches byte-for-byte matches. Fuzzy deduplication (using techniques like MinHash) catches semantically similar documents that aren't identical. That's what prevents overfitting on redundant content.

---

**Question 3:** Your training pipeline currently uses AIQ Toolkit to preprocess raw documents before feeding them into NeMo Framework for model training. A colleague suggests switching to NeMo Curator instead. Should you make this change?

A) No, AIQ Toolkit is designed for data preprocessing and is already in your pipeline
B) Yes, NeMo Curator is specifically designed for preprocessing training data at scale
C) No, AIQ Toolkit and NeMo Curator serve the same purpose so the change doesn't matter
D) Yes, but only if you're also using TensorRT-LLM for optimization

**Answer:** B) Yes, NeMo Curator is specifically designed for preprocessing training data at scale. AIQ Toolkit is for observability and quality monitoring of deployed models, not for preprocessing raw training data. This is a common exam trap.
