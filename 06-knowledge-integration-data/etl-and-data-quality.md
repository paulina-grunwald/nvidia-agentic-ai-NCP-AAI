# ETL Pipelines and Data Quality for Agent Knowledge Bases
**Objectives 6.3 & 6.4: Build ETL pipelines and conduct data quality checks, augmentation, and preprocessing**

ETL (Extract, Transform, Load) is critical for turning raw data into reliable knowledge for agents. This guide covers ingestion, transformation, quality assurance, and augmentation at scale.

## ETL Pipeline Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  EXTRACT    │    │  TRANSFORM   │    │  QUALITY     │    │  LOAD        │
│             │    │              │    │  ASSURANCE   │    │              │
├─────────────┤    ├──────────────┤    ├──────────────┤    ├──────────────┤
│ PDFs        │ -> │ Text extract │ -> │ Dedup        │ -> │ Vector DB    │
│ Web pages   │    │ Chunk        │    │ Validate     │    │ Metadata DB  │
│ APIs        │    │ Embed        │    │ Quality score│    │ Cache layer  │
│ Databases   │    │ Augment      │    │ Language ID  │    │              │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

## Stage 1: Data Extraction

### Document Loaders (LangChain)

```python
from langchain.document_loaders import (
    PDFLoader,
    WebBaseLoader,
    DirectoryLoader,
    TextLoader
)

# PDF extraction
pdf_loader = PDFLoader("documents/research.pdf")
pdf_docs = pdf_loader.load()

# Web scraping with metadata
web_loader = WebBaseLoader(
    web_paths=["https://example.com/article"],
    bs_kwargs={"features": "html.parser"}
)
web_docs = web_loader.load()

# Batch load directory
directory_loader = DirectoryLoader(
    path="documents/",
    glob="**/*.pdf",
    loader_cls=PDFLoader,
    show_progress=True
)
all_docs = directory_loader.load()

# Document metadata enrichment
for doc in all_docs:
    doc.metadata["source_type"] = "internal"
    doc.metadata["ingestion_date"] = datetime.now().isoformat()
```

### API-Based Data Ingestion

```python
import asyncio
from aiohttp import ClientSession
from datetime import datetime

async def ingest_from_api(api_endpoints: list) -> list:
    """Async API data ingestion"""
    documents = []

    async with ClientSession() as session:
        tasks = [
            fetch_and_process(session, endpoint)
            for endpoint in api_endpoints
        ]
        results = await asyncio.gather(*tasks)

    for result in results:
        documents.append({
            "content": result["data"],
            "metadata": {
                "source": result["url"],
                "fetch_time": datetime.now().isoformat(),
                "status": result["status_code"]
            }
        })

    return documents

async def fetch_and_process(session, url):
    """Fetch from API with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    return {
                        "data": await resp.text(),
                        "url": str(resp.url),
                        "status_code": resp.status
                    }
        except asyncio.TimeoutError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### Database Connectors

```python
import psycopg2
from sqlalchemy import create_engine

def extract_from_sql_database(connection_string: str, query: str):
    """Extract structured data and convert to documents"""
    engine = create_engine(connection_string)

    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()

    documents = []
    for row in rows:
        # Convert row to document format
        doc_content = "\n".join([
            f"{col}: {val}" for col, val in zip(result.keys(), row)
        ])
        documents.append({
            "content": doc_content,
            "metadata": {
                "source": "database",
                "table": extract_table_name(query),
                "row_id": row[0]  # Assuming first column is ID
            }
        })

    return documents
```

## Stage 2: Text Extraction and Preprocessing

### Advanced PDF Text Extraction

```python
import pymupdf  # fitz
from ocr_system import PaddleOCR  # For scanned PDFs

def extract_pdf_with_layout(pdf_path: str):
    """Preserve layout and identify tables/figures"""
    doc = pymupdf.open(pdf_path)
    extracted_blocks = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")  # Get text blocks with positions

        for block_idx, block in enumerate(blocks):
            x0, y0, x1, y1, text, block_type = block[:6]

            # Classify block type
            if block_type == 0:  # Text block
                extracted_blocks.append({
                    "type": "text",
                    "content": text,
                    "page": page_num,
                    "bbox": [x0, y0, x1, y1]
                })
            elif block_type == 1:  # Image block
                extracted_blocks.append({
                    "type": "image",
                    "page": page_num,
                    "bbox": [x0, y0, x1, y1],
                    "description": extract_image_text(page, bbox)
                })

    # For scanned PDFs, use OCR
    if len([b for b in extracted_blocks if b["type"] == "text"]) < 100:
        ocr = PaddleOCR(use_gpu=True)
        ocr_results = ocr.ocr(pdf_path)
        extracted_blocks.extend(ocr_results)

    return extracted_blocks

def extract_image_text(page, bbox):
    """Extract text from images in PDF using OCR"""
    # Crop and save image
    pix = page.get_pixmap(clip=bbox)
    # Process with vision model or OCR
    return "image_description"
```

### HTML and Web Content

```python
from bs4 import BeautifulSoup
import trafilatura  # Content extraction library

def extract_web_content(html: str):
    """Extract main content from HTML, removing boilerplate"""

    # Use trafilatura for robust extraction
    content = trafilatura.extract(html)  # Main article content
    metadata = trafilatura.extract_metadata(html)

    soup = BeautifulSoup(html, "html.parser")

    # Preserve semantic structure
    extracted = {
        "title": metadata.title or soup.title.string,
        "main_content": content,
        "author": metadata.author,
        "date": metadata.date,
        "language": metadata.language,
        "sections": extract_sections(soup)
    }

    return extracted

def extract_sections(soup):
    """Extract hierarchical structure"""
    sections = []
    for heading in soup.find_all(['h1', 'h2', 'h3']):
        level = int(heading.name[1])
        content = []

        # Collect content until next heading of same/higher level
        for sibling in heading.find_next_siblings():
            if sibling.name and sibling.name.startswith('h'):
                break
            if sibling.name in ['p', 'ul', 'ol']:
                content.append(sibling.get_text())

        sections.append({
            "heading": heading.get_text(),
            "level": level,
            "content": "\n".join(content)
        })

    return sections
```

## Stage 3: Chunking (Transform)

Text chunking is a critical transform step. Here's a production-ready approach:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

def smart_chunk(documents: List[dict], strategy: str = "semantic"):
    """
    Chunk documents with multiple strategies

    Strategies:
    - semantic: Use sentence boundaries and paragraph structure
    - fixed: Simple fixed-size chunks with overlap
    - recursive: Recursive split preserving structure
    """

    if strategy == "semantic":
        # Use sentence splitter first
        chunks = []

        for doc in documents:
            # Preserve semantic boundaries
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=512,
                chunk_overlap=64,
                length_function=len
            )

            doc_chunks = splitter.split_text(doc["content"])

            for chunk_idx, chunk in enumerate(doc_chunks):
                chunks.append({
                    "content": chunk,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_id": f"{doc['metadata'].get('source')}_{chunk_idx}",
                        "chunk_order": chunk_idx
                    }
                })

        return chunks

    elif strategy == "fixed":
        # Fixed-size chunks with overlap for table data
        chunk_size = 512
        overlap = 64
        chunks = []

        for doc in documents:
            content = doc["content"]

            for i in range(0, len(content), chunk_size - overlap):
                chunk = content[i:i + chunk_size]
                if len(chunk) > 100:  # Minimum chunk size
                    chunks.append({
                        "content": chunk,
                        "metadata": {
                            **doc["metadata"],
                            "char_start": i,
                            "char_end": i + len(chunk)
                        }
                    })

        return chunks

# Best practice: Dynamic chunk sizing based on content
def dynamic_chunk_size(content: str, content_type: str):
    """Adjust chunk size by content type"""

    chunk_sizes = {
        "research_paper": 1024,
        "code_sample": 256,
        "documentation": 512,
        "legal_document": 768,
        "chat_log": 256
    }

    return chunk_sizes.get(content_type, 512)
```

## Stage 4: Embedding Generation Pipeline

```python
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

class EmbeddingPipeline:
    """Production embedding pipeline with batching and GPU optimization"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def embed_documents(self, chunks: List[dict]) -> List[dict]:
        """Embed chunks with progress tracking"""

        texts = [chunk["content"] for chunk in chunks]
        embeddings = []

        # Process in batches for memory efficiency
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i + self.batch_size]

            # Generate embeddings
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=self.batch_size
            )
            embeddings.extend(batch_embeddings)

        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()  # Convert to list for storage
            chunk["embedding_model"] = self.model.get_sentence_embedding_dimension()

        return chunks

# NVIDIA NIM for Embeddings (Optimized)
async def embed_with_nim(chunks: List[dict]) -> List[dict]:
    """Use NVIDIA NIM endpoint for embeddings (distributed, batched)"""

    import aiohttp

    NIM_ENDPOINT = "http://localhost:8000/v1/embeddings"

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(chunks), 256):
            batch = chunks[i:i + 256]
            texts = [c["content"] for c in batch]

            async with session.post(
                NIM_ENDPOINT,
                json={"input": texts, "model": "nvidia-embed-qa-4"}
            ) as resp:
                result = await resp.json()

                for chunk, embedding in zip(batch, result["data"]):
                    chunk["embedding"] = embedding["embedding"]

    return chunks
```

## Stage 5: Data Quality Assurance

### Quality Checks and Validation

```python
from statistics import mean, stdev
from langdetect import detect_langs
import hashlib

class QualityAssurance:
    """Comprehensive data quality pipeline"""

    def __init__(self):
        self.quality_scores = []

    def validate_chunk(self, chunk: dict) -> dict:
        """Run all quality checks on a chunk"""

        content = chunk["content"]
        quality_report = {
            "chunk_id": chunk["metadata"]["chunk_id"],
            "checks": {}
        }

        # 1. Length validation
        quality_report["checks"]["length"] = {
            "pass": 100 <= len(content) <= 2000,
            "value": len(content),
            "reason": "Content should be between 100-2000 chars"
        }

        # 2. Duplicate detection (within batch)
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        quality_report["checks"]["hash"] = {
            "value": content_hash,
            "pass": True  # Check later in batch
        }

        # 3. Language detection
        try:
            langs = detect_langs(content)
            primary_lang = langs[0].lang if langs else "unknown"
            quality_report["checks"]["language"] = {
                "pass": primary_lang in ["en", "es", "fr"],
                "value": primary_lang,
                "confidence": langs[0].prob if langs else 0
            }
        except:
            quality_report["checks"]["language"] = {
                "pass": False,
                "error": "Could not detect language"
            }

        # 4. Encoding quality
        try:
            content.encode("utf-8")
            quality_report["checks"]["encoding"] = {"pass": True}
        except:
            quality_report["checks"]["encoding"] = {
                "pass": False,
                "error": "Invalid UTF-8 encoding"
            }

        # 5. Content completeness (check for common truncation)
        truncation_markers = [
            "...",
            "[continued",
            "See more",
            "Read more"
        ]
        quality_report["checks"]["completeness"] = {
            "pass": not any(marker in content for marker in truncation_markers),
            "truncation_detected": any(
                marker in content for marker in truncation_markers
            )
        }

        # 6. Minimal coherence check
        sentences = len([s for s in content.split(".") if s.strip()])
        avg_word_length = mean([len(w) for w in content.split()])

        quality_report["checks"]["coherence"] = {
            "pass": sentences >= 2 and 3 <= avg_word_length <= 12,
            "sentence_count": sentences,
            "avg_word_length": avg_word_length
        }

        # Calculate overall quality score
        checks_passed = sum(
            1 for check in quality_report["checks"].values()
            if check.get("pass", False)
        )
        total_checks = len(quality_report["checks"])
        quality_report["quality_score"] = checks_passed / total_checks

        return quality_report

# Batch deduplication
def deduplicate_chunks(chunks: List[dict]) -> List[dict]:
    """Remove exact and near-duplicate chunks"""

    seen_hashes = set()
    deduped = []

    for chunk in chunks:
        content_hash = hashlib.md5(
            chunk["content"].encode()
        ).hexdigest()

        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            deduped.append(chunk)

    # Optional: Remove semantic duplicates using embeddings
    # This is expensive but valuable for large document sets

    return deduped
```

## Stage 6: Data Augmentation

### Synthetic Data Generation with NVIDIA NeMo Curator

```python
from nemo_curator import (
    Curator,
    DomainConditionalRedactorFilter,
    DocumentFilter,
    RedundancyFilters
)

def augment_with_nemo_curator(data_path: str):
    """Use NVIDIA NeMo Curator for large-scale data preparation"""

    # Initialize curator
    curator = Curator()

    # Apply filters
    curator.add_filter(
        DomainConditionalRedactorFilter()  # Remove sensitive data
    )
    curator.add_filter(
        DocumentFilter(
            min_length=100,
            max_length=10000,
            language="en"
        )
    )
    curator.add_filter(
        RedundancyFilters()  # Remove near-duplicates
    )

    # Process data
    curated_data = curator.curate(data_path)

    return curated_data

# Synthetic data generation for augmentation
def generate_synthetic_qa(chunks: List[dict]) -> List[dict]:
    """Generate QA pairs from chunks for training/augmentation"""

    from transformers import pipeline

    qa_generator = pipeline("text2text-generation", model="google/flan-t5-large")

    synthetic_pairs = []

    for chunk in chunks:
        # Generate question from content
        prompt = f"Generate a question that can be answered by: {chunk['content'][:200]}"

        question = qa_generator(prompt, max_length=100)[0]["generated_text"]

        synthetic_pairs.append({
            "question": question,
            "answer": chunk["content"],
            "type": "synthetic",
            "metadata": chunk["metadata"]
        })

    return synthetic_pairs

# Add synthetic queries for better embedding models
def augment_with_query_variations(chunks: List[dict]) -> List[dict]:
    """Generate query variations to improve retrieval"""

    augmented = chunks.copy()

    for chunk in chunks:
        # Create variations of the content as potential queries
        sentences = chunk["content"].split(".")

        if len(sentences) > 1:
            for i, sentence in enumerate(sentences[:-1]):
                augmented.append({
                    "content": sentence.strip(),
                    "metadata": {
                        **chunk["metadata"],
                        "augmentation_type": "query_variation",
                        "original_chunk": chunk["metadata"]["chunk_id"]
                    }
                })

    return augmented
```

## Complete ETL Pipeline Example

```python
from datetime import datetime
import json

class KnowledgeBasePipeline:
    """End-to-end ETL pipeline for agent knowledge bases"""

    def __init__(self, vector_db, metric_db):
        self.vector_db = vector_db
        self.metric_db = metric_db
        self.qa = QualityAssurance()
        self.embedder = EmbeddingPipeline()

    def run(self, source: str, source_type: str):
        """Execute full pipeline"""

        pipeline_run_id = datetime.now().isoformat()
        metrics = {
            "run_id": pipeline_run_id,
            "stages": {}
        }

        # Extract
        print("STAGE 1: Extracting data...")
        if source_type == "pdf":
            docs = PDFLoader(source).load()
        elif source_type == "web":
            docs = WebBaseLoader(source).load()
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        metrics["stages"]["extract"] = {
            "document_count": len(docs),
            "total_bytes": sum(len(d.page_content) for d in docs)
        }

        # Transform: Chunk
        print("STAGE 2: Chunking documents...")
        chunks = smart_chunk(
            [{"content": d.page_content, "metadata": d.metadata} for d in docs]
        )
        metrics["stages"]["chunk"] = {"chunk_count": len(chunks)}

        # Quality checks
        print("STAGE 3: Quality assurance...")
        quality_reports = [self.qa.validate_chunk(c) for c in chunks]

        quality_stats = {
            "total_chunks": len(chunks),
            "passing_chunks": sum(
                1 for r in quality_reports if r["quality_score"] > 0.7
            ),
            "avg_score": sum(r["quality_score"] for r in quality_reports) / len(quality_reports)
        }

        metrics["stages"]["quality"] = quality_stats

        # Filter low-quality
        chunks = [
            c for c, r in zip(chunks, quality_reports)
            if r["quality_score"] > 0.7
        ]

        # Deduplicate
        print("STAGE 4: Deduplication...")
        original_count = len(chunks)
        chunks = deduplicate_chunks(chunks)
        metrics["stages"]["dedup"] = {
            "original": original_count,
            "after_dedup": len(chunks),
            "duplicates_removed": original_count - len(chunks)
        }

        # Generate embeddings
        print("STAGE 5: Generating embeddings...")
        chunks = self.embedder.embed_documents(chunks)

        # Load into vector DB
        print("STAGE 6: Loading into vector DB...")
        for chunk in chunks:
            self.vector_db.upsert(
                id=chunk["metadata"]["chunk_id"],
                embedding=chunk["embedding"],
                metadata=chunk["metadata"],
                text=chunk["content"]
            )

        metrics["stages"]["load"] = {
            "vectors_loaded": len(chunks),
            "timestamp": datetime.now().isoformat()
        }

        # Record metrics
        self.metric_db.insert("pipeline_runs", metrics)

        print(f"\nPipeline Complete! Summary:")
        print(json.dumps(metrics, indent=2))

        return metrics
```

## Data Quality Metrics Dashboard

```python
# Track metrics for monitoring
quality_metrics = {
    "extraction_rate": len(docs) / num_sources,
    "chunk_coverage": total_chars / original_chars,
    "quality_score": avg_quality_score,
    "dedup_ratio": (original - deduped) / original,
    "embedding_coverage": embedded / total_chunks,
    "ingestion_latency": end_time - start_time,
    "cost_per_embedding": gpu_cost / total_chunks
}
```

## Study Questions

**Q1:** Your PDF extraction gets garbage text from scanned documents. What's the best fix?

A) Increase chunk size to 1024 tokens
B) Add OCR layer (PaddleOCR or Tesseract) to detect and process image-based text
C) Switch to web scraping instead
D) Use fixed-size chunking instead of semantic chunking

**Answer: B** - Scanned PDFs require OCR. PaddleOCR with GPU acceleration detects text in images before chunking.

---

**Q2:** After ETL, your embeddings show 30% duplicate content. What should you fix in the pipeline?

A) Reduce chunk overlap from 64 to 0
B) Add deduplication earlier (after extraction or chunking)
C) Use a different embedding model
D) Increase chunk size

**Answer: B** - Deduplication should happen early, after extraction/chunking but before embedding (saves compute). Hash-based dedup catches exact duplicates; use embedding-based dedup for semantic duplicates.

---

**Q3:** Which preprocessing step directly improves vector search recall?

A) Language detection and filtering to English only
B) Removing stopwords before chunking
C) Adding synthetic QA pairs as augmentation
D) Encoding validation (UTF-8 check)

**Answer: C** - Synthetic QA pairs add diverse retrieval paths. Query variations of your content help embeddings learn multiple ways to express the same concept.

---

**Q4:** Your ETL pipeline processes 1M documents daily but gets bottlenecked at embedding generation. What's the best optimization?

A) Use smaller embedding model
B) Reduce batch size for more frequent updates
C) Use NVIDIA NIM endpoint with async batching
D) Switch to GPU-accelerated RAPIDS for preprocessing

**Answer: C** - NVIDIA NIM provides distributed, optimized embedding infrastructure. Async batching to 256 queries per request maximizes throughput without latency penalty.
