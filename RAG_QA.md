# RAG (Retrieval-Augmented Generation) — Deep-Dive Q&A

> Code examples are drawn directly from the three projects in this workspace:
> `agentic-rag`, `enterprise-rag-bot`, and `i4ms-chatbot`.

---

## Table of Contents
1. [What is RAG and why use it?](#1-what-is-rag-and-why-use-it)
2. [Why RAG over fine-tuning?](#2-why-rag-over-fine-tuning)
3. [How does embedding work?](#3-how-does-embedding-work)
4. [Embedding model choice — all-MiniLM-L6-v2 vs OpenAI](#4-embedding-model-choice)
5. [Why 384 dimensions?](#5-why-384-dimensions)
6. [Why cosine similarity?](#6-why-cosine-similarity)
7. [Document chunking strategy](#7-document-chunking-strategy)
8. [Dense retrieval with PGVector](#8-dense-retrieval-with-pgvector)
9. [Sparse retrieval with BM25](#9-sparse-retrieval-with-bm25)
10. [Hybrid retrieval and score fusion](#10-hybrid-retrieval-and-score-fusion)
11. [Min-max normalization vs RRF](#11-min-max-normalization-vs-rrf)
12. [Cross-encoder reranking](#12-cross-encoder-reranking)
13. [Hallucination prevention](#13-hallucination-prevention)
14. [Evaluation — RAGAS metrics](#14-evaluation--ragas-metrics)
15. [Scaling to 1M+ documents](#15-scaling-to-1m-documents)
16. [Document updates without full rebuild](#16-document-updates-without-full-rebuild)
17. [Full document lifecycle walkthrough](#17-full-document-lifecycle-walkthrough)

---

## 1. What is RAG and why use it?

**Q: What is Retrieval-Augmented Generation?**

RAG combines a retrieval step with an LLM generation step. Instead of asking the LLM to answer from its parametric memory alone, you first retrieve the most relevant text chunks from your own document store, then inject them into the prompt as grounding context. The LLM answers only from what it was given.

```
User query
    │
    ▼
[Embed query] → [Search vector store] → [Retrieve top-K chunks]
                                                  │
                                                  ▼
                               [Prompt = system + context chunks + query]
                                                  │
                                                  ▼
                                          [LLM generates answer]
```

**Why you need it:** LLMs hallucinate when asked about private, recent, or domain-specific information not in their training data. RAG grounds answers in real documents you control.

**Follow-up questions:**
- What happens if the retrieval step fails to find relevant chunks?
- How is RAG different from in-context learning?
- At what point does RAG break down and something else (fine-tuning, agent) become necessary?

---

## 2. Why RAG over fine-tuning?

**Q: You could have fine-tuned the LLM on your documents. Why use RAG instead?**

| Dimension | RAG | Fine-tuning |
|---|---|---|
| Update frequency | Add a document in seconds | Retrain takes hours/days |
| Cost | Inference + retrieval | GPU compute for training |
| Interpretability | Can show source chunks | Black-box parametric |
| Knowledge boundary | Explicit (your docs) | Blurry (mixed with base training) |
| Hallucination risk | Low (grounded context) | High (model "remembers" incorrectly) |

For enterprise policy/knowledge-base use cases (all three projects here), documents change frequently and users need to cite sources. RAG wins on every dimension.

**Follow-up questions:**
- When would you fine-tune instead of (or in addition to) RAG?
- How do you handle cases where the LLM's parametric knowledge contradicts the retrieved context?

---

## 3. How does embedding work?

**Q: Walk me through how text becomes a vector.**

An embedding model (a transformer encoder) processes a text string and produces a fixed-length dense vector. Semantically similar texts end up close in this vector space.

```python
# enterprise-rag-bot/rag/embeddings.py
from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # unit-length vectors
    )
```

`normalize_embeddings=True` is critical: it L2-normalizes each vector to unit length, so cosine similarity reduces to a simple dot product — which is faster and what PGVector's inner-product operator can accelerate with indexing.

**Follow-up questions:**
- Why must you use the same embedding model at ingestion time and query time?
- What happens if you normalize vs don't normalize?
- How does the all-MiniLM architecture differ from OpenAI's embedding models?

---

## 4. Embedding model choice

**Q: Why did `enterprise-rag-bot` use `all-MiniLM-L6-v2` while `agentic-rag` uses OpenAI `text-embedding-3-small`?**

**all-MiniLM-L6-v2 (enterprise-rag-bot):**
- Runs locally, zero API cost per embedding
- 80 MB model, CPU-friendly
- 384-dimensional output
- Good quality for English sentence-level similarity
- Offline-capable (no external dependency)

**text-embedding-3-small (agentic-rag):**
- 1536-dimensional output — richer representation
- Slightly better benchmark scores on MTEB
- Managed by OpenAI — no model loading overhead
- API cost per token (cost scales with corpus size)

**When to choose which:**
- Large document corpus + cost sensitivity + privacy → local model
- Highest quality + already paying for OpenAI + small corpus → API model

**Follow-up questions:**
- If you switched from MiniLM to OpenAI embeddings mid-production, what would you need to do?
- How would you A/B test which embedding model gives better retrieval quality?
- What is MTEB and why does it matter for model selection?

---

## 5. Why 384 dimensions?

**Q: Why is the embedding vector 384-dimensional? What does that number mean?**

384 is the output size of the `all-MiniLM-L6-v2` model — an architectural choice made by the model authors when they distilled the larger BERT-based model down to a 6-layer, 384-hidden-dim architecture. You don't choose this number; it comes with the model.

What matters is the tradeoff:
- More dimensions → richer representation → larger storage and slower ANN search
- Fewer dimensions → faster search → potentially worse retrieval quality

`all-MiniLM-L6-v2` at 384d hits a sweet spot: competitive quality at a fraction of the memory and compute cost of 768d or 1536d models.

**Storage math:** 1M documents × 5 chunks/doc × 384 floats × 4 bytes = ~7.7 GB of raw vectors.

**Follow-up questions:**
- Can you reduce dimensionality with PCA after embedding? What would you lose?
- How does OpenAI's `text-embedding-3-small` at 1536d compare in practice?
- Why doesn't simply using a larger dimension always mean better retrieval?

---

## 6. Why cosine similarity?

**Q: Why cosine similarity for vector search? Why not Euclidean distance?**

With `normalize_embeddings=True`, all vectors are unit-length. At unit length:
- **Cosine similarity = dot product** (they are mathematically identical)
- Euclidean distance between two unit vectors = `sqrt(2 - 2·cos(θ))` — it captures the same ordering but with extra computation

So for normalized embeddings, cosine similarity, dot product, and Euclidean distance all rank documents identically. The choice of "cosine" is effectively the library default — `langchain_postgres.PGVector` uses cosine distance unless you explicitly set `distance_strategy`.

For **unnormalized** embeddings (e.g., if you set `normalize_embeddings=False`), cosine similarity is preferred because it is length-invariant — a long document and a short document encoding the same topic get the same score, whereas Euclidean distance would penalize the longer vector just because of its magnitude.

**Follow-up questions:**
- When would Euclidean distance be preferable to cosine similarity?
- What is the `distance_strategy` parameter in PGVector and what are its options?
- Why does normalizing vectors make cosine similarity equivalent to dot product?

---

## 7. Document chunking strategy

**Q: How did you decide on chunk size and overlap? What splitter did you use?**

All three projects use `RecursiveCharacterTextSplitter` from LangChain with chunk sizes of 800 characters and overlaps of 120–150 characters.

```python
# enterprise-rag-bot/ingestion/splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
```

**Why `RecursiveCharacterTextSplitter`:**
It tries split points in order: `\n\n` (paragraph break) → `\n` (line break) → `.` (sentence end) → ` ` (word boundary) → character. This preserves semantic units. A naive fixed-length splitter would cut mid-sentence.

**Why 800 characters:**
- `all-MiniLM-L6-v2` has a 256-token limit; 800 chars ≈ 150–200 tokens — safely under the limit
- Small enough for accurate embedding; large enough to hold a complete idea
- Empirically good for policy-document retrieval (typically one paragraph)

**Why 120–150 character overlap:**
If a key sentence spans the boundary between chunk N and chunk N+1, the overlap ensures that sentence appears in full in at least one chunk. 150 chars ≈ 1–2 sentences.

**Follow-up questions:**
- How would you tune chunk size for a different document type (e.g., code files, legal contracts)?
- What is the tradeoff between larger chunks and retrieval precision?
- How does chunk size interact with the LLM's context window limit?

---

## 8. Dense retrieval with PGVector

**Q: How does PGVector store and search embeddings?**

PGVector is a PostgreSQL extension that adds a `vector` column type and similarity-search operators. LangChain's `PGVector` class wraps it so you can call `.similarity_search()` without writing SQL.

```python
# agentic-rag/app/rag/vector_store.py (simplified)
from langchain_postgres import PGVector
from app.rag.embeddings import get_embeddings

def get_vector_store() -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name="document_chunks",
        connection=settings.database_url,
    )

def vector_search(query: str, top_k: int) -> list[RetrievedChunk]:
    store = get_vector_store()
    results = store.similarity_search_with_score(query, k=top_k)
    return [
        RetrievedChunk(
            document_id=doc.metadata["chunk_id"],
            content=doc.page_content,
            vector_score=float(score),
            metadata=doc.metadata,
        )
        for doc, score in results
    ]
```

**Current gap — no ANN index:** None of the three projects create an HNSW or IVFFlat index. PGVector defaults to exact search (brute-force), which is O(n) per query. For collections under ~100K chunks this is acceptable; above that you need an ANN index.

```sql
-- What you'd add for production at scale:
CREATE INDEX ON langchain_pg_embedding
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**Follow-up questions:**
- What is the difference between HNSW and IVFFlat indexes in pgvector?
- At what document count does exact search become a latency problem?
- What are the parameters `m` and `ef_construction` in HNSW?

---

## 9. Sparse retrieval with BM25

**Q: What is BM25 and why combine it with vector search?**

BM25 (Best Matching 25) is a probabilistic keyword-ranking algorithm. It scores documents by how often a query term appears in them (TF) adjusted by how common the term is across all documents (IDF), with a saturation factor to prevent very frequent terms from dominating.

**Why it complements vector search:**
- Vector search understands *meaning*: "time off" matches "leave policy" because the vectors are close
- BM25 understands *exact terms*: searching "PTO" finds documents that literally say "PTO" even if the vector of "PTO" isn't close to "paid time off"
- Together they handle both semantic intent and keyword precision

```python
# enterprise-rag-bot/rag/retriever.py
from langchain_community.retrievers import BM25Retriever

def _build_bm25_retriever(top_k: int = 5) -> BM25Retriever:
    docs = _load_all_documents()  # load ALL chunks from PGVector collection

    if not docs:
        # BM25Retriever.from_documents() requires at least one document
        return BM25Retriever.from_documents(
            [Document(page_content="placeholder")], k=top_k
        )

    return BM25Retriever.from_documents(docs, k=top_k)
```

**Critical limitation:** The BM25 index is built in memory at startup. New documents ingested after startup won't appear in BM25 results until `refresh_retriever()` is called.

```python
def refresh_retriever():
    get_retriever.cache_clear()  # forces BM25 rebuild on next call
```

**Follow-up questions:**
- What are the BM25 parameters k1 and b, and how do they affect scoring?
- How would you replace the in-memory BM25 with a persistent solution (Elasticsearch)?
- Why does the BM25 index become stale after ingestion, but the vector index doesn't?

---

## 10. Hybrid retrieval and score fusion

**Q: Walk me through how the two retrievers are combined.**

There are two different fusion approaches across the three projects:

**agentic-rag and i4ms-chatbot: Custom min-max + alpha blend**

```python
# agentic-rag/app/rag/fusion.py
def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]

def fuse(vector_hits, bm25_hits, alpha):
    pool: dict[str, RetrievedChunk] = {}

    # Normalize vector scores to [0,1] and add to pool
    for hit, norm in zip(vector_hits, minmax([h.vector_score for h in vector_hits])):
        hit.vector_score = norm
        pool[hit.document_id] = hit

    # Normalize BM25 scores to [0,1] and merge into pool
    for hit, norm in zip(bm25_hits, minmax([h.bm25_score for h in bm25_hits])):
        if hit.document_id in pool:
            pool[hit.document_id].bm25_score = norm
        else:
            hit.bm25_score = norm
            pool[hit.document_id] = hit

    # Blend: hybrid = alpha * vector + (1-alpha) * bm25
    for chunk in pool.values():
        chunk.hybrid_score = alpha * chunk.vector_score + (1 - alpha) * chunk.bm25_score

    return sorted(pool.values(), key=lambda c: c.hybrid_score, reverse=True)
```

**enterprise-rag-bot: LangChain EnsembleRetriever (true RRF)**

```python
# enterprise-rag-bot/rag/retriever.py
from langchain.retrievers import EnsembleRetriever

ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5],  # equal weight; EnsembleRetriever uses RRF internally
)
```

`EnsembleRetriever` applies Reciprocal Rank Fusion: each document's rank from each retriever contributes `1 / (k + rank)` where k=60. The scores from all retrievers are summed, so a document ranked #1 by both retrievers wins convincingly.

**Follow-up questions:**
- What is the weakness of min-max normalization that RRF avoids?
- What does alpha=0.7 mean vs alpha=0.3 in your fusion?
- What happens to a document that only appears in one retriever's results?

---

## 11. Min-max normalization vs RRF

**Q: You mentioned min-max and RRF. What's the difference and which is better?**

**Min-max normalization:**
- Scales raw scores to [0,1] range within each retriever's result set
- Preserves relative score gaps (a document scored 0.9 vs 0.5 looks very different)
- **Problem:** The absolute scores from different retrievers are still on different scales. A BM25 score of 0.8 may not represent the same "relevance level" as a cosine score of 0.8

**Reciprocal Rank Fusion (RRF):**
```
RRF(document d) = Σ  1 / (k + rank_i(d))
                  i
```
Where k=60 (conventional constant that dampens the influence of top ranks).

- Uses only **rank position**, not raw score values
- Robust to scale differences between retrievers
- A document ranked #1 by both retrievers scores `1/61 + 1/61 ≈ 0.0328`
- A document ranked #10 by one retriever scores `1/70 ≈ 0.0143`

**Which is better:** RRF is generally preferred because it doesn't require the raw scores to be on the same scale. Min-max can work well when you've tuned the alpha parameter carefully.

**Follow-up questions:**
- Why is k=60 the conventional constant in RRF?
- Can RRF combine more than two retrievers? How?
- How would you tune alpha in the min-max approach without labeled data?

---

## 12. Cross-encoder reranking

**Q: What is cross-encoder reranking and why add it on top of fusion?**

Bi-encoders (used in vector search) encode the query and document **separately** and compare vectors. This is fast but imprecise — the model never sees the query and document together.

Cross-encoders encode the **(query, document) pair jointly**, giving the model full attention across both texts. This produces much more accurate relevance scores but is too slow to run on the full corpus.

**Solution:** Use bi-encoder for fast candidate retrieval, then cross-encoder to rerank the small set of candidates.

```python
# agentic-rag/app/rag/retriever.py
from sentence_transformers import CrossEncoder

_RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def _rerank(query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
    if _RERANKER is None or not chunks:
        return chunks[:top_n]

    # Cross-encoder sees (query, document) pairs together
    pairs = [(query, c.content) for c in chunks]
    scores = _RERANKER.predict(pairs)  # one relevance score per pair

    for chunk, score in zip(chunks, scores):
        chunk.rerank_score = float(score)

    ranked = sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)
    return ranked[:top_n]
```

**Pipeline flow:**
```
vector_search(top_k=5) ──┐
                          ├─→ fuse() → top 10 fused → rerank() → top 3 final
bm25_search(top_k=5)   ──┘
```

**Note:** `enterprise-rag-bot` does **not** use a cross-encoder — it relies on EnsembleRetriever + RRF alone for ranking.

**Model used:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — trained on MS MARCO passage ranking dataset, lightweight enough for CPU inference.

**Follow-up questions:**
- What is the latency cost of cross-encoder reranking? How do you manage it?
- When would you skip the cross-encoder for latency reasons?
- What is MS MARCO and why was training on it useful for your use case?

---

## 13. Hallucination prevention

**Q: How do you prevent the LLM from making up information?**

Hallucination prevention is layered:

**Layer 1 — Constrained system prompt:**
```python
# enterprise-rag-bot/rag/prompts.py
RAG_SYSTEM_PROMPT = """\
You are an internal HR policy assistant.

Rules:
1. Answer only from the policy context provided below.
2. If the answer is not present, say:
   "I could not find this information in the policy documents."
3. Keep the answer short and clear.
4. Mention the source file name when citing policy information.

Policy Context:
{context}"""
```

**Layer 2 — Grounded context injection:** The LLM receives the retrieved chunks, not a general question. It can only answer what's in the context window.

**Layer 3 — Source citation:** The agent format `[doc:<id>]` forces the model to reference specific chunks, making it harder to hallucinate unsourced claims.

**Layer 4 — Faithfulness evaluation (post-generation):**
```python
# agentic-rag/app/evaluation/metrics.py
def faithfulness(sample: EvalSample) -> EvalResult:
    context = "\n".join(sample.contexts)
    prompt = (
        "Evaluate FAITHFULNESS: whether every claim in the ANSWER is "
        "supported by the CONTEXT. Return JSON {\"score\": 0..1, \"reason\": str}.\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{sample.answer}"
    )
    score, reason = _ask_score(prompt)
    return EvalResult(metric="faithfulness", score=score, detail={"reason": reason})
```

**Follow-up questions:**
- What is the difference between faithfulness and answer relevancy as metrics?
- Can a model hallucinate even when it has the correct context? In what scenario?
- How would you handle a case where multiple retrieved chunks contradict each other?

---

## 14. Evaluation — RAGAS metrics

**Q: How do you measure RAG quality in production?**

The projects implement RAGAS-style LLM-as-judge evaluation from first principles across four metrics:

```python
# agentic-rag/app/evaluation/metrics.py

# 1. FAITHFULNESS — Is every claim grounded in the retrieved context?
def faithfulness(sample: EvalSample) -> EvalResult:
    # LLM judge reads context + answer, scores 0.0 (hallucinated) to 1.0 (grounded)
    ...

# 2. ANSWER RELEVANCY — Does the answer address the user's question?
def answer_relevancy(sample: EvalSample) -> EvalResult:
    # LLM judge reads question + answer, scores 0.0 (off-topic) to 1.0 (on-topic)
    ...

# 3. CONTEXT PRECISION — Are the retrieved chunks relevant to the question?
def context_precision(sample: EvalSample) -> EvalResult:
    # LLM judge reads question + chunks, scores 0.0 (all irrelevant) to 1.0 (all relevant)
    ...

# 4. CONTEXT RECALL — Does the context contain the full answer?
def context_recall(sample: EvalSample) -> EvalResult:
    # Requires a ground truth answer. Scores 0.0 (missing info) to 1.0 (full coverage)
    ...

ALL_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

def evaluate_sample(sample: EvalSample) -> list[EvalResult]:
    return [metric(sample) for metric in ALL_METRICS]
```

**EvalSample structure:**
```python
@dataclass
class EvalSample:
    question: str
    answer: str        # model output
    contexts: list[str]  # retrieved chunks
    ground_truth: str | None = None  # for context_recall only
```

**Langfuse integration:** Evaluation scores are sent to Langfuse as trace scores, enabling trend charts and alerting when quality dips.

**Follow-up questions:**
- Why use an LLM-as-judge instead of BLEU/ROUGE scores for RAG evaluation?
- What is the circular dependency problem with using the same LLM as judge and generator?
- How would you build a ground-truth evaluation dataset for a novel domain?

---

## 15. Scaling to 1M+ documents

**Q: Your current system works for small corpora. What would need to change to handle 1 million documents?**

**Current gap — no ANN index:** All three projects use exact (brute-force) search. At 1M chunks, exact cosine search would take several seconds per query.

**Fix 1 — Add HNSW index:**
```sql
-- Run once after initial bulk ingestion
CREATE INDEX ON langchain_pg_embedding
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Tune ef_search at query time for recall vs latency tradeoff
SET hnsw.ef_search = 100;
```

**Fix 2 — Replace in-memory BM25 with Elasticsearch:**
```python
# In-memory BM25 can't handle 1M documents — too much RAM and too slow to rebuild
# Replace with:
from elasticsearch import Elasticsearch
es = Elasticsearch(hosts=["http://localhost:9200"])
# Use BM25F ranking built into Elasticsearch
```

**Fix 3 — Batch embedding with async ingestion:**
The current pipeline embeds synchronously. At scale, use Airflow (already implemented in `enterprise-rag-bot`) with batched, parallelized embedding jobs.

**Fix 4 — Metadata filtering before vector search:**
```python
# Filter before ANN search to reduce the search space
results = vectorstore.similarity_search_with_score(
    query,
    k=top_k,
    filter={"department": "HR", "year": 2024},
)
```

**Follow-up questions:**
- What is HNSW and how does it trade off accuracy vs speed?
- What are the IVFFlat parameters `lists` and `probes`?
- How does Elasticsearch's BM25 differ from the rank-bm25 Python library?

---

## 16. Document updates without full rebuild

**Q: If a policy document changes, how do you update the system without re-ingesting everything?**

**Current state (all three projects):** Re-running ingestion adds new chunks without checking if those chunks already exist. `uuid4()` is used as the chunk ID, so the same text ingested twice gets two different UUIDs — duplicates accumulate.

```python
# agentic-rag/app/rag/chunking.py (current)
chunk_id = str(uuid.uuid4())  # random UUID every time → no deduplication
```

**What you'd build for production:**

**Option A — Content hash as chunk ID:**
```python
import hashlib

def chunk_id_from_content(text: str, source: str, chunk_index: int) -> str:
    # Same content always produces same ID → enables upsert deduplication
    content_hash = hashlib.sha256(
        f"{source}:{chunk_index}:{text}".encode()
    ).hexdigest()[:16]
    return f"chunk_{content_hash}"
```

**Option B — Document-level versioning:**
```python
# Track document versions in a separate table
INSERT INTO document_versions (doc_id, file_hash, ingested_at)
VALUES (:id, :hash, now())
ON CONFLICT (doc_id) DO UPDATE
SET file_hash = :hash, ingested_at = now();

-- On re-ingest: delete old chunks for this doc_id, then insert new ones
DELETE FROM langchain_pg_embedding
WHERE cmetadata->>'source' = :source_file;
```

**BM25 refresh:** After any update, call `refresh_retriever()` to rebuild the in-memory BM25 index with the new document set.

**Follow-up questions:**
- Why is using uuid4 for chunk IDs a problem in production?
- What is an upsert and how would you implement one for PGVector?
- How do you handle partial document updates (only a few pages changed)?

---

## 17. Full document lifecycle walkthrough

**Q: Walk me through what happens from the moment a user uploads a PDF to the moment they get an answer.**

```
INGESTION PHASE
═══════════════
PDF upload
    │
    ▼
Document loader (PyPDFLoader / UnstructuredFileLoader)
    │  extracts raw text
    ▼
RecursiveCharacterTextSplitter (chunk_size=800, overlap=150)
    │  produces list[Document] — each with page_content + metadata
    ▼
Embedding model (all-MiniLM-L6-v2 or text-embedding-3-small)
    │  converts each chunk to a dense vector
    ▼
PGVector.add_documents()
    │  stores (vector, text, metadata) in langchain_pg_embedding table
    ▼
BM25 index rebuild (refresh_retriever())
    │  loads all chunks from DB → tokenizes → builds in-memory index
    └──── INGESTION COMPLETE ────

QUERY PHASE
═══════════
User types: "What is the PTO policy for remote employees?"
    │
    ▼
Embed query → query vector (same model as ingestion!)
    │
    ├──[Dense]──▶ PGVector.similarity_search(k=5) → top 5 chunks by cosine
    │
    └──[Sparse]─▶ BM25Retriever.invoke(query, k=5) → top 5 chunks by BM25
                        │
                        ▼
              Score fusion (min-max + alpha blend  OR  RRF via EnsembleRetriever)
                        │
                        ▼
              Cross-encoder reranking (if available)
                → rerank top candidates → final top 3
                        │
                        ▼
              Format context:
              "Source: leave_policy.docx, Chunk: 3\n<chunk text>\n\n..."
                        │
                        ▼
              Build prompt:
              system (rules) + chat_history + context + user question
                        │
                        ▼
              LLM generation (GPT-4o / GPT-4o-mini)
                        │
                        ▼
              Cache answer in Redis (key: SHA-256(session:query), TTL=1h)
                        │
                        ▼
              Trace to Langfuse (retrieval latency, token usage, scores)
                        │
                        ▼
              Return: { answer, sources, contexts, cache_hit, trace_id }
```

**Follow-up questions:**
- At which step is the bottleneck in a typical RAG pipeline?
- How would Langfuse help you diagnose a sudden drop in answer quality?
- What would change in this pipeline if you switched to a streaming response?
