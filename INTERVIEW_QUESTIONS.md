# i4MS Chatbot — Interview Questions & Concepts

## Part A: Core Concepts by Technology

---

### 1. FastAPI

**What is FastAPI?**
FastAPI is a modern, high-performance Python web framework for building APIs. It is built on top of Starlette (for the web layer) and Pydantic (for data validation). It supports async/await natively, generates OpenAPI documentation automatically, and uses Python type hints for request/response validation at runtime.

**Key concepts used in this project:**

- **Path operations:** `@app.get("/health")`, `@app.post("/chat")` — decorators that map HTTP methods and URL paths to Python functions.
- **Pydantic models for request/response:** `ChatRequest` and `ChatResponse` are Pydantic `BaseModel` subclasses. FastAPI automatically validates incoming JSON against the model, returns 422 on invalid input, and serializes the response.
- **Dependency injection via Headers:** `x_role: str | None = Header(default=None)` extracts values from HTTP headers and injects them into the endpoint function.
- **Lifespan events:** The `@asynccontextmanager` lifespan function runs startup logic (configure logging) before the app serves requests and shutdown logic (flush Langfuse) after it stops.
- **StreamingResponse:** Used for the `/chat/stream` endpoint to send Server-Sent Events (SSE). The response is not buffered — chunks are sent to the client as they are generated.
- **HTTPException:** Raised to return error HTTP status codes (400 for blocked input, 403 for auth errors, 500 for internal failures) with structured error messages.

---

### 2. LangChain

**What is LangChain?**
LangChain is a framework for building applications powered by language models. It provides abstractions for LLM calls, prompt templates, document loaders, text splitters, embeddings, vector stores, retrievers, and tool-calling agents. It standardises how you compose these components into chains and pipelines.

**Key concepts used in this project:**

- **ChatOpenAI:** A wrapper around the OpenAI chat completions API. Configured with model name, temperature, and API key. Supports `.invoke()` for single calls and `.bind_tools()` to register tool schemas.
- **OpenAIEmbeddings:** Generates vector embeddings from text using OpenAI's embedding models (e.g., `text-embedding-3-small`). Used to convert policy document chunks into vectors for similarity search.
- **RecursiveCharacterTextSplitter:** Splits long documents into overlapping chunks using a hierarchy of separators (`\n\n`, `\n`, `. `, ` `). Chunk size (800) and overlap (120) control granularity and context preservation.
- **PGVector (langchain-postgres):** A LangChain vector store backed by PostgreSQL with the pgvector extension. Stores embeddings alongside metadata in a relational database, enabling filtered similarity search.
- **Tools (@tool decorator):** Functions decorated with `@tool` become callable by the LLM. The decorator extracts the function signature and docstring into a JSON schema that the model uses to decide when and how to call the tool.
- **Callback handlers:** LangChain's callback system lets you hook into every step of a chain (LLM call, tool invocation, retrieval). Used here to stream events to Langfuse for observability.

---

### 3. LangGraph

**What is LangGraph?**
LangGraph is a library for building stateful, multi-step agent workflows as directed graphs. Unlike simple LangChain chains (linear pipelines), LangGraph supports cycles, conditional branching, and persistent state — which is essential for ReAct-style agents that loop between reasoning and tool-calling.

**Key concepts used in this project:**

- **StateGraph:** The core abstraction. You define a typed state (here, `AgentState` with a `messages` list), add nodes (functions that transform state), and add edges (transitions between nodes).
- **Nodes:** `"agent"` (calls the LLM) and `"tools"` (executes tool calls). Each node receives the current state and returns a partial state update.
- **Conditional edges:** `_should_continue` inspects the last message. If the LLM made tool calls, route to the `"tools"` node; otherwise, route to `END`. This creates the ReAct loop.
- **add_messages reducer:** The `Annotated[list[AnyMessage], add_messages]` type tells LangGraph to *append* new messages to the list rather than replacing it. This accumulates the full conversation.
- **MemorySaver checkpointer:** Persists the graph state (all messages) keyed by `thread_id`. When the same `session_id` is reused, the previous conversation is restored — enabling multi-turn chat.
- **ToolNode (prebuilt):** A ready-made node that executes tool calls found in the last LLM message and returns `ToolMessage` results.
- **astream_events:** An async method that yields fine-grained events as the graph executes — used for the streaming endpoint.

---

### 4. OpenAI GPT-4o-mini

**What is GPT-4o-mini?**
GPT-4o-mini is a smaller, faster, and cheaper variant of OpenAI's GPT-4o model. It supports tool calling (function calling), structured outputs, and has a 128K context window. It balances quality and cost for production applications.

**How it is used in this project:**

- **Agent reasoning:** The main LLM that decides whether to call a tool or answer directly.
- **Text-to-SQL:** Given the database schema and a natural-language question, generates a SQLite SELECT query.
- **Evaluation judge:** Scores agent outputs on faithfulness, relevancy, precision, and recall.
- **Translation:** Translates Odia/Hindi input to English and English output back to the user's language.
- **Tool calling:** The model receives JSON schemas of available tools and returns structured `tool_calls` when it needs external data. This is native to the OpenAI API, not prompt-engineered.

---

### 5. pgvector (PostgreSQL + Vector Extension)

**What is pgvector?**
pgvector is a PostgreSQL extension that adds support for vector data types and similarity search operations (cosine distance, L2 distance, inner product). It lets you store embeddings alongside relational data in the same database, rather than needing a separate vector database.

**Key concepts used in this project:**

- **Vector column:** Each document chunk is stored with its embedding vector. pgvector indexes these for efficient nearest-neighbor search.
- **Similarity search with scores:** `similarity_search_with_relevance_scores(query, k=5)` finds the top-K most similar document chunks to the query embedding.
- **Collection:** Documents are grouped by `collection_name` (here, `i4ms_policy`), allowing multiple independent corpora in the same database.
- **JSONB metadata:** Each vector row stores metadata (source file, chunk index) as JSONB, enabling filtered retrieval.

---

### 6. SQLite

**What is SQLite?**
SQLite is a self-contained, serverless, zero-configuration relational database engine. It stores the entire database in a single file. In this project, it stands in for the production i4MS RDBMS so the project runs locally without needing the government database.

**Key concepts used in this project:**

- **PRAGMA query_only:** A SQLite pragma that makes the connection read-only at the database engine level — defense-in-depth against accidental writes.
- **Row factory:** `conn.row_factory = sqlite3.Row` makes query results accessible by column name (like a dict) instead of positional index.
- **Parameterized queries:** `conn.execute(sql, params)` uses `?` placeholders to prevent SQL injection. The tenant-scoping predicates are always parameterized.
- **Schema design:** 6 normalized tables with foreign keys modelling the i4MS domain — lessees, leases, permits, passes, payments, returns.

---

### 7. Docker & Docker Compose

**What is Docker?**
Docker packages an application and its dependencies into a container — a lightweight, isolated, reproducible environment. Docker Compose orchestrates multi-container applications (here: the API + PostgreSQL database).

**Key concepts used in this project:**

- **Multi-stage build:** The Dockerfile installs system dependencies (libpq for PostgreSQL), then Python packages, then copies the app code.
- **Service dependencies:** `depends_on: db: condition: service_healthy` ensures the API only starts after PostgreSQL passes its health check.
- **Named volumes:** `pgdata` persists the database across container restarts.
- **Environment variables:** `env_file: .env` injects configuration; `PG_HOST: db` overrides the host to use Docker's internal DNS.

---

### 8. Pydantic & Pydantic Settings

**What is Pydantic?**
Pydantic is a data validation library that uses Python type annotations to define data schemas. `BaseModel` subclasses validate, serialize, and document data structures. Pydantic Settings extends this to load configuration from environment variables and `.env` files.

**Key concepts used in this project:**

- **Request/response validation:** `ChatRequest` validates that `query` is a non-empty string; `ChatResponse` ensures the response has the correct shape.
- **Field constraints:** `Field(..., min_length=1)` rejects empty queries at the framework level.
- **Settings with aliases:** `Field(default="gpt-4o-mini", alias="LLM_MODEL")` maps the Python attribute `llm_model` to the environment variable `LLM_MODEL`.
- **lru_cache on get_settings():** The settings object is created once and reused, avoiding repeated `.env` file reads.

---

### 9. Langfuse

**What is Langfuse?**
Langfuse is an open-source observability platform for LLM applications. It captures traces (end-to-end request flows), spans (individual steps like LLM calls or tool invocations), and scores (quality metrics or user feedback) in a dashboard.

**Key concepts used in this project:**

- **Callback handler:** `CallbackHandler()` is passed to LangGraph via `config["callbacks"]`. It automatically captures every LLM call, tool invocation, and retrieval step as spans within a trace.
- **Custom spans:** `trace_span("hybrid_retrieve", ...)` creates manual spans for application-specific logic (retrieval fusion, text-to-SQL).
- **Score attachment:** `score_trace(trace_id, "faithfulness", 0.85)` attaches evaluation metrics to the trace. User feedback (`/feedback` endpoint) also attaches scores — closing the feedback loop.
- **Graceful degradation:** If Langfuse keys are missing or the package is not installed, all tracing becomes a no-op. The app never crashes due to observability failures.

---

### 10. BM25 (Best Matching 25)

**What is BM25?**
BM25 is a probabilistic ranking function for information retrieval. It scores documents based on term frequency (TF), inverse document frequency (IDF), and document length normalization. Unlike vector embeddings which capture semantic meaning, BM25 excels at exact keyword matching.

**Key concepts used in this project:**

- **BM25Okapi:** The specific BM25 variant used. It is the standard in search engines (Elasticsearch, Lucene).
- **Tokenization:** Text is split into lowercase alphanumeric tokens using regex (`\w+`).
- **In-memory index:** The BM25 index is built at ingestion time and lives in memory. In production, this would be backed by PostgreSQL full-text search (tsvector).
- **Hybrid retrieval:** BM25 scores are fused with vector scores to combine exact keyword matching with semantic understanding.

---

### 11. Cross-Encoder Reranker

**What is a cross-encoder?**
A cross-encoder is a transformer model that takes a (query, document) pair as input and outputs a relevance score. Unlike bi-encoders (which encode query and document separately), cross-encoders attend to both simultaneously — producing higher quality scores at the cost of speed.

**Key concepts used in this project:**

- **ms-marco-MiniLM-L-6-v2:** A lightweight cross-encoder trained on the MS MARCO passage ranking dataset. Balances accuracy and speed.
- **Reranking pipeline:** The hybrid retriever first retrieves a candidate set (fast), then the cross-encoder reranks the candidates (accurate). This two-stage approach gets the best of both worlds.
- **Score assignment:** Each candidate chunk gets a `rerank_score` and the top-N are returned.

---

### 12. Sentence Transformers

**What is Sentence Transformers?**
A Python library for computing dense vector embeddings of text using transformer models. It provides pre-trained models for semantic similarity, paraphrase detection, and cross-encoding. Built on top of Hugging Face Transformers and PyTorch.

**Used in this project for:** Loading and running the cross-encoder reranker model.

---

### 13. RAGAS (RAG Assessment)

**What is RAGAS?**
RAGAS is a framework for evaluating Retrieval-Augmented Generation pipelines. It defines metrics that assess both the retrieval quality and the generation quality without requiring human labels (using LLM-as-judge).

**Metrics implemented in this project:**

- **Faithfulness:** Is every claim in the answer supported by the retrieved context? (Tests for hallucination)
- **Answer Relevancy:** Does the answer directly address the question? (Tests for off-topic responses)
- **Context Precision:** What fraction of retrieved chunks are actually relevant? (Tests retrieval quality)
- **Context Recall:** Does the retrieved context contain the information needed to answer correctly? (Tests retrieval completeness)

---

## Part B: Scenario-Based Interview Questions

---

### Q0: Walk me through the project

**Answer:**

Sure. This is the **i4MS Chatbot** — an AI-powered assistant built for the **Integrated Minor Mineral Mining Management System** of the Government of Odisha. Let me walk through it layer by layer, starting from what the user sees and going down to the internals.

---

**What it does:**

The chatbot answers two kinds of questions for people who use the i4MS portal — mining officers and lease holders (lessees):

1. **Data questions** — "How many active leases are in Cuttack district?", "Show me my pending royalty payments" — these query actual records in the database.
2. **Policy questions** — "How long before a lease expires must I apply for renewal?", "What is the royalty rate for sand?" — these look up government rules and procedures from a knowledge base of 7 policy documents.

The user can also ask in **Odia** (the regional language), and the system will auto-detect it, process internally in English, and respond back in Odia.

---

**The request flow (what happens when a user sends a message):**

```
User sends query
       │
       ▼
 ┌─ Prompt Injection Guard ──┐
 │  16 regex patterns check   │
 │  for jailbreak, role       │
 │  hijack, instruction       │
 │  override attempts.        │
 │  Blocked → HTTP 400        │
 └────────────┬───────────────┘
              ▼
 ┌─ RBAC Resolution ─────────┐
 │  HTTP headers → Role       │
 │  (Lessee / Officer / Admin)│
 │  + lessee_id or district   │
 └────────────┬───────────────┘
              ▼
 ┌─ Language Detection ───────┐
 │  Odia? Hindi? English?     │
 │  If non-English: translate  │
 │  to English first           │
 └────────────┬───────────────┘
              ▼
 ┌─ LangGraph ReAct Agent ───┐
 │  LLM reads the question    │
 │  and decides which tool    │
 │  to call (or both):       │
 │                            │
 │  Tool 1: query_i4ms_db    │
 │    → Text-to-SQL           │
 │    → SQL Guard validates   │
 │    → Scope injection       │
 │    → Execute on SQLite     │
 │    → PII redaction         │
 │                            │
 │  Tool 2: search_policy     │
 │    → Hybrid RAG            │
 │    → pgvector (dense)      │
 │    → BM25 (sparse)         │
 │    → Score fusion          │
 │    → Cross-encoder rerank  │
 │                            │
 │  LLM reads tool output,   │
 │  generates final answer    │
 └────────────┬───────────────┘
              ▼
 ┌─ Translation (if needed) ──┐
 │  English → Odia/Hindi      │
 └────────────┬───────────────┘
              ▼
 ┌─ Response ─────────────────┐
 │  answer + citations +      │
 │  tools_used + trace_id +   │
 │  latency_ms                │
 └────────────────────────────┘
```

---

**The tech stack and why each piece is there:**

- **FastAPI** — The web framework. It serves the REST API (`/chat`, `/chat/stream`, `/feedback`, `/health`). It validates requests with Pydantic, supports async streaming, and auto-generates API docs.

- **LangGraph** — The agent orchestrator. It runs the ReAct loop (reason → act → observe → repeat) as a state machine graph. It also provides `MemorySaver` for multi-turn conversation memory — so follow-up questions like "What about Sambalpur?" work within a session.

- **LangChain** — Provides the building blocks: `ChatOpenAI` for LLM calls, `OpenAIEmbeddings` for vector embeddings, `PGVector` for the vector store, `RecursiveCharacterTextSplitter` for chunking documents, and the `@tool` decorator for registering functions as agent tools.

- **OpenAI GPT-4o-mini** — The language model that powers everything: agent reasoning, SQL generation, evaluation judging, and Odia translation. Chosen for its balance of quality, speed, and cost.

- **pgvector (PostgreSQL)** — Stores document embeddings for dense semantic search. Policy chunks are embedded with `text-embedding-3-small` and stored as vectors.

- **BM25 (rank-bm25)** — In-memory sparse keyword index for exact term matching. Combined with pgvector via score fusion for hybrid retrieval.

- **Cross-encoder reranker (sentence-transformers)** — A `ms-marco-MiniLM-L-6-v2` model that reranks the top retrieval candidates for final precision. It sees (query, document) together, so it is more accurate than bi-encoder similarity alone.

- **SQLite** — A local stand-in for the production i4MS database. Contains 6 tables (lessees, leases, permits, passes, royalty payments, statutory returns) with seed data.

- **Langfuse** — LLM observability. Every request produces a trace with spans for each step (LLM call, tool invocation, retrieval). Evaluation scores and user feedback are attached to traces.

- **Docker + Docker Compose** — Containerized deployment with pgvector and the API as separate services.

---

**Security — this is a government system, so it is designed for auditability:**

1. **SQL Guard** — The LLM generates SQL, but it passes through a validator that blocks 30+ forbidden keywords (INSERT, DELETE, DROP, etc.), rejects statement stacking, and injects a LIMIT. Only SELECT/WITH is allowed.

2. **Tenant scoping** — The LLM's SQL is wrapped as a subquery, and a parameterized WHERE clause is applied externally based on the caller's role. A lessee can only see their own records. An officer can see their district. The LLM never controls access boundaries.

3. **PII redaction** — PAN and mobile numbers are masked for non-admin roles at the database layer, so they never reach the LLM or the response.

4. **Prompt injection guard** — 16 regex patterns detect instruction overrides, role hijacking, jailbreak attempts, and prompt leaking. Blocked requests get HTTP 400.

5. **Read-only enforcement** — SQLite `PRAGMA query_only = ON`. In production, this would be a read-replica with SELECT-only DB privileges.

---

**Evaluation:**

The project includes an offline evaluation harness with 4 RAGAS-style metrics (faithfulness, answer relevancy, context precision, context recall), scored by an LLM judge. There is also a `/feedback` endpoint for real-world user feedback that attaches scores to Langfuse traces — closing the evaluation loop.

---

**What makes this project stand out from a typical chatbot:**

- It is not just a wrapper around an LLM. It has **agentic tool routing** (the model decides which tool to call), **hybrid retrieval** (dense + sparse + reranker), and **multi-turn memory**.
- Security is treated seriously — SQL injection prevention with four layers of defense-in-depth, tenant scoping in code (not in prompts), PII redaction, and prompt injection detection.
- It supports a **regional language** (Odia) without needing multilingual embeddings.
- It has **observability** (Langfuse) and **evaluation** (RAGAS metrics) built in — not bolted on.
- It is containerized and production-shaped, not just a notebook.

---

### Q1: What is the knowledge base from where you have gotten the data?

**Answer:** The chatbot has two separate knowledge sources:

1. **Policy knowledge base (RAG):** 7 Markdown documents stored in the `data/policies/` directory, covering the Odisha Minor Mineral Concession (OMMC) Rules 2016, e-Permit and e-Transit Pass procedures (OMPTS Rules 2007), lease and royalty provisions, registration SOPs, royalty rate schedules, enforcement and penalties, and a FAQ document. These are chunked, embedded using OpenAI's `text-embedding-3-small` model, and stored in pgvector for dense retrieval. A parallel BM25 index is built for sparse keyword-based retrieval. Both are fused and reranked for final results.

2. **Operational database (Text-to-SQL):** A SQLite database (`data/i4ms.db`) seeded with sample data modelling the i4MS domain — 4 lessees, 5 leases/sairat sources, 4 e-permits, 6 e-transit passes, 3 royalty payments, and 5 statutory returns. This stands in for the production i4MS RDBMS. In production, this would point at a read-replica of the actual government database.

The agent decides which source to query based on the user's question: data questions go to the database, policy questions go to the RAG pipeline, and mixed questions can use both.

---

### Q2: How does the chatbot decide whether to query the database or search policies?

**Answer:** The agent uses OpenAI's **tool-calling** (function calling) capability. Two tools are registered:

- `query_i4ms_database` — docstring says it handles counts, lookups, statuses about actual records
- `search_minor_mineral_policy` — docstring says it handles rules, procedures, and policy questions

The system prompt explicitly instructs the model: *"Data -> database. Rules -> policy search."* The LLM reads the user's question, matches it to the appropriate tool based on the tool descriptions, and emits a structured `tool_call`. The LangGraph routing then executes the selected tool. If the question requires both (e.g., "is my pass still valid?" needs the pass record AND the validity rule), the model can call both tools in sequence within the same ReAct loop.

---

### Q3: How do you prevent SQL injection when an LLM generates SQL?

**Answer:** We use a **four-layer defense-in-depth** approach:

1. **SQL Guard (`sql_guard.py`):** Validates that the generated SQL is a single SELECT or WITH statement. Blocks 30+ forbidden keywords (INSERT, UPDATE, DELETE, DROP, ALTER, PRAGMA, etc.) using token-level matching (not substring, so `updated_at` as a column name is allowed while `update` is blocked). Rejects statement stacking (`;` chaining). Injects a LIMIT clause if missing.

2. **External scope injection (`text_to_sql.py`):** The validated SQL is wrapped as a subquery, and a parameterized WHERE clause is applied externally based on the caller's role. The LLM never controls who sees what — even if it generates `SELECT * FROM lessee`, the outer query restricts results to the caller's scope.

3. **Parameterized queries:** Tenant predicates use `?` placeholders, not string interpolation. This prevents second-order injection through user-controlled values.

4. **SQLite PRAGMA query_only (`connection.py`):** The database connection itself is opened in read-only mode. Even if all other guards fail, the database engine rejects any write operation.

In production, this would additionally run against a **read-only database replica** with a DB role that only has SELECT privileges — so even a complete code bypass cannot write data.

---

### Q4: How does Role-Based Access Control (RBAC) work in this system?

**Answer:** Three roles exist:

- **LESSEE:** Sees only their own records. Scoping predicate: `WHERE lessee_id = ?` with their authenticated `lessee_id`.
- **OFFICER:** Sees all records within their assigned district. Predicate: `WHERE district = ?`.
- **ADMIN:** Sees everything. Predicate: `WHERE 1=1` (no restriction, but audited).

The role is resolved from HTTP headers (standing in for JWT claims in production) in the API layer. It is then bound to a `ContextVar` for the duration of the request. The key design principle is that **the LLM never controls access boundaries** — the model generates SQL to answer the question, but tenant scoping is injected by deterministic code around the model's output. Even if the model ignores instructions about scoping, the outer query enforces it.

Additionally, PII columns (PAN and mobile numbers) are redacted for non-admin roles at the database connection layer, so sensitive data never reaches the LLM or the response.

---

### Q5: What is hybrid retrieval and why did you use it instead of plain vector search?

**Answer:** Hybrid retrieval combines two complementary search methods:

- **Dense retrieval (pgvector):** Converts the query and documents into embedding vectors and finds the closest vectors by cosine similarity. Great for semantic matching — "lease expiry" matches "concession validity period" even without shared words.
- **Sparse retrieval (BM25):** Scores documents based on exact keyword overlap, term frequency, and inverse document frequency. Great for specific terms — "OMMC Rule 42" or "e-transit pass TP0003" where exact words matter.

Neither alone is perfect. Vector search can miss exact keyword matches; BM25 misses semantic equivalences. **Fusion** combines them:

1. Both retrievers return their top-K results independently
2. Scores are min-max normalized to the 0-1 range
3. A weighted blend (`hybrid_alpha * vector_score + (1 - alpha) * bm25_score`) produces a unified ranking
4. A **cross-encoder reranker** then reorders the top candidates for final precision

This three-stage pipeline (retrieve → fuse → rerank) is the gold-standard approach in modern RAG systems.

---

### Q6: How does the prompt injection guard work?

**Answer:** The guard (`input_guard.py`) runs on every incoming query **before** it reaches the agent. It uses a multi-layer approach:

1. **16 regex patterns** detect known attack signatures:
   - Instruction override: "ignore all previous instructions", "disregard your rules"
   - Role hijacking: "you are now a hacker", "pretend you are"
   - Fake system messages: "[SYSTEM]", "<<SYS>>", "system:"
   - Prompt leaking: "reveal your system prompt", "show your instructions"
   - Jailbreak keywords: "DAN mode", "bypass restrictions"
   - Code injection: "eval()", "import os", "subprocess"
   - Encoding tricks: hex escape sequences

2. **Structural heuristics** flag suspicious formatting:
   - Excessive input length (>2000 chars)
   - Multiple separator lines (--------, ========) suggesting delimiter injection
   - Unusual newline density suggesting prompt template injection

3. Each matched pattern has a **risk score** (0.0–1.0). The maximum score is compared against a configurable threshold (default 0.6). If exceeded, the request is blocked with HTTP 400 and logged for audit.

This is intentionally rule-based (not LLM-based) so it cannot itself be fooled by the same attack.

---

### Q7: How does multi-turn conversation memory work?

**Answer:** LangGraph's `MemorySaver` checkpointer stores the complete message history keyed by `thread_id` (mapped from the request's `session_id`). When a user sends a follow-up message in the same session:

1. The graph loads the previous state (all prior messages) from the checkpointer
2. The new `HumanMessage` is appended via the `add_messages` reducer
3. The LLM sees the full conversation context and can reference previous exchanges
4. After processing, the updated state is saved back to the checkpointer

This enables conversations like:
- User: "How many active leases are in Cuttack?" → Agent queries DB, responds with count
- User: "What about Sambalpur?" → Agent understands the context from history, queries DB for Sambalpur

Without memory, each request would start fresh and "What about Sambalpur?" would have no context.

---

### Q8: How does the Odia language support work without multilingual embeddings?

**Answer:** Instead of replacing the entire RAG pipeline with multilingual models, we use a **translate-then-process** approach:

1. **Detection:** Unicode script analysis checks if the input contains Odia characters (U+0B00–U+0B7F) or Devanagari characters (Hindi). This is a deterministic check, not an LLM call.
2. **Translate to English:** If non-English, the query is translated to English using the same GPT-4o-mini model already deployed.
3. **Process normally:** The agent, tools, RAG pipeline, and database all work in English as usual.
4. **Translate back:** The English response is translated to the detected language before returning to the user.

This avoids the need for multilingual embeddings, multilingual BM25 tokenization, or Odia-language policy documents. The tradeoff is two extra LLM calls (translate in + translate out), but these are fast and cheap with GPT-4o-mini.

---

### Q9: How does the streaming endpoint differ from the regular chat endpoint?

**Answer:** The regular `POST /chat` endpoint blocks until the entire agent loop completes (LLM reasoning, tool calls, final answer generation), then returns the complete `ChatResponse` as JSON. The user sees nothing until everything is done.

The streaming `POST /chat/stream` endpoint returns a `text/event-stream` response immediately and sends **Server-Sent Events (SSE)** as the agent processes:

- `event: thinking` — signals the agent has started
- `event: tool_call` — shows which tool is being called and its arguments (real-time visibility into agent reasoning)
- `event: tool_result` — confirms the tool finished
- `event: token` — each chunk of the final answer text as it is generated (token-by-token)
- `event: done` — the complete response JSON (for clients that want the structured object)
- `event: error` — if something fails mid-stream

The implementation uses FastAPI's `StreamingResponse` with LangGraph's `astream_events()` async generator. The client can render tokens incrementally for a chat-like typing effect.

---

### Q10: What are the four RAGAS evaluation metrics and why do they matter?

**Answer:**

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| **Faithfulness** | Is every claim in the answer supported by the retrieved context? | Detects **hallucination** — the model inventing facts not present in the source documents |
| **Answer Relevancy** | Does the answer directly address the question asked? | Detects **off-topic** responses where the model answers a different question |
| **Context Precision** | What fraction of retrieved chunks are actually relevant? | Measures **retrieval quality** — are we fetching useful context or noise? |
| **Context Recall** | Does the retrieved context cover all the information in the ground-truth answer? | Measures **retrieval completeness** — are we missing relevant documents? |

Together, these four metrics evaluate the complete RAG pipeline end-to-end: retrieval quality (precision + recall) and generation quality (faithfulness + relevancy). Each metric is scored 0.0–1.0 by an LLM judge (GPT-4o-mini) — no human labels required for automated evaluation.

---

### Q11: What is the difference between a bi-encoder and a cross-encoder, and why does this project use both?

**Answer:**

- **Bi-encoder (OpenAI embeddings):** Encodes the query and each document independently into separate vectors. Similarity is computed by comparing vectors (cosine distance). Very fast — you can pre-compute document embeddings and search millions of vectors in milliseconds using pgvector. But quality is limited because the model never sees query and document together.

- **Cross-encoder (ms-marco-MiniLM):** Takes the query and document as a single concatenated input and outputs a relevance score. Much more accurate because the model attends to both simultaneously (cross-attention). But very slow — you cannot pre-compute anything; every (query, document) pair requires a forward pass.

This project uses both in a **two-stage pipeline:** the bi-encoder retrieves a candidate set quickly (top-K from millions), then the cross-encoder reranks just those K candidates accurately. This is the standard retrieve-then-rerank pattern used in production search systems.

---

### Q12: Why is tenant scoping done in code rather than in the LLM prompt?

**Answer:** The system prompt tells the model to respect access boundaries, but this is **not relied upon for enforcement**. Here is why:

1. **LLMs are probabilistic.** Even with clear instructions, a model might occasionally generate SQL without a WHERE clause or with the wrong scope. In a government compliance system, "usually works" is not acceptable.

2. **Prompt injection attacks** could instruct the model to ignore scoping rules. If scoping were prompt-based, a crafted input could bypass it.

3. **Auditability.** Code-based scoping is deterministic, testable, and reviewable. You can write a unit test that proves `Role.LESSEE` with `lessee_id="LS001"` always produces `WHERE lessee_id = ?` with params `["LS001"]`. You cannot write such a test for a prompt.

The implementation wraps the LLM's SQL as a subquery and applies the predicate on the outer query:
```sql
SELECT * FROM (
  <LLM-generated SQL>
) AS scoped
WHERE scoped.lessee_id = ?
```

This means the model's SQL runs in a sandbox — it can select whatever it wants, but the results are always filtered by the caller's identity. If the inner query does not expose `lessee_id`/`district` columns, the outer predicate fails closed (no rows returned).

---

### Q13: What happens when a lessee asks about data outside their scope?

**Answer:** The system handles this at multiple levels:

1. **System prompt:** Instructs the model to explain that it can only report on the caller's own records.
2. **Scope injection:** Even if the model generates SQL for all lessees, the outer predicate filters to only the caller's `lessee_id`. The result set is already scoped before the model sees it.
3. **Tool output:** If no matching records exist within scope, the tool returns "No matching records found (within your access scope)."
4. **PII redaction:** Even if scoping somehow leaked a different lessee's row, PII columns (PAN, mobile) would be redacted.

The key design: the model cannot escalate privileges. It never controls whose data it sees — that is decided by the authenticated session before the model runs.

---

### Q14: How would you deploy this chatbot in a production government environment?

**Answer:** Several changes from the current local setup:

1. **Database:** Replace SQLite with a read-only connection to the actual i4MS PostgreSQL replica. Use a DB role with SELECT-only privileges.
2. **Authentication:** Replace header-based role resolution with actual JWT verification from the i4MS login system. The session token would carry the role, lessee_id, and district claims.
3. **HTTPS / API Gateway:** Put the FastAPI service behind an API gateway (e.g., Kong, AWS API Gateway) with TLS termination, rate limiting, and request logging.
4. **Horizontal scaling:** Run multiple API instances behind a load balancer. Replace `MemorySaver` (in-memory) with a persistent checkpointer like `PostgresSaver` or `RedisSaver` so conversation memory survives restarts and works across instances.
5. **Vector store:** pgvector is already production-ready; just point it at the production PostgreSQL cluster.
6. **BM25:** Replace the in-memory BM25 index with PostgreSQL full-text search (tsvector/tsquery) for persistence and scalability.
7. **Monitoring:** Keep Langfuse for LLM observability; add application monitoring (Prometheus + Grafana or equivalent).
8. **Audit logging:** Log every query, generated SQL, role, and response to a tamper-proof audit table for compliance.

---

### Q15: What is the ReAct pattern and why is it used here?

**Answer:** ReAct (Reason + Act) is an agent design pattern where the LLM alternates between:

1. **Reasoning:** Thinking about the question and deciding what action to take
2. **Acting:** Calling a tool (database query, policy search)
3. **Observing:** Reading the tool's output
4. **Reasoning again:** Deciding if the answer is complete or if another action is needed

In LangGraph, this is implemented as a cycle: `agent → tools → agent → ... → END`. The `_should_continue` function checks if the LLM's last message contains `tool_calls`. If yes, route to the tools node; if no (the LLM produced a final text answer), route to END.

This is better than a simple chain because:
- The agent can call multiple tools in sequence (e.g., look up a pass, then check the validity rule)
- It can recover from errors (if a tool returns no results, it can try a different approach)
- It can decide not to use any tool if the question is conversational

---

### Q16: How does min-max score fusion work in hybrid retrieval?

**Answer:** The vector retriever and BM25 retriever return scores on different scales (cosine similarity 0–1 vs. BM25 scores 0–20+). You cannot simply add them. Min-max normalization rescales both to 0–1:

```
normalized = (score - min_score) / (max_score - min_score)
```

After normalization, the scores are blended with a weight parameter `hybrid_alpha`:
```
hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score
```

With `alpha = 0.5` (the default), both signals contribute equally. You can tune alpha: higher values favor semantic matching; lower values favor keyword matching. The fused results are sorted by hybrid score, and the top candidates go to the reranker.

---

### Q17: What is the purpose of the feedback endpoint and how does it close the evaluation loop?

**Answer:** The `POST /feedback` endpoint lets end users rate the chatbot's responses. It takes a `trace_id` (returned in every `ChatResponse`), a numeric `score` (e.g., 1–5 stars), and an optional `comment`.

This feedback is attached to the corresponding Langfuse trace via `score_trace()`. Now the trace has both:
- **Automated eval scores** (faithfulness, relevancy) from the evaluation harness
- **Human feedback scores** from real users

This creates a **closed evaluation loop:** you can compare automated metrics against real user satisfaction, identify where the bot fails in practice (low user scores despite high auto-scores = the metrics are wrong; low auto-scores with high user scores = the metrics are too strict), and use the feedback data to improve prompts, retrieval, or the knowledge base.

---

### Q18: Why use ContextVar for the access context instead of passing it as a function argument?

**Answer:** LangChain tools have a fixed interface — they take the arguments that the LLM provides (in this case, just a `question` string). We cannot add an `access_context` parameter because:

1. The LLM would need to populate it, which means the model would choose whose data it can see — a security violation.
2. The tool schema would expose internal access control details to the model.

`ContextVar` solves this by binding the access context to the current execution context (like thread-local storage but async-safe). The API layer sets it per request; the tool reads it internally. The LLM never sees it, cannot modify it, and cannot even know it exists. This is the pattern used in web frameworks for request-scoped state (e.g., Flask's `g`, Django's `get_current_request`).

---

### Q19: How would you measure whether the expanded knowledge base actually improved answer quality?

**Answer:** Run a **before/after evaluation** using the evaluation harness:

1. **Baseline:** Run the evaluation dataset against the original 2-document knowledge base. Record faithfulness, answer relevancy, context precision, and context recall scores.
2. **Expanded:** Re-ingest with all 7 documents. Run the same evaluation dataset. Compare scores.
3. **New questions:** Add questions that specifically target the new documents (e.g., "What is the royalty rate for murrum?", "How do I register a crusher unit?"). These should score near-zero on the baseline (no relevant context) and high on the expanded set.
4. **Regression check:** Ensure the original questions do not get worse (the new documents should not introduce noise that hurts retrieval for existing queries).

Track all runs as Langfuse dataset experiments for reproducibility.

---

### Q20: What are the limitations of this chatbot?

**Answer:**

1. **MemorySaver is in-memory.** Conversations are lost on server restart. Production needs PostgresSaver or RedisSaver.
2. **BM25 index is in-memory.** Must be rebuilt on every server start by re-running ingestion. Production should use PostgreSQL full-text search.
3. **Translation adds latency.** Two extra LLM calls for Odia users. Could be mitigated with a dedicated translation model or caching.
4. **Prompt injection guard is regex-based.** It catches known patterns but cannot detect novel attacks. Could be augmented with a classifier model.
5. **No real authentication.** Roles are passed via headers, not verified JWTs. This is a development convenience, not a production design.
6. **Small seed dataset.** Only 4 lessees and 5 leases. Real-world testing needs thousands of records.
7. **Single-model dependency.** Everything runs on GPT-4o-mini. If the API is down, the entire system is unavailable. Could add fallback models.
8. **No PDF ingestion.** Policy documents must be manually converted to Markdown. Adding PyMuPDF or pdfplumber would automate this.

---

*Prepared for viva/interview preparation based on the i4ms-chatbot project.*
