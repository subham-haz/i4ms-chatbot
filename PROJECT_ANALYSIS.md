# i4MS Chatbot — Project Analysis

## 1. Project Overview

The **i4MS (Integrated Minor Mineral Mining Management System) Chatbot** is an AI-powered assistant built for the Directorate of Minor Minerals, Government of Odisha. It helps officers and lessees interact with the i4MS platform through natural language, answering both **data questions** (about leases, permits, transit passes, royalty payments) and **policy questions** (about OMMC Rules 2016, OMPTS Rules 2007, SOPs).

**Domain:** Government e-Governance / Mining Compliance  
**Type:** Agentic RAG + Text-to-SQL Chatbot  
**Stack:** Python 3.12 | FastAPI | LangGraph | LangChain | OpenAI GPT-4o-mini | pgvector | SQLite | Docker

---

## 2. Architecture

```
                  ┌────────────────────────────────────────┐
                  │        FastAPI API Layer                │
                  │  /chat  /chat/stream  /feedback /health │
                  │  Prompt Injection Guard (input_guard)   │
                  │  RBAC from session headers              │
                  └──────────┬─────────────────────────────┘
                             │
                  ┌──────────▼─────────────────────────────┐
                  │       Odia Language Layer                │
                  │  detect_language → translate_to_english  │
                  │  (process) → translate_from_english      │
                  └──────────┬─────────────────────────────┘
                             │
                  ┌──────────▼─────────────────────────────┐
                  │   LangGraph ReAct Agent + MemorySaver   │
                  │   (multi-turn conversation memory        │
                  │    keyed by session_id / thread_id)      │
                  └──────┬──────────┬──────────────────────┘
                         │          │
          ┌──────────────▼──┐  ┌────▼──────────────────┐
          │  Tool 1:        │  │  Tool 2:               │
          │  query_i4ms_    │  │  search_minor_mineral_ │
          │  database       │  │  policy                │
          │  (Text-to-SQL)  │  │  (Hybrid RAG)          │
          └────────┬────────┘  └────────┬──────────────┘
                   │                    │
        ┌──────────▼──────┐    ┌────────▼──────────────┐
        │  SQLite DB      │    │  Hybrid Retrieval     │
        │  + SQL Guard    │    │  pgvector (dense)     │
        │  + Scope Inject │    │  + BM25 (sparse)      │
        │  + PII Redact   │    │  + Cross-Encoder      │
        └─────────────────┘    │    Reranker            │
                               └────────────────────────┘
                  │
        ┌─────────▼──────────────┐
        │  Langfuse Observability │
        │  (traces, spans, scores,│
        │   user feedback loop)   │
        └─────────────────────────┘
```

---

## 3. Module-by-Module Breakdown

### 3.1 API Layer (`app/api/main.py`)
- **Framework:** FastAPI with lifespan management
- **Endpoints:**
  - `POST /chat` — Standard conversational endpoint with full response
  - `POST /chat/stream` — **[NEW]** Streaming endpoint using Server-Sent Events (SSE) for real-time token-by-token responses
  - `POST /feedback` — User feedback attachment to Langfuse traces
  - `GET /health` — Liveness probe with environment and Langfuse status
- **Security:** Prompt injection guard runs on every request before the agent
- **Access Control:** Role resolved from HTTP headers, simulating JWT/session auth

### 3.2 Agent Layer (`app/agents/`)
- **Orchestration:** LangGraph `StateGraph` implementing a **ReAct (Reason + Act)** loop
- **Graph Shape:** `START → agent → (conditional) → tools → agent → ... → END`
- **Conversation Memory:** **[NEW]** `MemorySaver` checkpointer maintains per-session message history, enabling multi-turn follow-up questions keyed by `session_id`
- **LLM:** OpenAI GPT-4o-mini with tool-calling capability
- **System Prompt:** Domain-specific instructions separating DATA vs POLICY questions, enforcing grounding, PII protection, and scope awareness
- **Streaming:** **[NEW]** `stream_agent()` async generator yields SSE events (thinking, tool_call, token, done, error)
- **Tools:**
  - `query_i4ms_database` — Text-to-SQL for structured data queries
  - `search_minor_mineral_policy` — Hybrid RAG for rule/procedure lookups
- **Citations:** Extracted from tool outputs using `[doc:<id>]` markers

### 3.3 RAG Pipeline (`app/rag/`)
- **Chunking:** `RecursiveCharacterTextSplitter` (800 chars, 120 overlap) supporting `.md` and `.txt` policy documents
- **Dense Retrieval:** OpenAI `text-embedding-3-small` embeddings stored in pgvector
- **Sparse Retrieval:** In-memory BM25Okapi index (`rank-bm25`)
- **Fusion:** Min-max normalized score blending with configurable `hybrid_alpha`
- **Reranking:** Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`)
- **Policy Corpus:** **[EXPANDED]** 7 documents:
  - `lease_and_royalty.md` — Lease grant, renewal, royalty, dead rent, suspension
  - `permit_and_pass.md` — e-Permit (Form L) and e-Transit Pass procedures
  - `ommc_rules_2016.md` — **[NEW]** Full OMMC Rules key provisions (concession types, eligibility, area limits, mining plan, penalties)
  - `registration_sop.md` — **[NEW]** Lessee/crusher/vehicle registration SOPs, user roles
  - `royalty_rates.md` — **[NEW]** Rate schedule for 14 minerals, dead rent, DMF, payment procedures
  - `enforcement_and_penalties.md` — **[NEW]** Inspections, check-gate verification, violations, seizure, appeals
  - `faq.md` — **[NEW]** 16 FAQs covering leases, permits, payments, and returns

### 3.4 Text-to-SQL (`app/database/`)
- **Flow:** `Natural language → LLM-generated SQL → SQL Guard → Scope Injection → Execute → Redact`
- **Schema:** 6 tables (lessee, lease, e_permit, e_transit_pass, royalty_payment, statutory_return)
- **Local DB:** SQLite with seed data (4 lessees, 5 leases, 4 permits, 6 passes, 3 royalties, 5 returns)
- **Security:** SQL Guard + external scope injection + PII redaction + SQLite PRAGMA query_only

### 3.5 Security (`app/security/`)
- **RBAC Roles:** LESSEE (own records), OFFICER (district-wide), ADMIN (full access)
- **SQL Guard:** Read-only validation, 30+ forbidden keywords, statement-stacking rejection, LIMIT injection
- **PII Redaction:** PAN and mobile masked for non-admin roles
- **Prompt Injection Guard:** **[NEW]** `input_guard.py` — Multi-layer defense:
  - 16 regex patterns detecting instruction override, role hijacking, fake system messages, prompt leaking, jailbreaks, code injection, encoding tricks
  - Structural heuristics: excessive length, suspicious separators, unusual formatting
  - Configurable risk-score threshold (default 0.6) with detailed verdict logging

### 3.6 Language Support (`app/core/language.py`) — **[NEW]**
- **Language Detection:** Unicode script analysis for Odia (U+0B00–U+0B7F), Hindi (Devanagari), and English
- **Translation Pipeline:** Odia/Hindi → English (before agent) → Odia/Hindi (after agent)
- **Implementation:** Uses the same OpenAI model for translation (zero additional dependencies)
- **Design:** Transparent to the agent — it always processes in English while users interact in their preferred language

### 3.7 Observability (`app/observability/`)
- **Platform:** Langfuse (cloud-hosted)
- **Features:** Full trace capture, custom span instrumentation, score attachment, user feedback loop, graceful degradation

### 3.8 Evaluation (`app/evaluation/`)
- **Metrics (LLM-as-Judge, RAGAS-style):** Faithfulness, Answer Relevancy, Context Precision, Context Recall
- **Harness:** End-to-end agent evaluation with Langfuse dataset sync

### 3.9 Infrastructure
- **Dockerfile:** Python 3.12-slim, production-ready
- **Docker Compose:** pgvector (PostgreSQL 16) + API service with health checks
- **Configuration:** Pydantic Settings with `.env` file support, 20+ parameters

---

## 4. Features Summary

| Feature | Status | Description |
|---------|--------|-------------|
| Agentic RAG (Policy Q&A) | Implemented | Hybrid dense+sparse+reranker retrieval over policy documents |
| Text-to-SQL (Data Q&A) | Implemented | LLM-generated SQL with multi-layer security guards |
| Role-Based Access Control | Implemented | 3 roles with parameterized tenant scoping |
| SQL Injection Prevention | Implemented | Denylist + read-only + scope injection + PRAGMA |
| PII Redaction | Implemented | PAN/mobile masked for non-admin roles |
| Langfuse Observability | Implemented | Traces, spans, scores, user feedback |
| RAGAS-style Evaluation | Implemented | 4 metrics with LLM-as-judge |
| Docker Deployment | Implemented | Compose with pgvector + API |
| **Conversation Memory** | **NEW** | Multi-turn chat with LangGraph MemorySaver |
| **Streaming Responses** | **NEW** | SSE endpoint with token-by-token streaming |
| **Odia Language Support** | **NEW** | Auto-detect + translate Odia/Hindi input and output |
| **Prompt Injection Guard** | **NEW** | 16-pattern regex guard with risk scoring |
| **Expanded Knowledge Base** | **NEW** | 7 policy documents (from 2) covering full i4MS domain |

---

## 5. Technology Summary

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web Framework | FastAPI | Async API server with SSE streaming |
| LLM Orchestration | LangGraph + LangChain | ReAct agent with tool routing + memory |
| Language Model | OpenAI GPT-4o-mini | Generation, Text-to-SQL, Eval, Translation |
| Embeddings | OpenAI text-embedding-3-small | Dense vector retrieval |
| Vector Database | pgvector (PostgreSQL 16) | Similarity search |
| Sparse Retrieval | rank-bm25 (BM25Okapi) | Keyword-based retrieval |
| Reranker | sentence-transformers (cross-encoder) | Final precision boost |
| Application DB | SQLite | Local stand-in for i4MS RDBMS |
| Observability | Langfuse | Traces, spans, scores, feedback |
| Evaluation | Custom RAGAS-style metrics | Faithfulness, relevancy, precision, recall |
| Config | Pydantic Settings | Type-safe env-based configuration |
| Containerization | Docker + Docker Compose | Reproducible deployment |
| Testing | pytest | Unit tests for security-critical paths |

---

## 6. File Structure

```
i4ms-chatbot/
├── app/
│   ├── agents/
│   │   ├── graph.py          # LangGraph ReAct agent + MemorySaver + streaming
│   │   ├── prompts.py        # Domain-specific system prompt
│   │   └── tools.py          # Tool definitions (DB + RAG)
│   ├── api/
│   │   └── main.py           # FastAPI endpoints (/chat, /chat/stream, /feedback, /health)
│   ├── core/
│   │   ├── config.py         # Pydantic Settings (20+ params)
│   │   ├── language.py       # [NEW] Odia/Hindi language detection + translation
│   │   ├── logging_config.py # Structured logging setup
│   │   └── schemas.py        # Shared data models
│   ├── database/
│   │   ├── connection.py     # SQLite connection with read-only enforcement
│   │   ├── schema.sql        # i4MS domain schema (6 tables)
│   │   └── text_to_sql.py    # NL → SQL → guard → scope → execute
│   ├── evaluation/
│   │   ├── evaluate.py       # Offline eval harness + Langfuse dataset sync
│   │   └── metrics.py        # 4 RAGAS-style LLM-as-judge metrics
│   ├── observability/
│   │   └── langfuse_client.py # Langfuse wrapper (traces, spans, scores)
│   ├── rag/
│   │   ├── bm25_index.py     # Sparse retrieval (BM25Okapi)
│   │   ├── chunking.py       # Document loading + text splitting
│   │   ├── embeddings.py     # OpenAI embeddings client
│   │   ├── fusion.py         # Min-max score blending
│   │   ├── retriever.py      # Hybrid retrieval orchestrator + reranker
│   │   └── vector_store.py   # pgvector wrapper
│   └── security/
│       ├── input_guard.py    # [NEW] Prompt injection detection (16 patterns)
│       ├── rbac.py           # Role-based access control (3 roles)
│       └── sql_guard.py      # SQL validation + denylist + LIMIT injection
├── data/
│   └── policies/             # Policy knowledge base (7 documents)
│       ├── lease_and_royalty.md
│       ├── permit_and_pass.md
│       ├── ommc_rules_2016.md          # [NEW]
│       ├── registration_sop.md         # [NEW]
│       ├── royalty_rates.md            # [NEW]
│       ├── enforcement_and_penalties.md # [NEW]
│       └── faq.md                       # [NEW]
├── scripts/
│   ├── ingest.py             # Document ingestion pipeline
│   ├── init_db.py            # Database seeding with sample data
│   └── run_eval.py           # Evaluation runner
├── tests/
│   └── test_core.py          # Unit tests (SQL guard, RBAC, fusion, injection guard, language)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## 7. How Each Enhancement Works

### 7.1 Conversation Memory (Multi-Turn Chat)
The agent uses LangGraph's `MemorySaver` checkpointer to persist message history per session. Each unique `session_id` maps to a `thread_id` in the checkpointer, so follow-up questions like *"What about the Sambalpur ones?"* work naturally after asking about active leases. The system prompt is injected once per thread, and the full conversation history flows through the ReAct loop on every turn.

### 7.2 Streaming Responses (SSE)
The `POST /chat/stream` endpoint returns a `text/event-stream` response using FastAPI's `StreamingResponse`. The agent's `astream_events()` method yields structured events:
- `thinking` — agent is processing the query
- `tool_call` — a tool invocation with name and arguments
- `tool_result` — tool execution completed
- `token` — a chunk of the final answer text (real-time)
- `done` — complete `ChatResponse` JSON payload
- `error` — failure details

### 7.3 Odia Language Support
A transparent translation layer sits between the API and the agent:
1. **Detect:** Unicode script analysis identifies Odia (U+0B00–U+0B7F), Hindi (Devanagari), or English
2. **Translate in:** Non-English queries are translated to English using the project's OpenAI model
3. **Process:** The agent works entirely in English (tools, RAG, SQL — all English)
4. **Translate out:** The English response is translated back to the detected language
This design avoids multilingual embeddings or separate models — it reuses what's already deployed.

### 7.4 Prompt Injection Guard
Every incoming query passes through `check_input()` before reaching the agent. The guard scores risk (0.0–1.0) using:
- **16 regex patterns** for: instruction override, role hijacking, fake system messages, prompt leaking, jailbreak keywords, code injection, encoding tricks
- **Structural heuristics:** excessive length (>2000 chars), suspicious separators (----, ====), unusual formatting
- Requests scoring above the threshold (0.6) are blocked with a 400 HTTP error and logged for audit

### 7.5 Expanded Knowledge Base
The policy corpus grew from 2 to 7 documents, covering the full i4MS domain:
- OMMC Rules 2016 key provisions (concession types, eligibility, area limits, mining plan, determination, penalties)
- Registration SOPs (lessee, crusher, vehicle, user roles, account security)
- Royalty rate schedule (14 minerals with rates, dead rent, DMF, payment procedures)
- Enforcement and penalties (inspections, check-gate verification, common violations, seizure, appeals)
- FAQ (16 frequently asked questions covering leases, permits, payments, returns)

---

*Generated on 2026-06-27 by analyzing the complete i4ms-chatbot codebase.*
