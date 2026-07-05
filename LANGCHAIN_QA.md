# LangChain Stack — Deep-Dive Q&A

> Code examples are drawn from all three projects: `agentic-rag`, `enterprise-rag-bot`,
> and `i4ms-chatbot`. LangChain version: v0.2+ (LCEL-era API).

---

## Table of Contents
1. [What is LangChain and what problem does it solve?](#1-what-is-langchain-and-what-problem-does-it-solve)
2. [LCEL — the pipe syntax](#2-lcel--the-pipe-syntax)
3. [ChatPromptTemplate and MessagesPlaceholder](#3-chatprompttemplate-and-messagesplaceholder)
4. [Output parsers](#4-output-parsers)
5. [Document loaders](#5-document-loaders)
6. [RecursiveCharacterTextSplitter](#6-recursivecharactertextsplitter)
7. [HuggingFaceEmbeddings and embedding abstraction](#7-huggingfaceembeddings-and-embedding-abstraction)
8. [PGVector — LangChain's vector store interface](#8-pgvector--langchains-vector-store-interface)
9. [BM25Retriever](#9-bm25retriever)
10. [EnsembleRetriever and RRF](#10-ensembleretriever-and-rrf)
11. [BaseRetriever — the retriever interface](#11-baseretriever--the-retriever-interface)
12. [PostgresChatMessageHistory](#12-postgreschatmessagehistory)
13. [ChatOpenAI and bind_tools](#13-chatopenai-and-bind_tools)
14. [@tool decorator and tool schema generation](#14-tool-decorator-and-tool-schema-generation)
15. [Langfuse integration — @observe and callbacks](#15-langfuse-integration--observe-and-callbacks)
16. [lru_cache patterns for singleton resources](#16-lru_cache-patterns-for-singleton-resources)
17. [LangChain vs LangGraph — where each fits](#17-langchain-vs-langgraph--where-each-fits)

---

## 1. What is LangChain and what problem does it solve?

**Q: Why use LangChain? Can't you just call the OpenAI API directly?**

You can call the OpenAI API directly. LangChain adds:
- **Unified interface:** Swap `ChatOpenAI` for `ChatAnthropic` or `ChatGroq` without changing application code
- **Composability (LCEL):** Chain components with `|` — prompt → LLM → parser
- **Integrations:** `langchain_postgres`, `langchain_huggingface`, `langchain_community` — prebuilt connectors for PGVector, HuggingFace, BM25, etc.
- **Retriever abstraction:** All retrievers implement `.invoke(query)` → `list[Document]`, regardless of backend
- **Memory/history:** `PostgresChatMessageHistory`, `RedisChatMessageHistory` with a common interface
- **Tracing hooks:** Callback system lets Langfuse (and others) intercept every LLM call automatically

**When LangChain adds friction:**
- Simple single-LLM-call applications (direct API is cleaner)
- When you need control the library abstracts away (custom streaming, batch optimization)
- When library version churn breaks your code (LangChain has frequent breaking changes)

**Follow-up questions:**
- What changed between LangChain v0.1 (chain-based) and v0.2 (LCEL-based)?
- When would you replace LangChain entirely with direct API calls?
- What is the difference between `langchain`, `langchain-core`, `langchain-community`, and `langchain-openai`?

---

## 2. LCEL — the pipe syntax

**Q: What is LCEL and how does the `|` operator work?**

LCEL (LangChain Expression Language) lets you compose components using `|` (pipe), similar to Unix pipes. Each component implements `Runnable` which means it has `.invoke()`, `.batch()`, and `.stream()`.

```python
# enterprise-rag-bot/rag/chain.py
from langchain_core.output_parsers import StrOutputParser
from rag.prompts import RAG_PROMPT
from rag.llm import get_llm

# LCEL chain: prompt → LLM → parser
chain = RAG_PROMPT | get_llm() | StrOutputParser()

answer = chain.invoke({
    "context": context,
    "chat_history": chat_history,
    "question": question,
})
```

**How `|` works internally:**
```python
# RAG_PROMPT | get_llm() creates a RunnableSequence equivalent to:
class RunnableSequence:
    def invoke(self, input):
        result1 = RAG_PROMPT.invoke(input)   # dict → ChatPromptValue
        result2 = get_llm().invoke(result1)  # ChatPromptValue → AIMessage
        return result2

# Adding | StrOutputParser() extends the sequence:
# dict → ChatPromptValue → AIMessage → str
```

**Benefits over explicit method calls:**
- `.batch([input1, input2])` automatically parallelizes with threads
- `.stream(input)` enables streaming output with no code changes
- `.with_retry(stop_after_attempt=3)` adds retry logic to any step

**Follow-up questions:**
- How do you add retry logic or fallbacks to an LCEL chain?
- What is the difference between `RunnableSequence` and `RunnableParallel`?
- How does streaming work through an LCEL chain? What does each component need to implement?

---

## 3. ChatPromptTemplate and MessagesPlaceholder

**Q: How do you build a prompt that includes chat history?**

```python
# enterprise-rag-bot/rag/prompts.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

RAG_SYSTEM_PROMPT = """\
You are an internal HR policy assistant.

Rules:
1. Answer only from the policy context provided below.
2. If the answer is not present, say: "I could not find this information in the policy documents."
3. Keep the answer short and clear.
4. Mention the source file name when citing policy information.

Policy Context:
{context}"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),                            # system message with {context}
    MessagesPlaceholder(variable_name="chat_history", optional=True),  # injects N prior messages
    ("human", "{question}"),                                  # current user question
])
```

**What `MessagesPlaceholder` does:**
It accepts a list of `HumanMessage` / `AIMessage` objects at render time and inserts them as real message objects (not text strings). This preserves the message role format that chat models expect.

```python
from langchain_core.messages import HumanMessage, AIMessage

# When invoking the chain:
chain.invoke({
    "context": retrieved_text,
    "chat_history": [
        HumanMessage(content="What is the leave policy?"),
        AIMessage(content="Annual leave is 25 days."),
    ],
    "question": "What about for contractors?",
})
# Produces:
# [SystemMessage("...policy context..."), HumanMessage("What is the leave policy?"),
#  AIMessage("Annual leave is 25 days."), HumanMessage("What about for contractors?")]
```

**Follow-up questions:**
- What is the difference between a `SystemMessage` and a `HumanMessage` in the OpenAI chat format?
- What happens if `chat_history` is an empty list and `optional=True` is set?
- How would you add a different system prompt for different user roles?

---

## 4. Output parsers

**Q: What is `StrOutputParser` and when would you use a different parser?**

```python
from langchain_core.output_parsers import StrOutputParser

# The LLM returns an AIMessage. StrOutputParser extracts .content as a plain string.
chain = prompt | llm | StrOutputParser()
result: str = chain.invoke(...)
```

**Other parsers:**

```python
# JSON output parser — for structured LLM responses
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

class EvalScore(BaseModel):
    score: float
    reason: str

parser = JsonOutputParser(pydantic_object=EvalScore)
chain = eval_prompt | llm | parser
result: EvalScore = chain.invoke(...)

# Used in agentic-rag/app/evaluation/metrics.py (manual JSON parsing instead):
def _ask_score(prompt: str) -> tuple[float, str]:
    raw = _judge().invoke(prompt).content
    text = str(raw).strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```")
    data = json.loads(text)
    return float(data.get("score", 0.0)), str(data.get("reason", ""))
```

**Pydantic output parser:**
```python
from langchain_core.output_parsers import PydanticOutputParser

class Citation(BaseModel):
    document_id: str
    snippet: str
    confidence: float

parser = PydanticOutputParser(pydantic_object=Citation)
# Automatically adds format instructions to the prompt and validates output
```

**Follow-up questions:**
- What happens when the LLM outputs malformed JSON and you're using JsonOutputParser?
- How would you implement a custom output parser for a domain-specific format?
- What is the `OutputFixingParser` and when is it useful?

---

## 5. Document loaders

**Q: How do you load different file types into LangChain `Document` objects?**

LangChain's `langchain_community.document_loaders` provides loaders for most formats. Each loader returns `list[Document]` where each `Document` has `.page_content` (text) and `.metadata` (source file, page number, etc.).

```python
# PDF loading
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("policy.pdf")
docs = loader.load()
# docs[0].page_content = "Leave Policy\n\nEmployees are entitled to..."
# docs[0].metadata = {"source": "policy.pdf", "page": 0}

# Text/Markdown file
from langchain_community.document_loaders import TextLoader
loader = TextLoader("readme.md", encoding="utf-8")
docs = loader.load()

# Unstructured (handles PDF, DOCX, HTML, email, etc. automatically)
from langchain_community.document_loaders import UnstructuredFileLoader
loader = UnstructuredFileLoader("report.docx")
docs = loader.load()

# Directory of files
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("./documents/", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()
```

**In the enterprise-rag-bot ingestion pipeline:**
```python
# enterprise-rag-bot/ingestion/pipeline.py
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_document(file_path: str) -> list[Document]:
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    elif file_path.endswith((".txt", ".md")):
        return TextLoader(file_path).load()
    raise ValueError(f"Unsupported file type: {file_path}")
```

**Follow-up questions:**
- How does the metadata from loaders propagate through chunking and into PGVector?
- What is the difference between `loader.load()` and `loader.lazy_load()`?
- How would you handle large PDFs (100+ pages) without loading everything into memory at once?

---

## 6. RecursiveCharacterTextSplitter

**Q: How does `RecursiveCharacterTextSplitter` work and why is it preferred?**

```python
# enterprise-rag-bot/ingestion/splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=800,       # target characters per chunk
        chunk_overlap=150,    # characters repeated between adjacent chunks
        separators=["\n\n", "\n", ".", " ", ""],
        # tries each separator in order; falls back to the next if chunk still too large
    )
```

**Splitting algorithm:**
1. Try splitting on `"\n\n"` (paragraph breaks)
2. If any resulting piece is still > 800 chars, try `"\n"`
3. If still too large, try `"."`
4. If still too large, try `" "`
5. If still too large, split on character boundary

This preserves paragraph → sentence → word hierarchy, keeping semantically cohesive units together.

**After splitting, metadata is propagated:**
```python
splitter = get_splitter()
chunks = splitter.split_documents(raw_docs)
# Each chunk inherits metadata from its source document
# chunks[0].metadata = {"source": "policy.pdf", "page": 0, "chunk_index": 0}
```

**Follow-up questions:**
- What is the difference between `split_documents()` and `split_text()`?
- Why does token-based splitting (by token count, not character count) matter for certain models?
- How would you use `MarkdownHeaderTextSplitter` for structured documents?

---

## 7. HuggingFaceEmbeddings and embedding abstraction

**Q: How does LangChain abstract the embedding model so you can swap it out?**

All embeddings implement `Embeddings` from `langchain_core`:
```python
class Embeddings(ABC):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

This means PGVector, BM25, and all retrievers only depend on this interface — you can swap HuggingFace for OpenAI without touching retrieval code.

```python
# enterprise-rag-bot/rag/embeddings.py — local HuggingFace model
from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache

@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
```

```python
# agentic-rag — OpenAI embeddings (same interface, different backend)
from langchain_openai import OpenAIEmbeddings

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openai_api_key,
    )
```

**Swapping in one line:**
```python
# Switch from HuggingFace to OpenAI — nothing else changes
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# PGVector, BM25, and all retrievers continue to work unchanged
```

**Important constraint:** Ingestion and query must use the same embedding model. Changing models requires re-embedding all documents.

**Follow-up questions:**
- What is the `Embeddings` ABC in `langchain_core` and why does it matter?
- How would you test that your embedding model is producing sensible vectors?
- What is `batch_size` in `HuggingFaceEmbeddings` and when does it matter?

---

## 8. PGVector — LangChain's vector store interface

**Q: How does LangChain's PGVector class work?**

`langchain_postgres.PGVector` wraps the pgvector PostgreSQL extension. It creates two tables automatically: `langchain_pg_collection` (collection registry) and `langchain_pg_embedding` (vector + text + metadata storage).

```python
# agentic-rag/app/rag/vector_store.py (simplified)
from langchain_postgres import PGVector
from app.rag.embeddings import get_embeddings
from functools import lru_cache

@lru_cache(maxsize=1)
def get_vector_store() -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name="document_chunks",
        connection=settings.database_url,
        # distance_strategy defaults to cosine — not explicitly set in any project
    )
```

**Adding documents (at ingestion time):**
```python
store = get_vector_store()
store.add_documents(chunks)  # embeds + stores in one call
```

**Querying (at retrieval time):**
```python
# Returns list[tuple[Document, float]] — document + cosine distance score
results = store.similarity_search_with_score(query, k=5)

# As a retriever (returns list[Document]):
retriever = store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)
docs = retriever.invoke(query)

# With metadata filter:
results = store.similarity_search_with_score(
    query, k=5,
    filter={"department": "HR"},
)
```

**Underlying SQL (for each query):**
```sql
SELECT document, cmetadata, embedding <=> $1 AS distance
FROM langchain_pg_embedding
WHERE collection_id = $2
ORDER BY distance
LIMIT 5;
-- <=> is the pgvector cosine distance operator
```

**Follow-up questions:**
- What is the difference between `<=>` (cosine distance) and `<#>` (negative inner product) in pgvector?
- How does PGVector handle the case where the collection doesn't exist yet?
- What does `pre_delete_collection=True` do in PGVector's constructor?

---

## 9. BM25Retriever

**Q: How is BM25Retriever used in LangChain?**

```python
# enterprise-rag-bot/rag/retriever.py
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

def _build_bm25_retriever(top_k: int = 5) -> BM25Retriever:
    docs = _load_all_documents()  # load all chunks from PGVector

    if not docs:
        # BM25Retriever requires at least one document
        return BM25Retriever.from_documents(
            [Document(page_content="placeholder")], k=top_k
        )

    return BM25Retriever.from_documents(docs, k=top_k)
```

**Internals:**
- `BM25Retriever.from_documents()` calls `rank_bm25.BM25Okapi` on the tokenized corpus
- The entire index is held in memory — no persistence
- `k` is the number of documents to return per query
- Tokenization is basic whitespace splitting by default (you can provide a custom `preprocess_func`)

**Custom tokenizer:**
```python
import re

def custom_tokenizer(text: str) -> list[str]:
    # lowercase + remove punctuation + split on whitespace
    return re.sub(r"[^\w\s]", "", text.lower()).split()

retriever = BM25Retriever.from_documents(docs, k=5, preprocess_func=custom_tokenizer)
```

**Stale index problem:**
The BM25 index is built once. New documents added to PGVector don't appear in BM25 until the retriever is rebuilt. This is why `refresh_retriever()` clears the `lru_cache`.

```python
def refresh_retriever():
    get_retriever.cache_clear()  # forces full rebuild on next call
    logger.info("Retriever cache cleared")
```

**Follow-up questions:**
- Why does BM25Retriever need all documents loaded into memory at startup?
- How would you replace BM25Retriever with an Elasticsearch retriever in LangChain?
- What is `BM25Okapi` vs `BM25Plus`? Which is used by default?

---

## 10. EnsembleRetriever and RRF

**Q: What is `EnsembleRetriever` and how does it fuse results?**

```python
# enterprise-rag-bot/rag/retriever.py
from langchain.retrievers import EnsembleRetriever
from functools import lru_cache

@lru_cache(maxsize=1)
def get_retriever(top_k: int = 5) -> EnsembleRetriever:
    vector_retriever = get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    bm25_retriever = _build_bm25_retriever(top_k=top_k)

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],  # equal weighting; used in RRF score calculation
    )
```

**How EnsembleRetriever applies RRF:**

When you call `ensemble.invoke(query)`:
1. Each retriever runs independently and returns its ranked list of documents
2. For each document in each list, compute: `score_i = weight_i / (k + rank_i)` where k=60
3. Sum scores across all retrievers for each document
4. Sort by total score descending
5. Deduplicate and return the merged list

```python
# RRF math for a document appearing at rank 2 in vector search, rank 5 in BM25:
vector_score = 0.5 / (60 + 2)  # = 0.00806
bm25_score   = 0.5 / (60 + 5)  # = 0.00769
total        = 0.01575          # this document's final RRF score
```

**Why weights still matter with RRF:**
The `weights` parameter in EnsembleRetriever multiplies the per-retriever RRF contribution. `[0.5, 0.5]` gives equal influence; `[0.7, 0.3]` would make vector search contribute more.

**Follow-up questions:**
- What is the k=60 constant in RRF and why was 60 chosen?
- Can `EnsembleRetriever` combine more than 2 retrievers? What changes?
- How would you tune the `weights` parameter if you find BM25 matching too aggressively?

---

## 11. BaseRetriever — the retriever interface

**Q: What is the `BaseRetriever` interface and why does it matter?**

All LangChain retrievers implement `BaseRetriever`:

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class BaseRetriever(ABC):
    def invoke(self, query: str, **kwargs) -> list[Document]:
        return self._get_relevant_documents(query, **kwargs)

    @abstractmethod
    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        ...
```

This means `EnsembleRetriever`, `BM25Retriever`, `PGVector.as_retriever()`, and any custom retriever you write are interchangeable anywhere that accepts a `BaseRetriever`.

**Custom retriever:**
```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

class HybridRerankerRetriever(BaseRetriever):
    """Wraps the full pipeline (vector + BM25 + fusion + rerank) as a single retriever."""

    top_k: int = 5
    alpha: float = 0.5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        vector_hits = vector_search(query, self.top_k)
        bm25_hits = bm25_search(query, self.top_k)
        fused = fuse(vector_hits, bm25_hits, self.alpha)
        reranked = rerank(query, fused, top_n=3)
        return [
            Document(page_content=c.content, metadata=c.metadata)
            for c in reranked
        ]
```

**Follow-up questions:**
- What is `CallbackManagerForRetrieverRun` and when do you use it?
- What is the difference between a `Retriever` and a `VectorStore` in LangChain?
- Can you use a `BaseRetriever` inside an LCEL chain? What does it return?

---

## 12. PostgresChatMessageHistory

**Q: How do you persist conversation history in PostgreSQL with LangChain?**

```python
# enterprise-rag-bot/rag/memory.py
from langchain_postgres import PostgresChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

def get_chat_history(session_id: str) -> PostgresChatMessageHistory:
    return PostgresChatMessageHistory(
        table_name="chat_history",
        session_id=session_id,
        connection=settings.database_url,
    )

def save_chat(session_id: str, user_question: str, bot_answer: str) -> None:
    history = get_chat_history(session_id)
    history.add_messages([
        HumanMessage(content=user_question),
        AIMessage(content=bot_answer),
    ])

def get_recent_messages(session_id: str, limit: int = 10) -> list:
    history = get_chat_history(session_id)
    messages = history.get_messages()
    return messages[-limit:] if len(messages) > limit else messages
```

**The underlying schema** (LangChain creates this automatically):
```sql
CREATE TABLE chat_history (
    id          SERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL,
    message     JSONB NOT NULL,    -- {"type": "human", "data": {"content": "..."}}
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON chat_history (session_id);
```

**Why PostgreSQL for history (not Redis):**
- Redis TTL would evict old conversations automatically — not suitable for audit trails
- PostgreSQL persists across restarts, Redis can lose data unless AOF is configured
- History volume is low (not a hot-path cache); query performance is not a bottleneck

**Injecting history into the prompt:**
```python
chat_history = get_recent_messages(session_id, limit=10)
answer = chain.invoke({
    "context": context,
    "chat_history": chat_history,  # MessagesPlaceholder injects these
    "question": query,
})
```

**Follow-up questions:**
- What is the session_id scoping — can two users share a session?
- How would you implement conversation summarization when history gets too long?
- What is `RedisChatMessageHistory` and when would you use it over Postgres?

---

## 13. ChatOpenAI and bind_tools

**Q: How do you wire up the LLM so it knows about tools?**

```python
# agentic-rag/app/agents/graph.py
from langchain_openai import ChatOpenAI
from functools import lru_cache
from app.agents.tools import ALL_TOOLS

@lru_cache
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,              # "gpt-4o-mini"
        temperature=settings.llm_temperature,  # 0.0 for deterministic
        api_key=settings.openai_api_key,
    ).bind_tools(ALL_TOOLS)  # sends tool schemas to the API with every call
```

**What `.bind_tools()` does:**

1. Converts each `@tool` function into a JSON schema (function name, description, parameter types)
2. These schemas are sent in the `tools` parameter of the OpenAI Chat Completions API call
3. The LLM can then respond with `tool_calls` in its message — a structured call to one or more tools

```python
# What the API call looks like under the hood:
openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "knowledge_base_search",
                "description": "Search the internal knowledge base for relevant context...",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        },
        # ... other tools
    ]
)
```

**Temperature settings:**
- `temperature=0.0` for the agent LLM → more deterministic tool selection and answers
- `temperature=0.0` for the evaluation judge LLM → reproducible scores across runs

**Follow-up questions:**
- What is `tool_choice="auto"` vs `tool_choice="required"` in the OpenAI API?
- How does the `parallel_tool_calls` parameter affect agent behavior?
- What is the difference between `bind_tools()` and `with_structured_output()`?

---

## 14. @tool decorator and tool schema generation

**Q: How does the `@tool` decorator work and what does it produce?**

```python
# agentic-rag/app/agents/tools.py
from langchain_core.tools import tool

@tool
def knowledge_base_search(query: str) -> str:
    """Search the internal knowledge base for relevant context. Use this for any
    question that may be answered by company/internal documents."""
    chunks = hybrid_retrieve(query)
    if not chunks:
        return "No relevant documents found."
    return "\n\n".join(
        f"[doc:{c.document_id} | source:{c.metadata.get('source', 'n/a')}]\n{c.content}"
        for c in chunks
    )
```

**What `@tool` produces:**
```python
knowledge_base_search.name        # "knowledge_base_search"
knowledge_base_search.description # "Search the internal knowledge base..." (from docstring)
knowledge_base_search.args_schema # Pydantic model: {"query": str}

# You can still call it as a function:
result = knowledge_base_search.invoke({"query": "PTO policy"})
# Or via the agent framework (same thing internally):
result = knowledge_base_search(query="PTO policy")
```

**More complex tool with Pydantic schema:**
```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    top_k: int = Field(default=5, description="Number of results to return")

@tool(args_schema=SearchInput)
def knowledge_base_search(query: str, top_k: int = 5) -> str:
    """Search the knowledge base."""
    ...
```

**Follow-up questions:**
- What is the significance of the docstring for tool quality? How do you write a good one?
- Can a tool return structured data (dict/list) instead of a string? How does the LLM handle it?
- How would you add input validation or type coercion to a tool?

---

## 15. Langfuse integration — @observe and callbacks

**Q: How does Langfuse integrate with LangChain and how does `@observe` work?**

Two integration patterns are used:

**Pattern 1 — LangChain callback (for chains and agents):**
```python
# agentic-rag/app/agents/graph.py
from app.observability.langfuse_client import get_callback_handler

def run_agent(query: str, session_id: str | None = None) -> ChatResponse:
    handler = get_callback_handler()
    config = {}
    if handler is not None:
        config["callbacks"] = [handler]
        config["metadata"] = {
            "langfuse_session_id": session_id,
            "langfuse_tags": ["agentic-rag"],
        }

    result = agent.invoke(init, config=config)
```

The callback is invoked automatically at every LLM call, tool call, and chain step — no manual instrumentation needed.

**Pattern 2 — `@observe` decorator (for custom pipeline steps):**
```python
# enterprise-rag-bot/rag/chain.py
from langfuse import observe

@observe(name="retrieve_documents")
def _retrieve(query: str) -> list[Document]:
    # Langfuse starts a span when this function is entered
    # and ends it when the function returns — capturing latency automatically
    retriever = get_retriever()
    return retriever.invoke(query)

@observe(name="llm_generate")
def _generate(context: str, chat_history: list, question: str) -> str:
    chain = RAG_PROMPT | get_llm() | StrOutputParser()
    return chain.invoke({"context": context, "chat_history": chat_history, "question": question})

@observe(name="rag_pipeline")
def generate_answer(query: str, session_id: str = "default") -> dict:
    # This creates a parent span; _retrieve and _generate create child spans
    ...
```

**What Langfuse captures per trace:**
- Span tree showing the nested function call hierarchy
- Input and output at each span
- Latency per span
- Token usage for LLM spans
- User ID and session ID for grouping
- Custom scores (from RAGAS evaluation)

**Follow-up questions:**
- What is the difference between a Langfuse trace and a span?
- How do you attach a manual score (e.g., user thumbs-up) to a Langfuse trace?
- How would you use Langfuse to A/B test two different prompt templates?

---

## 16. lru_cache patterns for singleton resources

**Q: Why is `lru_cache` used so heavily across these projects?**

```python
# Pattern repeated across all three projects for embeddings, vector stores, LLMs, retrievers:
from functools import lru_cache

@lru_cache(maxsize=1)   # maxsize=1 → exactly one cached instance
def get_embeddings() -> HuggingFaceEmbeddings:
    # Loading this model downloads ~80 MB and takes ~2-5 seconds
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", ...)

@lru_cache(maxsize=1)
def get_vector_store() -> PGVector:
    # Creates DB connection pool — expensive, should be reused
    return PGVector(embeddings=get_embeddings(), ...)

@lru_cache(maxsize=1)
def get_retriever(top_k: int = 5) -> EnsembleRetriever:
    # Loads all documents from DB to build BM25 index — O(n) startup cost
    return EnsembleRetriever(...)

@lru_cache
def build_agent():
    # Compiles LangGraph — only needs to happen once
    return graph.compile()
```

**Why not module-level singletons?**
```python
# This approach is fine but harder to test and reset:
_embeddings = None
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(...)
    return _embeddings

# lru_cache is cleaner and provides .cache_clear() for resetting in tests:
get_retriever.cache_clear()  # force BM25 rebuild after new documents are ingested
```

**In tests:**
```python
def test_retriever():
    get_retriever.cache_clear()  # ensure fresh retriever for each test
    retriever = get_retriever()
    ...
```

**Follow-up questions:**
- What is the difference between `@lru_cache(maxsize=1)` and `@lru_cache(maxsize=None)`?
- `lru_cache` is not thread-safe for the initial population. Does that matter here?
- When would you use `@cache` (Python 3.9+) vs `@lru_cache`?

---

## 17. LangChain vs LangGraph — where each fits

**Q: You used both LangChain and LangGraph. What's the difference?**

**LangChain** provides:
- Component abstractions (`Embeddings`, `VectorStore`, `BaseRetriever`, `BaseLLM`)
- LCEL composition (`prompt | llm | parser`)
- Integrations (`langchain_openai`, `langchain_postgres`, `langchain_community`)
- Pre-built chains (`RetrievalQA`, `ConversationalRetrievalChain`)

**LangGraph** provides:
- Stateful graph execution (`StateGraph`, `add_messages` reducers)
- Cycle support (the agent loop — LangChain chains are DAGs only)
- Prebuilt agent primitives (`ToolNode`, `create_react_agent`)
- Human-in-the-loop checkpointing

```
LangChain LCEL (DAG — enterprise-rag-bot):
    RAG_PROMPT | get_llm() | StrOutputParser()
    → one pass, no loops, no branching

LangGraph (cyclic — agentic-rag, i4ms-chatbot):
    START → agent → tools → agent → ... → END
    → loops as many times as needed
```

**Where each is used in these projects:**

| Project | LangChain | LangGraph |
|---|---|---|
| enterprise-rag-bot | LCEL chain, EnsembleRetriever, PGVector, BM25Retriever, PostgresChatMessageHistory | Not used |
| agentic-rag | Embeddings, PGVector, tool decorator, ChatOpenAI | StateGraph, ToolNode, add_messages |
| i4ms-chatbot | Embeddings, PGVector, tool decorator | StateGraph, ToolNode, add_messages |

**Follow-up questions:**
- Can you use LangGraph without LangChain? What would you lose?
- What is `create_react_agent` in LangGraph and when would you use it instead of building the graph manually?
- What is LangGraph's checkpointing feature and how does it enable human-in-the-loop workflows?
