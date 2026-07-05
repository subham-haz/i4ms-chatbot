# Agentic AI & LangGraph Agents — Deep-Dive Q&A

> Code examples are drawn from `agentic-rag` and `i4ms-chatbot` in this workspace,
> both of which use LangGraph's ReAct agent pattern.

---

## Table of Contents
1. [What is an AI agent?](#1-what-is-an-ai-agent)
2. [Agent vs chain vs RAG pipeline](#2-agent-vs-chain-vs-rag-pipeline)
3. [ReAct pattern](#3-react-pattern)
4. [LangGraph StateGraph](#4-langgraph-stategraph)
5. [Agent state with TypedDict and add_messages](#5-agent-state-with-typeddict-and-add_messages)
6. [Defining tools with the @tool decorator](#6-defining-tools-with-the-tool-decorator)
7. [ToolNode — automatic tool execution](#7-toolnode--automatic-tool-execution)
8. [Conditional edges and loop routing](#8-conditional-edges-and-loop-routing)
9. [How and when does the agent loop terminate?](#9-how-and-when-does-the-agent-loop-terminate)
10. [Multi-step reasoning walkthrough](#10-multi-step-reasoning-walkthrough)
11. [Observability — tracing the agent loop](#11-observability--tracing-the-agent-loop)
12. [Citation extraction from tool outputs](#12-citation-extraction-from-tool-outputs)
13. [Memory and state across turns](#13-memory-and-state-across-turns)
14. [Security — safe tool execution](#14-security--safe-tool-execution)
15. [When to use agents vs simpler approaches](#15-when-to-use-agents-vs-simpler-approaches)

---

## 1. What is an AI agent?

**Q: What is an AI agent and how is it different from a regular LLM call?**

A regular LLM call is a single request-response: you send a prompt, you get a completion. An agent is a **loop**:

1. The LLM reasons about the user's goal
2. It decides to call a tool (or not)
3. If it called a tool, the result is fed back to the LLM
4. The LLM reasons again with the new information
5. Steps 2–4 repeat until the LLM decides it has enough to answer

This loop gives the LLM the ability to **gather information dynamically** rather than relying solely on what was passed in the prompt.

```
User: "What is our PTO policy, and how many days do I have left if I started Jan 1?"

Single LLM call:
    Prompt → LLM → "I don't have access to HR records" (or hallucinate)

Agent loop:
    Step 1: LLM decides → call knowledge_base_search("PTO policy")
    Step 2: Tool returns policy text
    Step 3: LLM decides → call calculator("25 - 5")
    Step 4: Tool returns "20"
    Step 5: LLM decides → generate final answer
    Answer: "You have 20 PTO days remaining. Policy: ..."
```

**Follow-up questions:**
- What is the difference between a tool-calling LLM and an autonomous agent?
- How do you prevent an agent from looping indefinitely?
- What guardrails would you put on an agent that can take real-world actions (send emails, write to DB)?

---

## 2. Agent vs chain vs RAG pipeline

**Q: When would you use a LangGraph agent vs a simple RAG chain?**

| Dimension | Simple RAG chain | LangGraph Agent |
|---|---|---|
| Control flow | Fixed: retrieve → prompt → generate | Dynamic: LLM decides what to do |
| Tool count | One (retrieval) | Many (search, calculator, SQL, web) |
| Multi-step reasoning | No | Yes — loops as needed |
| Latency | Lower (fewer LLM calls) | Higher (1+ extra call per tool use) |
| Predictability | High | Lower (LLM chooses path) |
| Complexity | Low | High |

**Rule of thumb:**
- If the answer always comes from the same retrieval pipeline → use a chain
- If answering requires different tools depending on the question → use an agent
- If you need to decompose a question into sub-questions → use an agent

In `enterprise-rag-bot`, every question goes through the same retrieve-then-generate pipeline (chain). In `agentic-rag` and `i4ms-chatbot`, the LLM can choose between `knowledge_base_search`, `calculator`, `text_to_sql`, etc., depending on what the question needs.

**Follow-up questions:**
- What is a DAG-based (directed acyclic graph) pipeline vs a cyclic agent graph?
- When does adding an agent actually make a system worse?
- How do you test an agent vs testing a deterministic pipeline?

---

## 3. ReAct pattern

**Q: What is ReAct and how is it implemented in your agent?**

ReAct = **Re**asoning + **Act**ing. The model alternates between two modes:
- **Thought/Reason:** The LLM examines the current state and decides what to do next
- **Action:** The LLM generates a structured tool call; the tool executes and returns an observation
- **Observation → Thought → Action → ...** until the LLM produces a final answer

In OpenAI's function-calling API, ReAct is implemented implicitly:
- Thought = any text generation before tool calls
- Action = `tool_calls` field in the LLM response
- Observation = `ToolMessage` returned by the tool execution

```
Messages list after one ReAct cycle:
[
    SystemMessage("You are a helpful assistant..."),
    HumanMessage("What is the PTO policy?"),
    AIMessage(tool_calls=[{"name": "knowledge_base_search", "args": {"query": "PTO policy"}}]),
    ToolMessage(content="Annual PTO is 25 days...", tool_call_id="call_abc"),
    AIMessage(content="Based on our policy documents, the PTO policy states...")
]
```

**Follow-up questions:**
- How does ReAct differ from Chain-of-Thought (CoT) prompting?
- What is the role of the system prompt in guiding the agent's reasoning?
- What happens when the LLM generates a malformed tool call?

---

## 4. LangGraph StateGraph

**Q: How does LangGraph's StateGraph work?**

`StateGraph` is a directed graph where:
- **Nodes** are Python functions that receive the current state and return updates to it
- **Edges** define which node to visit next (fixed or conditional)
- **State** is a shared object (TypedDict) that accumulates data as the graph executes

```python
# agentic-rag/app/agents/graph.py
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

def build_agent():
    graph = StateGraph(AgentState)

    # Node 1: LLM reasoning step
    graph.add_node("agent", _agent_node)

    # Node 2: Tool execution (LangGraph's prebuilt ToolNode handles dispatch)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    # Entry point: always start at the agent node
    graph.add_edge(START, "agent")

    # Conditional routing: go to tools OR finish
    graph.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", END: END}
    )

    # After tools, go back to agent for more reasoning
    graph.add_edge("tools", "agent")

    return graph.compile()
```

**Graph shape:**
```
START ──▶ agent ──(has tool_calls?)──▶ tools ──┐
              │                                  │
              └──(no tool_calls)──▶ END          │
              ◀─────────────────────────────────┘
```

**Follow-up questions:**
- What is the difference between `add_edge` and `add_conditional_edges`?
- How would you add a second conditional branch (e.g., route to a different node based on query type)?
- What does `graph.compile()` do internally?

---

## 5. Agent state with TypedDict and add_messages

**Q: What is `AgentState` and why use the `add_messages` annotation?**

```python
# agentic-rag/app/agents/graph.py
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

`AgentState` is the shared state object. Every node in the graph receives it as input and returns a dict with keys to update.

**Why `Annotated[list[AnyMessage], add_messages]`:**

Without annotation, returning `{"messages": [new_msg]}` from a node would **replace** the entire messages list with just `[new_msg]`, losing all history.

With `add_messages`, LangGraph treats the annotation as a **reducer**: it merges the returned list into the existing one by appending. This is how the conversation history accumulates across multiple reasoning steps.

```python
def _agent_node(state: AgentState) -> dict:
    response = _get_llm().invoke(state["messages"])
    # Returns {"messages": [response]} — add_messages APPENDS this, not replaces
    return {"messages": [response]}
```

**Follow-up questions:**
- What would happen if you forgot the `add_messages` annotation and just used `list[AnyMessage]`?
- Can you have multiple state fields? What types can they be?
- How would you add a counter to AgentState to limit the number of tool calls?

---

## 6. Defining tools with the @tool decorator

**Q: How do you define a tool that the agent can call?**

```python
# agentic-rag/app/agents/tools.py
from langchain_core.tools import tool
from app.rag.retriever import hybrid_retrieve

@tool
def knowledge_base_search(query: str) -> str:
    """Search the internal knowledge base for relevant context. Use this for any
    question that may be answered by company/internal documents."""
    chunks = hybrid_retrieve(query)
    if not chunks:
        return "No relevant documents found."

    # [doc:<id>] markers enable citation extraction later
    return "\n\n".join(
        f"[doc:{c.document_id} | source:{c.metadata.get('source', 'n/a')}]\n{c.content}"
        for c in chunks
    )

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression (e.g. '12.5 * (3 + 4)')."""
    return evaluate_expression(expression)  # safe AST-based eval, not eval()

@tool
def web_search(query: str) -> str:
    """Search the public web for current information not in the knowledge base."""
    return f"[stub] Would search: '{query}'"

ALL_TOOLS = [knowledge_base_search, calculator, web_search]
```

**What `@tool` does:**
1. Converts the function into a `LangChain Tool` object
2. Uses the **docstring** as the tool description sent to the LLM (the LLM reads this to decide when to call the tool)
3. Generates a JSON schema from the function signature for the tool call format
4. Makes the function callable via `tool.invoke({"query": "..."})`

**Binding tools to the LLM:**
```python
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(ALL_TOOLS)
# Now the LLM knows about all tools and can generate calls to them
```

**Follow-up questions:**
- Why is the docstring so important for tool quality?
- What would happen if two tools have identical or overlapping descriptions?
- How do you handle tool errors (exception inside a tool function)?

---

## 7. ToolNode — automatic tool execution

**Q: What is `ToolNode` and what does it do automatically?**

`ToolNode` is a prebuilt LangGraph component that:
1. Receives an `AIMessage` that contains `tool_calls`
2. Looks up which function to call based on `tool_call.name`
3. Calls the function with `tool_call.args`
4. Wraps the result in a `ToolMessage`
5. Returns it to be appended to the state

```python
from langgraph.prebuilt import ToolNode

# ToolNode automatically dispatches to the right function
tools_node = ToolNode(ALL_TOOLS)

graph.add_node("tools", tools_node)
```

**Without ToolNode, you'd write:**
```python
def manual_tools_node(state: AgentState) -> dict:
    last_msg = state["messages"][-1]
    results = []
    for call in last_msg.tool_calls:
        tool_fn = {t.name: t for t in ALL_TOOLS}[call["name"]]
        output = tool_fn.invoke(call["args"])
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    return {"messages": results}
```

`ToolNode` handles multiple parallel tool calls in one message, error handling, and output formatting — so you don't have to.

**Follow-up questions:**
- What happens if the LLM calls a tool name that doesn't exist in ALL_TOOLS?
- Can one AIMessage contain multiple tool calls? How does ToolNode handle that?
- How would you add timeout or retry logic around tool execution?

---

## 8. Conditional edges and loop routing

**Q: How does the agent decide to call another tool or stop?**

```python
# agentic-rag/app/agents/graph.py
def _should_continue(state: AgentState) -> str:
    """
    Routing function for the conditional edge after the agent node.
    Returns "tools" to continue the loop, or END to finish.
    """
    last = state["messages"][-1]

    # If the LLM generated tool_calls, route to the tools node
    if getattr(last, "tool_calls", None):
        return "tools"

    # No tool calls = the LLM produced a final answer → end the loop
    return END

# Wire it into the graph:
graph.add_conditional_edges(
    "agent",                          # source node
    _should_continue,                  # routing function
    {"tools": "tools", END: END}      # mapping: return value → destination node
)
```

This is what creates the loop:
- Every time the agent node runs, `_should_continue` is called
- If the LLM called a tool → execute tools → back to agent
- If the LLM produced a final answer → the graph terminates

**Follow-up questions:**
- How would you add a maximum iteration limit to prevent infinite loops?
- What would a routing function look like if you had multiple tool nodes (e.g., one for retrieval, one for execution)?
- What if the LLM generates both tool calls and text in the same response?

---

## 9. How and when does the agent loop terminate?

**Q: What exactly causes the agent to stop looping and return an answer?**

The agent terminates when the LLM generates a response with **no `tool_calls`** — meaning it decided it has enough information to answer directly.

Three ways this happens in practice:

1. **Natural completion:** LLM retrieves context, formulates answer, returns text only
2. **Nothing relevant found:** Tool returned "No relevant documents found" → LLM answers "I don't have information about this"
3. **Max iterations hit (if configured):** A recursion_limit can be set at compile or invoke time

```python
# Setting a max iteration limit at invoke time:
result = agent.invoke(init, config={"recursion_limit": 10})
# Raises GraphRecursionError if limit is exceeded
```

**Trace of a terminated loop:**
```
SystemMessage
HumanMessage("What is the notice period?")
AIMessage(tool_calls=[knowledge_base_search(query="notice period policy")])
ToolMessage(content="Notice period is 30 days for all employees...")
AIMessage(content="The notice period is 30 days for all employees.")  ← no tool_calls → END
```

**Follow-up questions:**
- What happens if you set `recursion_limit=2` and the agent needs 3 tool calls to answer?
- How would you handle a `GraphRecursionError` gracefully in the API layer?
- What would cause an agent to loop without ever terminating (infinite loop)?

---

## 10. Multi-step reasoning walkthrough

**Q: Walk me through a real multi-step agent interaction.**

**User question:** "Compare our PTO policy to industry standard and calculate if I worked 240 hours in Q1, how many PTO days did I accrue?"

```
Step 1 — Agent node receives [SystemMessage, HumanMessage]
    LLM thinks: "I need the PTO policy first"
    LLM output: AIMessage(tool_calls=[knowledge_base_search("PTO accrual policy")])

Step 2 — Tools node executes knowledge_base_search
    Returns: ToolMessage("PTO accrues at 1 day per 40 hours worked. Max 25 days/year.")

Step 3 — Agent node receives updated messages
    LLM thinks: "I have the accrual rate. 240 hours / 40 = 6 days. Let me verify with calculator."
    LLM output: AIMessage(tool_calls=[calculator("240 / 40")])

Step 4 — Tools node executes calculator
    Returns: ToolMessage("6.0")

Step 5 — Agent node receives updated messages
    LLM thinks: "I have all the information. I'll answer now."
    LLM output: AIMessage(content="Based on the PTO policy [doc:chunk_abc], you accrue 1 day
                per 40 hours worked. With 240 hours in Q1, you have accrued 6 PTO days.
                Industry standard is typically 10–15 days/year, so this is within range.")
    No tool_calls → _should_continue returns END

Final messages list:
[SystemMessage, HumanMessage, AIMessage(tool_calls), ToolMessage, AIMessage(tool_calls), ToolMessage, AIMessage(final)]
```

**Follow-up questions:**
- How does the LLM "know" it has all the information it needs to stop?
- What if the tool result contains conflicting information from two retrieved chunks?
- How would you expose the intermediate reasoning steps to the end user?

---

## 11. Observability — tracing the agent loop

**Q: How do you observe what the agent is doing in production?**

```python
# agentic-rag/app/agents/graph.py
from app.observability.langfuse_client import get_callback_handler

def run_agent(query: str, session_id: str | None = None) -> ChatResponse:
    agent = build_agent()

    # Attach Langfuse callback — this traces every LLM call and tool call
    config: dict = {}
    handler = get_callback_handler()
    if handler is not None:
        config["callbacks"] = [handler]
        config["metadata"] = {
            "langfuse_session_id": session_id,
            "langfuse_tags": ["agentic-rag"],
        }

    init = {
        "messages": [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
    }

    result = agent.invoke(init, config=config)
    ...
```

**What Langfuse captures per agent run:**
- Full span tree: one span per LLM call + one span per tool execution
- Token usage per LLM call (input + output tokens)
- Latency at each step
- Tool call arguments and return values
- Session grouping for multi-turn conversations
- User-submitted feedback scores (thumbs up/down → RAGAS score linkage)

**Retrieval-level tracing:**
```python
# agentic-rag/app/rag/retriever.py
with trace_span("hybrid_retrieve", input={"query": query}) as span:
    vector_hits = vector_search(query, settings.retrieval_top_k)
    bm25_hits = bm25_search(query, settings.retrieval_top_k)
    fused = _fuse(vector_hits, bm25_hits, settings.hybrid_alpha)
    reranked = _rerank(query, fused, settings.rerank_top_n)

    if span:
        span.update(output={
            "n_vector": len(vector_hits),
            "n_bm25": len(bm25_hits),
            "n_fused": len(fused),
            "returned": [c.document_id for c in reranked],
        })
```

**Follow-up questions:**
- What is the difference between a Langfuse trace and a span?
- How would you alert on a sudden drop in mean faithfulness score?
- What would you log at the tool level to diagnose retrieval quality problems?

---

## 12. Citation extraction from tool outputs

**Q: How does the agent track which documents it actually used in its answer?**

```python
# agentic-rag/app/agents/graph.py
import re

# Tools embed [doc:<id>] markers in their output:
# "[doc:chunk_abc | source:leave_policy.docx]\nAnnual PTO is 25 days..."

_CITATION_RE = re.compile(r"\[doc:([^\s|\]]+)")

def _extract_citations(messages: list[AnyMessage]) -> list[Citation]:
    citations: dict[str, Citation] = {}

    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = str(msg.content)
            for doc_id in _CITATION_RE.findall(content):
                snippet = content[:200]
                citations.setdefault(
                    doc_id, Citation(document_id=doc_id, snippet=snippet)
                )

    return list(citations.values())

def _tools_used(messages: list[AnyMessage]) -> list[str]:
    used: list[str] = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", []) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name and name not in used:
                used.append(name)
    return used
```

**Flow:**
1. `knowledge_base_search` tool formats output with `[doc:<chunk_id>]` markers
2. LLM may copy these markers into its final answer to cite sources
3. `_extract_citations()` scans `ToolMessage` objects for any `[doc:...]` patterns
4. `_tools_used()` scans for all `AIMessage.tool_calls` to log which tools ran

**Follow-up questions:**
- What's the difference between the documents the LLM cited in its answer vs the documents the tool returned?
- How would you verify that every claim in the LLM's answer can be traced to a specific citation?
- What if the LLM uses a retrieved fact but doesn't include the [doc:...] marker?

---

## 13. Memory and state across turns

**Q: How does the agent handle multi-turn conversations? Does it remember previous turns?**

Within a single agent invocation, the `AgentState.messages` list carries all context (including tool results). But the state is rebuilt from scratch for each new HTTP request.

**Persistence across requests:** The system prompt + user message is the starting point every time. To carry conversation history, you load prior messages from a database and prepend them.

```python
# agentic-rag/app/agents/graph.py — rebuilds state per request
def run_agent(query: str, session_id: str | None = None) -> ChatResponse:
    # No persistent memory — each call starts fresh
    init = {
        "messages": [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
    }
    result = agent.invoke(init, config=config)
```

**For multi-turn support, you'd add:**
```python
from langchain_postgres import PostgresChatMessageHistory

def run_agent_with_history(query: str, session_id: str) -> ChatResponse:
    history = PostgresChatMessageHistory(
        table_name="chat_history",
        session_id=session_id,
        connection=settings.database_url,
    )
    prior_messages = history.get_messages()[-10:]  # last 5 turns

    init = {
        "messages": [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            *prior_messages,        # inject prior conversation
            HumanMessage(content=query),
        ]
    }
    result = agent.invoke(init, config=config)

    # Persist new turn
    history.add_messages([
        HumanMessage(content=query),
        AIMessage(content=str(result["messages"][-1].content)),
    ])
    return build_response(result)
```

**Follow-up questions:**
- What is the LangGraph "checkpointing" feature and how does it differ from manually loading history?
- At what conversation length do you need to start summarizing or truncating history?
- How would you handle a user who wants to "reset" their conversation?

---

## 14. Security — safe tool execution

**Q: The agent has a calculator tool. Why not just use Python's `eval()`?**

```python
# agentic-rag/app/agents/safe_math.py (referenced in tools.py)
# tools.py delegates to evaluate_expression(), NOT raw eval()

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression (e.g. '12.5 * (3 + 4)')."""
    return evaluate_expression(expression)  # safe AST-based evaluation
```

`eval()` executes arbitrary Python, so a malicious expression like `"__import__('os').system('rm -rf /')"` would run. The safe alternative is AST parsing:

```python
import ast
import operator

SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def evaluate_expression(expr: str) -> str:
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def _eval_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return SAFE_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        return SAFE_OPS[type(node.op)](operand)
    raise ValueError(f"Unsupported node type: {type(node)}")
```

**i4ms-chatbot** takes this further with 4-layer SQL injection defense for the text-to-SQL tool:
1. Regex pattern blocking (DROP, DELETE, INSERT, UPDATE, EXEC keywords)
2. Allowlist of permitted SQL operations (SELECT only)
3. Parameterized queries for all user-controlled values
4. Row-level security via RBAC ContextVar

**Follow-up questions:**
- Besides `eval()`, what other dangerous Python builtins exist that LLMs might accidentally call?
- How would you sandbox a tool that needs to execute code (e.g., a Python REPL tool)?
- What is prompt injection and how can it affect an agent with tool access?

---

## 15. When to use agents vs simpler approaches

**Q: Given the added complexity, when is a LangGraph agent actually worth it?**

**Use an agent when:**
- The answer requires different information depending on the question type
- The user might ask follow-up questions that require prior tool results
- You need multi-step reasoning (retrieve → compute → retrieve again)
- Different tools are conditionally needed (RAG for policies, SQL for metrics, calculator for math)

**Use a simple chain/pipeline when:**
- Every question follows the same retrieve-then-answer path
- Latency is critical (each agent step adds ~0.5–2s for an LLM call)
- The question domain is narrow and predictable
- You need fully deterministic behavior

**Quantified cost of agent overhead:**
- Simple RAG chain: 1 LLM call (~1–2s)
- Agent with 1 tool call: 2 LLM calls (~2–4s)
- Agent with 2 tool calls: 3 LLM calls (~3–6s)

**Decision framework:**
```
Is the retrieval strategy always the same?
    Yes → use a chain (enterprise-rag-bot pattern)
    No  → use an agent (agentic-rag / i4ms-chatbot pattern)

Does answering require combining multiple tools?
    Yes → agent
    No  → chain

Do you need the LLM to decide what to look up?
    Yes → agent
    No  → hardcode the retrieval and skip the extra LLM call
```

**Follow-up questions:**
- How would you migrate a working agent system back to a simpler chain if the agent proves unreliable?
- Can you combine an agent outer loop with deterministic inner sub-pipelines?
- How do you write integration tests for a non-deterministic agent?
