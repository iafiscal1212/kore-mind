# kore vs Competitors: Honest Comparison

## The Numbers

|  | **kore** | **Letta/MemGPT** | **LangChain Memory** | **LlamaIndex Memory** |
|--|---------|-----------------|---------------------|----------------------|
| **Python LOC** | **969** (676+293) | ~25,000 | ~3,000 | ~4,000 |
| **Source files** | **10** (7+3) | ~180 | ~15 | ~12 |
| **Dependencies** | **0** (stdlib only) | 57 | 8 | 29 |
| **Requires server** | **No** | Yes (port 8283) | No | No |
| **Hello world lines** | **6** | ~25 | ~10 | ~15 |
| **API methods to learn** | **5** | ~20 | ~15 | ~10 |
| **Works offline** | **Yes** (+ Ollama) | No | No | No |
| **Min Python** | 3.10 | 3.11 | 3.10 | 3.9 |

## Hello World Comparison

### kore (6 lines)

```python
from kore_mind import Mind

mind = Mind("agent.db")
mind.experience("User prefers concise answers")
memories = mind.recall("style")
identity = mind.reflect()
```

### Letta/MemGPT (~25 lines)

```python
from letta_client import Letta

# Requires: letta server running on port 8283
client = Letta(base_url="http://localhost:8283")

agent = client.agents.create(
    model="openai/gpt-4o-mini",
    embedding="openai/text-embedding-3-small",
    memory_blocks=[
        {"label": "human", "value": "Unknown user"},
        {"label": "persona", "value": "Helpful assistant"},
    ],
)

response = client.agents.messages.create(
    agent_id=agent.id,
    messages=[{
        "role": "user",
        "content": "Remember that I prefer concise answers",
    }],
)

# Recall requires sending another message through the agent
response = client.agents.messages.create(
    agent_id=agent.id,
    messages=[{"role": "user", "content": "What do you know about my style?"}],
)
```

### LangChain (~10 lines)

```python
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Requires: OPENAI_API_KEY
llm = ChatOpenAI(model="gpt-3.5-turbo")
memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)

chain.predict(input="Remember that I prefer concise answers")
chain.predict(input="What do you know about my style?")
```

### LlamaIndex (~15 lines)

```python
from llama_index.core.memory import Memory
from llama_index.core.memory.memory import StaticMemoryBlock
from llama_index.core.agent.workflow import AgentWorkflow

# Requires: OPENAI_API_KEY
memory = Memory.from_defaults(
    session_id="session-1",
    memory_blocks=[
        StaticMemoryBlock(label="preferences", value="Unknown"),
    ],
    token_limit=30000,
)

agent = AgentWorkflow.from_tools_or_functions([], memory=memory)
response = agent.run("Remember that I prefer concise answers")
```

## What kore does that others don't

| Feature | kore | MemGPT | LangChain | LlamaIndex |
|---------|------|--------|-----------|------------|
| Memory decay | **Yes** (exponential, configurable) | No | No | No |
| Memory consolidation | **Yes** (auto-merge similar) | No | No | No |
| Emergent identity | **Yes** (computed from patterns) | Partial | No | No |
| Works without LLM | **Yes** | No | No | No |
| One-file mind (SQLite) | **Yes** | No (Postgres) | No | Partial |
| Reinforcement on recall | **Yes** | No | No | No |

## What others do that kore doesn't (yet)

| Feature | MemGPT | LangChain | LlamaIndex |
|---------|--------|-----------|------------|
| Multi-agent orchestration | Yes | Yes | Yes |
| Tool calling | Yes | Yes | Yes |
| RAG pipeline | No | Yes | Yes |
| Production cloud hosting | Yes | Yes (LangSmith) | Yes (LlamaCloud) |
| Async support | Yes | Yes | Yes |
| Enterprise support | Yes | Yes | Yes |

## Dependency Tree

```
kore-mind
└── (nothing — stdlib only)

kore-bridge
└── kore-mind

vs

letta (57 deps)
├── fastapi, uvicorn, grpcio    (server stack)
├── sqlalchemy, alembic          (ORM + migrations)
├── numpy, llama-index           (ML stack)
├── openai, anthropic, mistral   (LLM clients)
├── nltk, tiktoken               (NLP)
└── ... 45 more

langchain (8 core, ~30 with integrations)
├── pydantic, langsmith
├── tenacity, jsonpatch
└── ... per-integration deps

llama-index (29 core)
├── sqlalchemy, aiohttp, numpy
├── nltk, tiktoken, networkx
└── ... 20 more
```

## The SQLite Analogy

| Analogy | Simple | Complex |
|---------|--------|---------|
| Databases | **SQLite** | PostgreSQL, MySQL |
| LLM inference | **llama.cpp** | vLLM, TensorRT |
| LLM memory | **kore** | MemGPT, LangChain |

kore wins when you want:
- Zero setup
- One file, portable
- No server
- Embed in anything
- Understand the entire codebase in 30 minutes

kore loses when you need:
- Enterprise multi-tenant
- Production cloud hosting
- Complex agent workflows
- 50+ integrations out of the box

## Philosophy

> Every great infrastructure tool started by being the simplest
> correct solution to a real problem.
>
> SQLite didn't compete with Oracle. It competed with `fopen()`.
> kore doesn't compete with LangChain. It competes with `json.dump()`.
