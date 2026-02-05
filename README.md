# Enterprise AI Infrastructure Concepts

A hands-on learning repository for deploying AI at enterprise scale. Each module is **production-quality**, **independently runnable**, and follows industry best practices.

## 🎯 Philosophy

| Principle | Description |
|-----------|-------------|
| **Production Quality** | Code written as if by a principal engineer at scale |
| **Clean Code** | SOLID principles, proper abstractions, meaningful names |
| **No Dummies** | Every feature is fully functional—no placeholders |
| **Design Patterns** | Appropriate patterns applied where beneficial |
| **Independent Modules** | Each concept is self-contained and testable |

## 📚 What's Inside

See **[CONCEPTS.md](./CONCEPTS.md)** for the full catalog of topics covering:

- **LLM Gateway & Routing** - Multi-provider orchestration with failover
- **RAG Pipelines** - End-to-end retrieval-augmented generation
- **Agent Orchestration** - Multi-agent systems and tool calling
- **Model Serving** - Inference optimization and serving patterns
- **Guardrails & Safety** - Enterprise-grade content filtering
- **And 10+ more topics...**

## 🚀 Getting Started

```bash
# Clone the repository
git clone <repo-url>
cd ai-infra-concepts

# Each module has its own setup
cd <module-name>
make setup    # or docker-compose up
make test     # Run tests
```

## 📁 Structure

```
.
├── shared/              # Shared utilities across modules
├── llm-gateway/         # LLM routing and orchestration
├── rag-pipeline/        # Retrieval-augmented generation
├── agent-framework/     # Agent orchestration
└── ...                  # Additional modules
```

## 🛠️ Tech Stack

- **Python 3.11+** with type hints
- **FastAPI** for APIs
- **PostgreSQL + Redis** for persistence
- **Docker** for containerization
- **pytest** for testing

---

*Start with [CONCEPTS.md](./CONCEPTS.md) to understand the learning path.*
