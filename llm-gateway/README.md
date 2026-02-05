# LLM Gateway & Routing

A production-grade LLM gateway that provides unified access to multiple LLM providers with intelligent routing, failover, rate limiting, and semantic caching.

## Features

- **Multi-Provider Support** - OpenAI, Anthropic (extensible to others)
- **Intelligent Routing** - Cost-based, latency-based, round-robin, fallback chain
- **Resilience Patterns** - Circuit breaker, retry with exponential backoff
- **Rate Limiting** - Token bucket algorithm with per-key limits
- **Semantic Caching** - Cache responses based on semantic similarity
- **Streaming Support** - Server-sent events for real-time responses
- **Observability** - Structured logging, metrics, health checks

## Quick Start

### Prerequisites

- Python 3.11+
- Redis (optional, for distributed caching/rate limiting)
- API keys for OpenAI and/or Anthropic

### Installation

```bash
# Using make
make setup

# Or manually
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Running

```bash
# Development server
make run

# Or with Docker
make docker-up
```

### Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run linting
make lint
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         API Layer                           │
│                    (FastAPI Endpoints)                      │
├─────────────────────────────────────────────────────────────┤
│                        Gateway                              │
│         (Orchestrates request lifecycle)                    │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│   Cache  │  Router  │ Circuit  │   Rate   │   Providers    │
│          │          │ Breaker  │ Limiter  │                │
├──────────┴──────────┴──────────┴──────────┼────────────────┤
│                                           │   OpenAI       │
│            Resilience Layer               │   Anthropic    │
│                                           │   (more...)    │
└───────────────────────────────────────────┴────────────────┘
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/completions` | Create a completion |
| POST | `/v1/completions/stream` | Create a streaming completion |
| GET | `/v1/providers` | List available providers |
| GET | `/v1/providers/{id}/health` | Get provider health status |
| GET | `/v1/cache/stats` | Get cache statistics |
| GET | `/health` | Health check endpoint |

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `REDIS_URL` | - | Redis URL for distributed features |
| `DEFAULT_ROUTING_STRATEGY` | `fallback` | Default routing strategy |
| `CACHE_ENABLED` | `true` | Enable semantic caching |
| `CACHE_SIMILARITY_THRESHOLD` | `0.95` | Similarity threshold for cache hits |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | `60` | Default rate limit |

## Usage Examples

### Basic Completion

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/v1/completions",
        json={
            "messages": [
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "model": "gpt-4",
        },
    )
    print(response.json())
```

### With Routing Hints

```python
response = await client.post(
    "http://localhost:8000/v1/completions",
    json={
        "messages": [{"role": "user", "content": "Hello!"}],
        "routing": {
            "strategy": "cost",
            "preferred_providers": ["anthropic"],
            "exclude_providers": ["openai"],
        },
    },
)
```

### Streaming Response

```python
async with httpx.AsyncClient() as client:
    async with client.stream(
        "POST",
        "http://localhost:8000/v1/completions/stream",
        json={"messages": [{"role": "user", "content": "Tell me a story"}]},
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                print(line[6:])
```

## Development

### Project Structure

```
llm-gateway/
├── src/llm_gateway/
│   ├── providers/      # LLM provider implementations
│   ├── routing/        # Routing strategies and logic
│   ├── resilience/     # Circuit breaker, retry, rate limiting
│   ├── cache/          # Semantic caching
│   ├── api/            # FastAPI routes and middleware
│   └── gateway.py      # Main orchestrator
├── tests/
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
└── docker-compose.yml
```

### Adding a New Provider

1. Create a new file in `providers/` (e.g., `azure.py`)
2. Implement the `LLMProvider` abstract base class
3. Register in `providers/factory.py`
4. Add configuration in `config.py`

## License

MIT
