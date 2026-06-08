# Research Summarizer Agent

A FastAPI-based AI research agent that searches the web and reads page contents to produce structured research summaries. Built with LangChain + OpenAI and powered by Serper for web search.

## Architecture

```
POST /chat  →  LangChain Agent (OpenAI)
                  ├── Tool: web_search   (Serper API → Google results)
                  └── Tool: read_page    (trafilatura + BeautifulSoup)
```

## Requirements

- Python 3.11
- OpenAI-compatible API key (OpenAI, Azure, or any compatible endpoint)
- [Serper API key](https://serper.dev) (free tier: 2,500 searches/month)

## Setup

### 1. Clone / place files

```
research-agent/
├── main.py
├── requirements.txt
└── .env
```

### 2. Create virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with your keys
```

Required `.env` values:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI (or compatible) API key |
| `OPENAI_BASE_URL` | API base URL injected at runtime |
| `OPENAI_MODEL` | Model name (default: `gpt-4o`) |
| `SERPER_API_KEY` | Your Serper API key from serper.dev |

Optional tuning:

| Variable | Default | Description |
|---|---|---|
| `MAX_SEARCH_RESULTS` | `5` | Results returned per search query |
| `MAX_PAGE_READS` | `3` | Max pages the agent will read |
| `AGENT_TIMEOUT` | `120` | Max seconds per request |

### 5. Run

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Reference

### `GET /health`

Returns service status.

```json
{
  "status": "ok",
  "version": "1.0.0",
  "model": "gpt-4o",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### `POST /chat`

Run a research query.

**Request:**
```json
{
  "topic": "The impact of quantum computing on cryptography"
}
```

**Response:**
```json
{
  "topic": "The impact of quantum computing on cryptography",
  "summary": "## Research Summary: ...\n\n### Overview\n...",
  "duration_seconds": 28.4,
  "timestamp": "2024-01-15T10:30:28Z"
}
```

### Interactive docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Example curl

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"topic": "Recent advances in fusion energy"}'
```

## Agent Behavior

The agent follows a structured research loop:

1. **Plan** — Identifies key sub-questions from the topic
2. **Search** — Issues 2-3 targeted `web_search` calls via Serper
3. **Read** — Calls `read_page` on the most relevant URLs (up to `MAX_PAGE_READS`)
4. **Synthesize** — Produces a structured markdown summary with overview, key findings, analysis, and sources

The final summary always includes:
- Overview
- Key Findings (bullets)
- Detailed Analysis (paragraphs)
- Sources (with URLs)
- Research Confidence note
