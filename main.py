"""
Research Summarizer Agent
FastAPI app with /health and /chat endpoints.
Powered by LangChain + OpenAI with Serper search and page-reading tools.
"""

import os
import json
import logging
import httpx
import requests
import trafilatura

from typing import Any
from datetime import datetime

from dotenv import load_dotenv
from bs4 import BeautifulSoup

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("research-agent")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_base_url: str = Field("https://api.openai.com/v1", env="OPENAI_BASE_URL")
    openai_model: str = Field("gpt-4o", env="OPENAI_MODEL")
    serper_api_key: str = Field(..., env="SERPER_API_KEY")
    max_search_results: int = Field(5, env="MAX_SEARCH_RESULTS")
    max_page_reads: int = Field(3, env="MAX_PAGE_READS")
    agent_timeout: int = Field(120, env="AGENT_TIMEOUT")

    class Config:
        env_file = ".env"


settings = Settings()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def web_search(query: str) -> str:
    """
    Search the web using Serper API.
    Use this to find relevant pages and sources for a research topic.
    Input should be a concise search query string.
    Returns a JSON list of search results with title, link, and snippet.
    """
    logger.info(f"[web_search] Query: {query!r}")
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": settings.serper_api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "num": settings.max_search_results,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.error(f"[web_search] Request failed: {exc}")
        return json.dumps({"error": str(exc)})

    results = []
    for item in data.get("organic", []):
        results.append(
            {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "position": item.get("position"),
            }
        )

    # Also surface knowledge graph snippet if present
    kg = data.get("knowledgeGraph", {})
    if kg.get("description"):
        results.insert(
            0,
            {
                "title": kg.get("title", "Knowledge Graph"),
                "link": kg.get("website", ""),
                "snippet": kg.get("description", ""),
                "position": 0,
            },
        )

    logger.info(f"[web_search] Returning {len(results)} results")
    return json.dumps(results, ensure_ascii=False)


@tool
def read_page(url: str) -> str:
    """
    Fetch and extract the main readable text content from a web page URL.
    Use this after web_search to read the actual content of a promising page.
    Input must be a full URL (including https://).
    Returns the extracted plain text content of the page (up to ~4000 chars).
    """
    logger.info(f"[read_page] Fetching: {url!r}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; ResearchBot/1.0; +https://example.com)"
        )
    }

    try:
        resp = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        html = resp.text
    except Exception as exc:
        logger.warning(f"[read_page] httpx failed for {url}: {exc}")
        return json.dumps({"error": f"Could not fetch page: {exc}"})

    # 1. Try trafilatura first (best main-content extraction)
    content = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )

    # 2. Fallback to BeautifulSoup if trafilatura returns nothing
    if not content:
        logger.info("[read_page] trafilatura returned empty, falling back to BS4")
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        content = soup.get_text(separator="\n", strip=True)

    if not content:
        return json.dumps({"error": "Could not extract readable content from page."})

    # Truncate to keep token usage reasonable
    MAX_CHARS = 4000
    if len(content) > MAX_CHARS:
        content = content[:MAX_CHARS] + "\n\n[... content truncated ...]"

    logger.info(f"[read_page] Extracted {len(content)} chars from {url}")
    return json.dumps({"url": url, "content": content}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert research analyst. Your job is to research a given topic thoroughly and produce a well-structured research summary.

Follow this process:
1. **Plan** - Identify 2-3 key sub-questions or angles that need investigation for the topic.
2. **Search** - Use `web_search` with targeted queries to find relevant sources.
3. **Read** - Use `read_page` on the most promising URLs (up to {max_reads} pages) to extract detailed content.
4. **Synthesize** - When you have gathered enough information, write a comprehensive research summary.

Your final response MUST follow this exact structure:

## Research Summary: [Topic]

### Overview
[2-3 sentence high-level summary]

### Key Findings
[Bullet points of the most important facts, insights, and data points]

### Detailed Analysis
[2-4 paragraphs covering the topic in depth, organized by sub-themes]

### Sources
[List each source you read with title and URL]

### Research Confidence
[Brief note on the quality and coverage of sources found]

Be factual, objective, and thorough. If information is conflicting across sources, note the disagreement.
""".format(
    max_reads=settings.max_page_reads
)


def build_agent() -> AgentExecutor:
    """Build and return a LangChain agent with the research tools."""
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0.2,
    )

    tools = [web_search, read_page]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=12,
        handle_parsing_errors=True,
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Research Summarizer Agent",
    description="An AI-powered research agent that searches the web and reads pages to produce structured research summaries.",
    version="1.0.0",
)


# --- Request / Response schemas ---

class ChatRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The research topic or question to investigate.",
        examples=["The impact of large language models on software engineering"],
    )


class ChatResponse(BaseModel):
    topic: str
    summary: str
    duration_seconds: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    timestamp: str


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Health check endpoint. Returns service status and configuration."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        model=settings.openai_model,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Run the research agent on the given topic.

    The agent will:
    1. Search the web using Serper
    2. Read promising page contents
    3. Return a structured research summary
    """
    logger.info(f"[/chat] Received research request for topic: {req.topic!r}")
    start = datetime.utcnow()

    try:
        agent_executor = build_agent()
        result: dict[str, Any] = await agent_executor.ainvoke(
            {"input": f"Research this topic thoroughly: {req.topic}"},
        )
        summary: str = result.get("output", "No summary generated.")
    except Exception as exc:
        logger.error(f"[/chat] Agent error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Agent encountered an error: {str(exc)}",
        )

    duration = (datetime.utcnow() - start).total_seconds()
    logger.info(f"[/chat] Completed in {duration:.2f}s")

    return ChatResponse(
        topic=req.topic,
        summary=summary,
        duration_seconds=round(duration, 2),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
