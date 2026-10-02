"""FastAPI: /search, /answer, /sign/{code} + trang chat tĩnh.

uvicorn vnlaw_rag.api:app --port 8000
"""

import os
from dataclasses import asdict
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .pipeline import LawRAG

app = FastAPI(title="vn-traffic-law-rag")
_rag: LawRAG | None = None
STATIC = Path(__file__).parent / "static"


def rag() -> LawRAG:
    global _rag
    if _rag is None:
        _rag = LawRAG.from_dir(os.getenv("VNLAW_DATA", "data"))
    return _rag


class Query(BaseModel):
    q: str
    k: int = 5
    as_of: date | None = None


def _hit(h) -> dict:
    return {"id": h.id, "citation": h.citation, "score": round(h.score, 4), "text": h.chunk["text"],
            "expanded_from": h.expanded_from}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    r = rag().retriever
    return {"chunks": len(r.chunks), "dense": r.dense is not None, "reranker": r.reranker is not None}


@app.post("/search")
def search(body: Query):
    hits = rag().search(body.q, k=body.k, as_of=body.as_of)
    return {"query": body.q, "debug": hits[0].debug if hits else {}, "hits": [_hit(h) for h in hits]}


@app.post("/answer")
def answer(body: Query):
    return asdict(rag().answer(body.q, k=body.k, as_of=body.as_of))


@app.get("/sign/{code}")
def sign(code: str, vehicle: str | None = None):
    info = rag().lookup_sign(code, vehicle)
    if info is None:
        raise HTTPException(404, f"chưa có biển {code} trong sign_map.json")
    return {"code": info.code, "name": info.name, "query": info.query, "hits": [_hit(h) for h in info.hits]}
