"""Optional HTTP layer.

Kept out of the core so the library has no runtime dependency on a web stack.
Run it with ``citekit serve ./data/docs`` after ``pip install "citekit[server]"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .pipeline import CiteKit

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import FastAPI


def create_app(kit: CiteKit) -> "FastAPI":
    """Build a FastAPI app around an already-indexed :class:`CiteKit`."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover - optional path
        raise RuntimeError('install the server extra: pip install "citekit[server]"') from exc

    class AskRequest(BaseModel):
        question: str
        top_k: int = 5

    app = FastAPI(title="CiteKit", version="0.1.0", description="Cited answers or a refusal.")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", **kit.stats()}

    @app.get("/search")
    def search(q: str, top_k: int = 5) -> dict[str, object]:
        if not q.strip():
            raise HTTPException(status_code=400, detail="q is required")
        return {"results": [
            {"chunk_id": c.id, "score": c.score, "text": c.text, "source": c.chunk.source}
            for c in kit.retrieve(q, top_k=top_k)
        ]}

    @app.post("/ask")
    def ask(payload: AskRequest) -> dict[str, object]:
        result = kit.ask(payload.question, top_k=payload.top_k)
        return result.as_dict()

    return app


def serve(kit: CiteKit, *, host: str = "127.0.0.1", port: int = 8000) -> int:
    """Run uvicorn against the app."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional path
        raise RuntimeError('install the server extra: pip install "citekit[server]"') from exc
    uvicorn.run(create_app(kit), host=host, port=port)
    return 0