"""
Retail Sales & Inventory Copilot — entrypoint.

Starts the whole application (API + frontend) on port 8000, as required by
the hackathon submission rules:  `python app.py`  and nothing else.
"""
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.analytics import Catalogue
from src.llm import answer_question

APP_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = APP_DIR / "frontend" / "dist"

app = FastAPI(title="Retail Sales & Inventory Copilot")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Load the dataset once at startup — this is the "index" for this problem
# statement; there is no embedding step needed since the data is small,
# structured, and queried deterministically rather than retrieved by
# similarity search.
catalogue = Catalogue()


class QuestionRequest(BaseModel):
    question: str
    store_id: Optional[str] = None


@app.get("/api/health")
def health():
    return {"status": "ok", "as_of": catalogue.as_of.date().isoformat(),
            "products": len(catalogue.products), "stores": len(catalogue.stores)}


@app.get("/api/stores")
def list_stores():
    return catalogue.stores.to_dict(orient="records")


@app.get("/api/briefing")
def briefing(store_id: Optional[str] = None):
    """What needs attention today: stock-out risks, slow movers, sales moves."""
    if store_id and store_id not in catalogue.stores["store_id"].values:
        raise HTTPException(status_code=404, detail=f"Unknown store_id '{store_id}'")
    return catalogue.daily_briefing(store_id)


@app.post("/api/ask")
def ask(req: QuestionRequest):
    """Plain-language Q&A grounded in the store's actual data."""
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    if req.store_id and req.store_id not in catalogue.stores["store_id"].values:
        raise HTTPException(status_code=404, detail=f"Unknown store_id '{req.store_id}'")
    try:
        result = answer_question(req.question.strip(), catalogue)
    except Exception as exc:
        # A model or data error should never surface as a 500 with no
        # explanation — the manager still needs a usable response.
        return {
            "answer": "Something went wrong answering that — please rephrase or try a narrower question "
                      "(e.g. name a specific product or store).",
            "grounding": {},
            "llm_used": False,
            "error": str(exc),
        }
    return result


# --- Serve the built frontend -------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets") \
        if (FRONTEND_DIR / "assets").exists() else None

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/{path:path}")
    def spa_catch_all(path: str):
        candidate = FRONTEND_DIR / path
        if candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
