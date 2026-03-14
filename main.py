import os
from contextlib import asynccontextmanager
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

model: SentenceTransformer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
    yield
    model = None


app = FastAPI(title="Embedding API", lifespan=lifespan)


class EmbedRequest(BaseModel):
    texts: List[str]
    normalize: bool = True


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")
    vectors = model.encode(
        request.texts,
        normalize_embeddings=request.normalize,
        show_progress_bar=False,
    )
    return EmbedResponse(
        embeddings=vectors.tolist(),
        model=EMBEDDING_MODEL,
        dimension=model.get_sentence_embedding_dimension(),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": EMBEDDING_MODEL}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
