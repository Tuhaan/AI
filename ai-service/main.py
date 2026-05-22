
#fastapi server entry point is main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from recommendation_engine import RecommendationEngine

load_dotenv()

app = FastAPI(
    title="TorchBearer Recommendation Service",
    description="AI-powered space recommendations using TF-IDF and Cosine Similarity",
    version="1.0.0",
)

# CORS — allow Express backend and frontend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("EXPRESS_BACKEND_URL", "http://localhost:3000"),
        os.getenv("FRONTEND_URL", "http://localhost:5173"),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise the recommendation engine (connects to MongoDB)
engine = RecommendationEngine(
    mongo_uri=os.getenv("MONGO_URI", ""),
    db_name=os.getenv("MONGO_DB_NAME", "torchbearer"),
)


# ── Request / Response models

class RecommendationRequest(BaseModel):
    
    #Accepts either skills (alumni) or interests (students).At least one must be provided.
    
    skills: Optional[List[str]] = None
    interests: Optional[List[str]] = None
    top_n: int = 5


class RecommendedSpace(BaseModel):
    id: str
    title: str
    description: str = ""          # default empty string — never null
    tags: List[str] = []
    similarity_score: float
    created_by: Optional[str] = None


class RecommendationResponse(BaseModel):
    success: bool
    query_terms: List[str]
    recommendations: List[RecommendedSpace]
    total_spaces_analyzed: int


# Routes 

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "TorchBearer AI Recommendation Service"}


@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    
    # Merge skills and interests into a single query list
    query_terms: List[str] = []
    if request.skills:
        query_terms.extend([ #combines all user preferances
            skill.strip()
            for skill in request.skills
            if skill.strip()
        ])
    if request.interests:
        query_terms.extend([
            interest.strip()
            for interest in request.interests
            if interest.strip()
        ]) #strip removes extra spaces

    query_terms = list(set(query_terms)) #avoids repeated keywords

    if not query_terms:
        raise HTTPException(
            status_code=400,
            detail="At least one skill or interest must be provided.",
        )

    try:
        results, total = engine.recommend(
            query_terms=query_terms,
            top_n=request.top_n,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation engine error: {str(exc)}",
        )

    # Sanitise results — ensure no null values None values slip through Pydantic validation
    safe_results = []
    for r in results:
        safe_results.append(RecommendedSpace(
            id=str(r.get("id", "")),
            title=r.get("title") or "Untitled",
            description=r.get("description") or "",
            tags=[t for t in (r.get("tags") or []) if t],
            similarity_score=float(r.get("similarity_score", 0)),
            created_by=r.get("created_by") or None,
        ))

    return RecommendationResponse(
        success=True,
        query_terms=query_terms,
        recommendations=safe_results,
        total_spaces_analyzed=total,
    )
