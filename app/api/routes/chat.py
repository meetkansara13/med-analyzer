from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from app.core.dependencies import rag, tracker
from app.mlops.tracker import LatencyTimer
import re

router = APIRouter(prefix="/chat", tags=["Medical Q&A"])

class ChatRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: float
    latency_ms: float

EXAMPLES = [
    "What is the patient's diagnosis?",
    "What medications are prescribed?",
    "What are the lab results?",
    "Are there any critical values?",
    "What follow-up is recommended?",
    "What are the patient's symptoms?",
]

MEDICAL_QUESTION_KEYWORDS = {
    "diagnosis", "disease", "symptom", "symptoms", "medicine", "medication",
    "tablet", "dose", "prescription", "lab", "report", "blood", "scan",
    "xray", "x-ray", "mri", "ct", "doctor", "patient", "hospital", "pain",
    "fever", "troponin", "bp", "sugar", "glucose", "cholesterol", "treatment",
    "discharge", "allergy", "infection", "health", "medical", "clinical",
    "test", "tests", "procedure", "procedures", "performed", "results",
    "finding", "findings", "critical", "urgent", "follow-up", "follow", "care"
}

NON_MEDICAL_QUESTION_KEYWORDS = {
    "loan", "mortgage", "bank", "salary", "property", "rent", "invoice", "gst",
    "tax", "visa", "passport", "agreement", "shopping", "movie", "football"
}

MEDICAL_CHAT_REDIRECT = (
    "This question does not seem medical. Please ask about a medical report, symptoms, "
    "diagnosis, medicines, lab results, treatment, or follow-up care."
)

def looks_medical_question(question: str) -> bool:
    question_l = question.lower()
    medical_hits = sum(1 for kw in MEDICAL_QUESTION_KEYWORDS if kw in question_l)
    non_medical_hits = sum(1 for kw in NON_MEDICAL_QUESTION_KEYWORDS if kw in question_l)
    token_hits = len(re.findall(r"[a-zA-Z]+", question_l))
    if token_hits <= 2:
        return True
    if any(
        phrase in question_l for phrase in [
            "in this report", "from this report", "according to this report",
            "what tests", "what procedures", "important results", "critical findings",
            "urgent findings", "follow-up", "what was performed"
        ]
    ):
        return True
    return medical_hits > 0 and medical_hits >= non_medical_hits

@router.post("/query", response_model=ChatResponse)
async def chat_query(request: ChatRequest):
    timer = LatencyTimer()
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if not looks_medical_question(request.question):
        raise HTTPException(status_code=422, detail=MEDICAL_CHAT_REDIRECT)
    result = rag.query(question=request.question, doc_id=request.doc_id)
    latency = timer.elapsed_ms()
    tracker.track_query(request.question, len(result["answer"].split()), result["confidence"], len(result["sources"]), latency)
    return ChatResponse(answer=result["answer"], sources=result["sources"], confidence=result["confidence"], latency_ms=latency)

@router.get("/examples")
async def get_examples():
    return {"examples": EXAMPLES}
