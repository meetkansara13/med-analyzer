from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid, io
from loguru import logger
from app.core.ner_pipeline import MedicalNERPipeline
from app.core.summarizer import MedicalSummarizer
from app.core.dependencies import rag, tracker
from app.mlops.tracker import LatencyTimer
import re

router = APIRouter(prefix="/analyze", tags=["Document Analysis"])

ner = MedicalNERPipeline()
summarizer = MedicalSummarizer()

MEDICAL_KEYWORDS = {
    "patient", "diagnosis", "symptom", "symptoms", "treatment", "prescribed",
    "medicine", "medication", "tablet", "capsule", "dose", "mg", "hospital",
    "doctor", "discharge", "lab", "blood", "ecg", "troponin", "x-ray", "mri",
    "prescription", "history", "examination", "admission", "allergy", "clinic",
    "disease", "hypertension", "diabetes", "chest pain", "fever", "pulse",
    "bp", "heart rate", "report", "medical", "clinical", "findings"
}

NON_MEDICAL_KEYWORDS = {
    "loan", "home loan", "mortgage", "interest rate", "emi", "bank", "salary",
    "income", "property", "invoice", "gst", "tax", "passport", "visa",
    "agreement", "rent", "purchase order", "quotation", "credit card"
}

MEDICAL_REJECTION_MESSAGE = (
    "This document does not seem to be medical. Please upload a medical report, "
    "prescription, discharge summary, lab result, scan report, or another clinical document. "
    "Feel free to ask anything related to medical reports or health information."
)

def looks_medical(text: str) -> bool:
    text_l = text.lower()
    medical_hits = sum(1 for kw in MEDICAL_KEYWORDS if kw in text_l)
    non_medical_hits = sum(1 for kw in NON_MEDICAL_KEYWORDS if kw in text_l)

    number_hits = len(re.findall(r"\b\d+(?:\.\d+)?\s?(?:mg|ml|mmhg|bpm|%|ng/ml|g/dl)\b", text_l))
    section_hits = sum(
        1 for kw in [
            "diagnosis", "medications", "prescription", "discharge", "physical examination",
            "chief complaint", "history of present illness", "lab results", "impression"
        ]
        if kw in text_l
    )

    score = medical_hits + number_hits + section_hits - (non_medical_hits * 2)
    return score >= 4 and medical_hits >= 2

class TextAnalysisRequest(BaseModel):
    text: str
    doc_type: str = "general"
    doc_id: Optional[str] = None

class AnalysisResponse(BaseModel):
    doc_id: str
    entities: dict
    entity_stats: dict
    summary: dict
    status: str
    latency_ms: float

@router.post("/text", response_model=AnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    timer = LatencyTimer()
    doc_id = request.doc_id or str(uuid.uuid4())[:8]
    if len(request.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Text too short to analyze")
    if not looks_medical(request.text):
        raise HTTPException(
            status_code=422,
            detail=MEDICAL_REJECTION_MESSAGE
        )
    entities = ner.extract_entities(request.text)
    entity_stats = ner.get_entity_stats(entities)
    summary = summarizer.summarize(request.text, doc_type=request.doc_type)
    rag.ingest_document(text=request.text, doc_id=doc_id, metadata={"doc_type": request.doc_type})
    latency = timer.elapsed_ms()
    tracker.track_analysis(doc_id, request.doc_type, len(request.text.split()), entities, summary, latency)
    return AnalysisResponse(doc_id=doc_id, entities=entities, entity_stats=entity_stats, summary=summary, status="success", latency_ms=latency)

@router.post("/upload")
async def analyze_uploaded_file(file: UploadFile = File(...), doc_type: str = "general"):
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files supported")
    content = await file.read()
    if file.filename.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            text = "".join([
                (page.extract_text() or "") + "\n"
                for page in reader.pages
            ]).strip()
            logger.info(f"Extracted {len(text)} chars from PDF ({len(reader.pages)} pages)")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {e}")
    else:
        text = content.decode("utf-8", errors="ignore")
    if not text or len(text.strip()) < 20:
        raise HTTPException(status_code=422, detail=f"Could not extract text from file (got {len(text)} chars). The PDF may be image-based or corrupted.")
    req = TextAnalysisRequest(text=text, doc_type=doc_type, doc_id=file.filename.replace(".", "_"))
    return await analyze_text(req)

@router.delete("/clear")
async def clear_documents():
    rag.clear_documents()
    return {"status": "cleared"}
