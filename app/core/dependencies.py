from functools import lru_cache
from app.core.rag_pipeline import MedicalRAGPipeline
from app.mlops.tracker import MLflowTracker

# ✅ lru_cache means each class is only instantiated once (on first request)
# and never at import time — fixes the Render port-scan timeout

@lru_cache()
def get_rag() -> MedicalRAGPipeline:
    return MedicalRAGPipeline()

@lru_cache()
def get_tracker() -> MLflowTracker:
    return MLflowTracker()