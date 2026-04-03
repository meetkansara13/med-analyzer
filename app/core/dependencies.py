from app.core.rag_pipeline import MedicalRAGPipeline
from app.mlops.tracker import MLflowTracker

rag = MedicalRAGPipeline()
tracker = MLflowTracker()
