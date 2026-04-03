import mlflow
from loguru import logger
from typing import Dict
from datetime import datetime
import time

class MLflowTracker:
    def __init__(self):
        from app.core.config import get_settings
        settings = get_settings()
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT)
            logger.success(f"MLflow connected: {settings.MLFLOW_TRACKING_URI}")
        except Exception as e:
            logger.warning(f"MLflow unavailable (offline mode): {e}")

    def track_analysis(self, doc_id, doc_type, text_length, entities, summary_result, latency_ms):
        try:
            with mlflow.start_run(run_name=f"analysis_{doc_id}_{datetime.now().strftime('%H%M%S')}"):
                mlflow.set_tags({"doc_id": doc_id, "doc_type": doc_type})
                mlflow.log_metrics({
                    "text_length_words": text_length,
                    "latency_ms": latency_ms,
                    "total_entities": sum(len(v) for v in entities.values()),
                    "diseases_found": len(entities.get("diseases", [])),
                    "drugs_found": len(entities.get("drugs", [])),
                    "symptoms_found": len(entities.get("symptoms", [])),
                    "compression_ratio": summary_result.get("compression_ratio", 0),
                })
        except Exception as e:
            logger.warning(f"MLflow tracking skipped: {e}")

    def track_query(self, question, answer_length, confidence, sources_count, latency_ms):
        try:
            with mlflow.start_run(run_name=f"query_{datetime.now().strftime('%H%M%S')}"):
                mlflow.set_tags({"query_type": "rag_qa"})
                mlflow.log_metrics({
                    "question_length": len(question.split()),
                    "answer_length": answer_length,
                    "confidence": confidence,
                    "sources_retrieved": sources_count,
                    "latency_ms": latency_ms,
                })
        except Exception as e:
            logger.warning(f"MLflow query tracking skipped: {e}")

class LatencyTimer:
    def __init__(self):
        self.start = time.time()
    def elapsed_ms(self) -> float:
        return round((time.time() - self.start) * 1000, 2)
