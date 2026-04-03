from transformers import pipeline
import torch
from loguru import logger
from typing import List, Dict


class MedicalNERPipeline:
    def __init__(self):
        self.pipeline = None
        # ✅ Removed _load_model() from __init__ — model loads on first use

    def _load_model(self):
        if self.pipeline is not None:
            return  # already loaded
        from app.core.config import get_settings
        settings = get_settings()
        logger.info(f"Loading NER model: {settings.NER_MODEL}")
        try:
            device = 0 if torch.cuda.is_available() else -1
            self.pipeline = pipeline(
                task="ner",
                model=settings.NER_MODEL,
                aggregation_strategy="simple",
                device=device
            )
            logger.success("NER model loaded")
        except Exception as e:
            logger.warning(f"Medical NER failed, using fallback: {e}")
            self.pipeline = pipeline(
                task="ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple",
                device=-1
            )

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        self._load_model()  # ✅ lazy load here
        if not text.strip():
            return {}
        chunks = self._chunk_text(text)
        all_entities = []
        for chunk in chunks:
            try:
                all_entities.extend(self.pipeline(chunk))
            except Exception as e:
                logger.warning(f"NER chunk failed: {e}")
        return self._categorize_entities(all_entities)

    def _chunk_text(self, text: str, max_length: int = 400) -> List[str]:
        words = text.split()
        chunks, current = [], []
        for word in words:
            current.append(word)
            if len(current) >= max_length:
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))
        return chunks or [text]

    def _categorize_entities(self, entities: List[Dict]) -> Dict[str, List[str]]:
        categorized = {
            "diseases": [], "drugs": [], "symptoms": [],
            "anatomy": [], "procedures": [], "other": []
        }
        medical_map = {
            "DISEASE": "diseases", "DRUG": "drugs", "CHEMICAL": "drugs",
            "SYMPTOM": "symptoms", "BODY_PART": "anatomy", "ANATOMY": "anatomy",
            "PROCEDURE": "procedures", "TREATMENT": "procedures",
        }
        seen = set()
        for entity in entities:
            word = entity.get("word", "").strip()
            label = entity.get("entity_group", "").upper()
            if not word or len(word) < 2 or word in seen:
                continue
            seen.add(word)
            if entity.get("score", 0) < 0.75:
                continue
            categorized[medical_map.get(label, "other")].append(word)
        return {k: v for k, v in categorized.items() if v}

    def get_entity_stats(self, entities: Dict[str, List[str]]) -> Dict[str, int]:
        return {k: len(v) for k, v in entities.items()}