from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from loguru import logger

class MedicalSummarizer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        logger.info("Loading summarizer...")
        try:
            model_name = "sshleifer/distilbart-cnn-12-6"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.eval()
            logger.success("Summarizer loaded")
        except Exception as e:
            logger.error(f"Summarizer failed: {e}")
            self.model = None
            self.tokenizer = None

    def summarize(self, text: str, max_length: int = 300, min_length: int = 80, doc_type: str = "general") -> dict:
        if not text.strip():
            return {"summary": "No text provided", "key_findings": [], "recommendations": [], "compression_ratio": 0, "summary_length": 0}

        text_chunk = " ".join(text.split()[:900])

        try:
            if self.model and self.tokenizer:
                inputs = self.tokenizer(
                    text_chunk,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True
                )
                with torch.no_grad():
                    output_ids = self.model.generate(
                        inputs["input_ids"],
                        max_new_tokens=max_length,
                        min_new_tokens=min_length,
                        num_beams=4,
                        length_penalty=2.0,
                        early_stopping=True
                    )
                summary_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            else:
                summary_text = text_chunk[:500] + "..."
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            summary_text = text_chunk[:500] + "..."

        return {
            "summary": summary_text,
            "key_findings": self._extract_findings(summary_text),
            "recommendations": self._extract_recommendations(summary_text),
            "doc_type": doc_type,
            "original_length": len(text.split()),
            "summary_length": len(summary_text.split()),
            "compression_ratio": round(len(summary_text.split()) / max(len(text.split()), 1), 2)
        }

    def _extract_findings(self, summary: str) -> list:
        keywords = ["shows", "reveals", "indicates", "found", "detected", "elevated", "low", "normal", "abnormal"]
        return [s.strip() + "." for s in summary.split(".") if len(s.strip()) > 20 and any(k in s.lower() for k in keywords)][:5]

    def _extract_recommendations(self, summary: str) -> list:
        keywords = ["recommend", "suggest", "should", "advised", "follow-up", "prescribe", "monitor"]
        return [s.strip() + "." for s in summary.split(".") if len(s.strip()) > 20 and any(k in s.lower() for k in keywords)][:3]