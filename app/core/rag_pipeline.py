from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.llms import LLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from loguru import logger
from typing import Dict, Optional, Any
import re
from collections import OrderedDict

MEDICAL_QA_PROMPT = PromptTemplate(
    template="""You are an expert medical AI assistant. Answer only from the provided context.

Context:
{context}

Question: {question}

Rules:
- Give a direct medical answer in 1 to 4 sentences.
- Prefer diagnosis names, medication names, abnormal results, and recommendations exactly as written in the context.
- Do not answer with vague fragments like "illness" or "normal".
- If the context does not contain the answer, say "This information is not available in the document."

Answer:""",
    input_variables=["context", "question"]
)


class FlanT5LLM(LLM):
    """Custom LangChain LLM wrapper around flan-t5."""
    model: Any = None
    tokenizer: Any = None
    max_new_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "flan-t5"

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=self.max_new_tokens,
                num_beams=4,
                early_stopping=True
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


class MedicalRAGPipeline:
    def __init__(self):
        from app.core.config import get_settings
        self.settings = get_settings()
        self.embeddings = None
        self.vectorstore = None
        self.chain = None
        self.retriever = None
        self.llm = None
        # ✅ Removed _init_embeddings() and _init_llm() from __init__
        #    Models load on first use instead

    def _init_embeddings(self):
        if self.embeddings is not None:
            return  # already loaded
        logger.info("Initializing embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.EMBEDDINGS_MODEL,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.success("Embeddings ready")

    def _init_llm(self):
        if self.llm is not None:
            return  # already loaded
        logger.info("Initializing LLM...")
        try:
            model_name = "google/flan-t5-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.eval()
            self.llm = FlanT5LLM(model=model, tokenizer=tokenizer)
            logger.success("LLM ready")
        except Exception as e:
            logger.error(f"LLM load failed: {e}")
            self.llm = None

    def ingest_document(self, text: str, doc_id: str, metadata: dict = {}) -> int:
        self._init_embeddings()  # ✅ lazy load
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
        metadatas = [{**metadata, "doc_id": doc_id, "chunk": i} for i in range(len(chunks))]
        store = self._get_vectorstore()
        store.add_texts(texts=chunks, metadatas=metadatas)
        self._build_chain()
        logger.info(f"Ingested {len(chunks)} chunks for doc: {doc_id}")
        return len(chunks)

    def _get_vectorstore(self):
        self._init_embeddings()  # ✅ lazy load
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                collection_name=self.settings.CHROMA_COLLECTION,
                embedding_function=self.embeddings,
                persist_directory=self.settings.CHROMA_PERSIST_DIR
            )
        return self.vectorstore

    def _build_chain(self):
        self._init_llm()  # ✅ lazy load
        if self._get_vectorstore() and self.llm:
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 10}
            )
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            self.chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | MEDICAL_QA_PROMPT
                | self.llm
                | StrOutputParser()
            )
            self.retriever = retriever

    def _search_docs(self, question: str, doc_id: Optional[str] = None, k: int = 4):
        store = self._get_vectorstore()
        if doc_id:
            docs = store.similarity_search(question, k=max(k * 2, 8), filter={"doc_id": doc_id})
        else:
            docs = store.similarity_search(question, k=max(k * 2, 8))
        return self._rerank_docs(question, docs, limit=k)

    def _question_keywords(self, question: str):
        question_l = question.lower()
        keywords = set(re.findall(r"[a-zA-Z0-9\-]+", question_l))
        boosted = set()

        if any(x in question_l for x in ["disease", "diagnosis", "condition", "problem", "what kind", "main disease"]):
            boosted.update(["diagnosis", "diagnosed", "impression", "assessment", "nstemi", "stemi", "acs", "disease", "condition"])
        if any(x in question_l for x in ["symptom", "complaint", "presenting"]):
            boosted.update(["symptom", "complaint", "presented", "history", "pain", "fever", "cough", "breath", "palpitations", "syncope"])
        if any(x in question_l for x in ["medication", "medicine", "drug", "prescribed", "treatment"]):
            boosted.update(["mg", "tablet", "capsule", "po", "iv", "daily", "medication", "prescribed", "treatment"])
        if any(x in question_l for x in ["lab", "value", "troponin", "result"]):
            boosted.update(["troponin", "creatinine", "glucose", "hb", "a1c", "wbc", "platelet", "elevated", "low", "high"])
        if any(x in question_l for x in ["follow-up", "follow up", "recommend", "plan", "discharge"]):
            boosted.update(["follow-up", "follow", "review", "repeat", "monitor", "recommended", "advised", "discharge", "plan"])
        if any(x in question_l for x in ["test", "tests", "procedure", "procedures", "result", "results"]):
            boosted.update(["test", "tests", "procedure", "procedures", "ecg", "echo", "mri", "ct", "scan", "x-ray", "result", "impression"])
        if any(x in question_l for x in ["critical", "urgent", "emergency", "serious"]):
            boosted.update(["critical", "urgent", "emergency", "severe", "elevated", "abnormal", "impression", "finding"])

        return keywords | boosted

    def _is_boilerplate_line(self, text: str) -> bool:
        text_l = (text or "").lower().strip()
        if not text_l:
            return True
        boilerplate_patterns = [
            "general hospital", "medical center", "research centre",
            "healthcare avenue", "www.", "tel:", "fax:", "department of",
            "patient discharge summary patient", "contact", "consultation",
            "gender", "policy no.", "policy no", "section 1", "section 2",
            "section 10", "star health", "patient information",
            "vital signs & anthropometric measurements",
        ]
        return any(pattern in text_l for pattern in boilerplate_patterns)

    def _clean_answer_text(self, answer: str) -> str:
        answer = re.sub(r"\s+", " ", (answer or "")).strip()
        answer = re.sub(r"^[•\-\s]+", "", answer)
        return answer

    def _looks_low_value(self, answer: str) -> bool:
        answer_l = self._clean_answer_text(answer).lower()
        if not answer_l:
            return True
        if len(answer_l.split()) < 5:
            return True
        low_value_patterns = [
            "section 1 patient information", "section 10 follow-up plan",
            "star health", "policy no", "general hospital",
            "medical center", "contact", "consultation",
        ]
        return any(pattern in answer_l for pattern in low_value_patterns)

    def _direct_question_type(self, question: str) -> str:
        q = (question or "").lower()
        if any(x in q for x in ["diagnosis", "diagnoses", "condition", "conditions", "what does the report say about"]):
            return "diagnosis"
        if any(x in q for x in ["medicine", "medicines", "drug", "drugs", "prescribed"]):
            return "medicines"
        if any(x in q for x in ["lab", "labs", "blood sugar", "troponin", "glucose", "creatinine", "hba1c", "test result", "abnormal result"]):
            return "labs"
        if any(x in q for x in ["follow-up", "follow up", "next step", "review", "monitoring"]):
            return "followup"
        if any(x in q for x in ["symptom", "symptoms", "complaint", "complaints", "fatigue"]):
            return "symptoms"
        if any(x in q for x in ["critical", "urgent", "emergency"]):
            return "critical"
        if any(x in q for x in ["test", "tests", "procedure", "procedures", "result", "results"]):
            return "procedures"
        if any(x in q for x in ["discharge", "treatment plan", "plan", "diet", "rehab"]):
            return "discharge"
        return ""

    def _collect_clean_lines(self, docs):
        lines = []
        for doc in docs:
            for raw_line in doc.page_content.splitlines():
                line = self._clean_answer_text(raw_line)
                if line and not self._is_boilerplate_line(line):
                    lines.append(line)
        return list(OrderedDict.fromkeys(lines))

    def _join_answer_lines(self, lines, limit=4):
        chosen = [self._clean_answer_text(line) for line in lines if self._clean_answer_text(line)]
        chosen = list(OrderedDict.fromkeys(chosen))
        return " ".join(chosen[:limit])

    def _direct_answer_from_docs(self, question: str, docs):
        qtype = self._direct_question_type(question)
        if not qtype:
            return ""
        lines = self._collect_clean_lines(docs)
        if not lines:
            return ""
        selected = []
        for line in lines:
            line_l = line.lower()
            if qtype == "diagnosis":
                if any(x in line_l for x in ["diagnosis", "final diagnosis", "impression", "assessment", "hypertension", "diabetes", "hypothyroidism", "nstemi", "stemi", "acs"]):
                    selected.append(line)
            elif qtype == "medicines":
                if re.search(r"\b\d+(?:\.\d+)?\s?(mg|mcg|ml)\b", line_l) or any(x in line_l for x in ["tablet", "tab", "capsule", "po", "once daily", "twice daily", "od", "bd"]):
                    selected.append(line)
            elif qtype == "labs":
                if any(x in line_l for x in ["troponin", "hba1c", "glucose", "creatinine", "tsh", "cholesterol", "platelet", "hemoglobin", "hb"]) or re.search(r"\b\d+(?:\.\d+)?\s?(mg/dl|g/dl|ng/ml|mmhg|%)\b", line_l):
                    selected.append(line)
            elif qtype == "followup":
                if any(x in line_l for x in ["follow-up", "follow up", "review", "monitor", "repeat", "after 7 days", "in 2 weeks", "advised"]):
                    selected.append(line)
            elif qtype == "symptoms":
                if any(x in line_l for x in ["pain", "fatigue", "fever", "cough", "breath", "trembling", "loss of consciousness", "weakness", "urination", "thirst"]):
                    selected.append(line)
            elif qtype == "critical":
                if any(x in line_l for x in ["critical", "urgent", "severe", "elevated", "abnormal", "troponin", "hypoglycaemia", "rhabdomyolysis"]):
                    selected.append(line)
            elif qtype == "procedures":
                if any(x in line_l for x in ["ecg", "echo", "x-ray", "mri", "ct", "scan", "showed", "revealed", "impression", "test", "performed"]):
                    selected.append(line)
            elif qtype == "discharge":
                if any(x in line_l for x in ["discharge", "diet", "rehabilitation", "rehab", "monitor", "activity", "lifting", "follow-up"]):
                    selected.append(line)
        selected = list(OrderedDict.fromkeys(selected))
        if not selected:
            return ""
        return self._join_answer_lines(selected, limit=4)

    def _rerank_docs(self, question: str, docs, limit: int = 4):
        if not docs:
            return []
        keywords = self._question_keywords(question)
        scored = []
        for idx, doc in enumerate(docs):
            text = doc.page_content.lower()
            score = 0
            if self._is_boilerplate_line(text):
                score -= 12
            for keyword in keywords:
                if keyword in text:
                    score += 3
            if "diagnosis" in question.lower() or "disease" in question.lower():
                if any(x in text for x in ["diagnosis", "impression", "assessment", "nstemi", "stemi", "acs"]):
                    score += 8
                if any(x in text for x in ["mg po", "once daily", "twice daily"]) and score < 8:
                    score -= 2
            if "symptom" in question.lower() or "complaint" in question.lower():
                if any(x in text for x in ["presented", "complaint", "pain", "fever", "cough", "palpitations", "syncope"]):
                    score += 6
            if "procedure" in question.lower() or "test" in question.lower() or "result" in question.lower():
                if any(x in text for x in ["ecg", "echo", "x-ray", "mri", "ct", "scan", "result", "impression", "performed"]):
                    score += 7
            if "critical" in question.lower() or "urgent" in question.lower():
                if any(x in text for x in ["critical", "urgent", "severe", "elevated", "abnormal", "impression"]):
                    score += 7
            chunk_bonus = 1 / (idx + 1)
            scored.append((score + chunk_bonus, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        unique = OrderedDict()
        for _, doc in scored:
            key = (doc.metadata.get("doc_id"), doc.metadata.get("chunk"), doc.page_content[:120])
            if key not in unique:
                unique[key] = doc
            if len(unique) >= limit:
                break
        return list(unique.values())

    def _extractive_fallback(self, question: str, docs) -> str:
        question_l = question.lower()
        lines = self._collect_clean_lines(docs)
        if not lines:
            return "This information is not available in the document."
        keyword_groups = []
        if any(x in question_l for x in ["disease", "diagnosis", "condition", "problem", "what kind"]):
            keyword_groups.append(["diagnosis", "impression", "assessment", "nstemi", "stemi", "pneumonia", "infection", "diabetes", "hypertension", "acs"])
        if any(x in question_l for x in ["medication", "medicine", "drug", "prescribed"]):
            keyword_groups.append(["mg", "tablet", "capsule", "po", "iv", "daily", "twice"])
        if any(x in question_l for x in ["lab", "value", "troponin", "creatinine", "glucose", "a1c"]):
            keyword_groups.append(["troponin", "creatinine", "glucose", "hb", "a1c", "wbc", "platelet", "elevated", "low", "high"])
        if any(x in question_l for x in ["follow-up", "follow up", "recommend", "next", "test"]):
            keyword_groups.append(["follow-up", "follow up", "review", "repeat", "monitor", "recommended", "advised"])
        if any(x in question_l for x in ["symptom", "complaint", "fever", "pain", "breath"]):
            keyword_groups.append(["pain", "fever", "cough", "breath", "palpitations", "syncope", "complaint"])
        if any(x in question_l for x in ["discharge", "treatment plan", "plan", "rehab", "diet"]):
            keyword_groups.append(["discharge", "instructions", "rehabilitation", "rehab", "diet", "monitor", "activity", "lifting", "follow-up", "medication"])
        if any(x in question_l for x in ["critical", "urgent", "emergency", "finding"]):
            keyword_groups.append(["critical", "urgent", "severe", "elevated", "abnormal", "positive", "impression", "finding", "troponin", "depression"])
        if any(x in question_l for x in ["test", "tests", "procedure", "procedures", "result", "results"]):
            keyword_groups.append(["ecg", "echo", "x-ray", "mri", "ct", "scan", "performed", "result", "results", "impression", "showed", "revealed"])

        matches = []
        for line in lines:
            line_l = line.lower()
            if any(any(keyword in line_l for keyword in group) for group in keyword_groups):
                matches.append(line)
        if not matches:
            matches = lines[:3]

        if any(x in question_l for x in ["disease", "diagnosis", "condition", "problem", "what kind"]):
            for line in matches:
                if any(x in line.lower() for x in ["diagnosis", "impression", "assessment", "nstemi", "stemi", "acs"]):
                    return self._clean_answer_text(line)
        if any(x in question_l for x in ["diagnosis", "condition", "hypertension", "diabetes", "hypothyroidism"]):
            diagnosis_lines = [self._clean_answer_text(line) for line in matches if any(x in line.lower() for x in ["diagnosis", "impression", "assessment", "hypertension", "diabetes", "hypothyroidism", "nstemi", "stemi", "acs"])]
            if diagnosis_lines:
                return self._join_answer_lines(diagnosis_lines, limit=4)
        if any(x in question_l for x in ["medicine", "medicines", "drug", "prescribed"]):
            med_lines = [self._clean_answer_text(line) for line in matches if re.search(r"\b\d+(?:\.\d+)?\s?(mg|mcg|ml)\b", line.lower()) or any(x in line.lower() for x in ["tablet", "tab", "capsule", "po", "once daily", "twice daily", "od", "bd"])]
            if med_lines:
                return self._join_answer_lines(med_lines, limit=6)
        if any(x in question_l for x in ["discharge", "treatment plan", "plan", "rehab", "diet"]):
            discharge_lines = [self._clean_answer_text(line) for line in matches if any(x in line.lower() for x in ["discharge", "instructions", "rehabilitation", "diet", "monitor", "lifting", "activity"])]
            if discharge_lines:
                return self._join_answer_lines(discharge_lines, limit=4)
        if any(x in question_l for x in ["critical", "urgent", "emergency", "finding"]):
            critical_lines = [self._clean_answer_text(line) for line in matches if any(x in line.lower() for x in ["critical", "urgent", "severe", "elevated", "abnormal", "troponin", "st depression", "impression"])]
            if critical_lines:
                return self._join_answer_lines(critical_lines, limit=3)
        if any(x in question_l for x in ["test", "tests", "procedure", "procedures", "result", "results"]):
            procedure_lines = [self._clean_answer_text(line) for line in matches if any(x in line.lower() for x in ["ecg", "echo", "x-ray", "mri", "ct", "scan", "performed", "showed", "revealed", "impression", "result"])]
            if procedure_lines:
                return self._join_answer_lines(procedure_lines, limit=4)
        if any(x in question_l for x in ["lab", "blood sugar", "troponin", "glucose", "creatinine", "hba1c", "result"]):
            lab_lines = [self._clean_answer_text(line) for line in matches if any(x in line.lower() for x in ["troponin", "hba1c", "glucose", "creatinine", "tsh", "cholesterol", "hb"]) or re.search(r"\b\d+(?:\.\d+)?\s?(mg/dl|g/dl|ng/ml|mmhg|%)\b", line.lower())]
            if lab_lines:
                return self._join_answer_lines(lab_lines, limit=5)
        if any(x in question_l for x in ["follow-up", "follow up", "review", "monitoring", "next step"]):
            followup_lines = [self._clean_answer_text(line) for line in matches if any(x in line.lower() for x in ["follow-up", "follow up", "review", "monitor", "repeat", "after 7 days", "in 2 weeks", "advised"])]
            if followup_lines:
                return self._join_answer_lines(followup_lines, limit=4)
        if any(x in question_l for x in ["hypertension", "blood pressure", "mmhg"]):
            bp_lines = [self._clean_answer_text(line) for line in matches if any(x in line.lower() for x in ["hypertension", "blood pressure", "mmhg", "bp"])]
            if bp_lines:
                return self._join_answer_lines(bp_lines, limit=3)

        answer = self._clean_answer_text(" ".join(matches[:3]))
        return answer if answer else "This information is not available in the document."

    def _looks_like_prompt_leak(self, answer: str) -> bool:
        if not answer:
            return True
        answer_l = answer.lower().strip()
        leaked_patterns = [
            "do not answer with vague fragments",
            "rules:",
            "answer only from the provided context",
            "if the context does not contain the answer",
            "give a direct medical answer",
            "provide a direct medical answer",
            "prefer diagnosis names",
        ]
        return any(pattern in answer_l for pattern in leaked_patterns) or self._is_boilerplate_line(answer_l)

    def query(self, question: str, doc_id: Optional[str] = None) -> Dict:
        try:
            docs = self._search_docs(question, doc_id=doc_id, k=4)
            if not docs:
                return {
                    "answer": "No matching document content was found for this question. Please analyze the document again and try once more.",
                    "sources": [],
                    "confidence": 0.0
                }

            context = "\n\n".join(doc.page_content for doc in docs)
            direct_answer = self._direct_answer_from_docs(question, docs)
            if direct_answer:
                answer = direct_answer
            elif self.llm:
                prompt = MEDICAL_QA_PROMPT.format(context=context, question=question)
                answer = self.llm.invoke(prompt)
            else:
                answer = "Based on the document:\n" + "\n".join([d.page_content for d in docs])[:800]

            answer_clean = self._clean_answer_text(answer)
            if (
                answer_clean.lower() in {"illness.", "illness", "disease.", "disease", "normal.", "normal"}
                or "not available" in answer_clean.lower()
                or self._looks_like_prompt_leak(answer_clean)
                or self._looks_low_value(answer_clean)
            ):
                answer = self._extractive_fallback(question, docs)
                answer_clean = self._clean_answer_text(answer)

            confidence = self._estimate_confidence(answer_clean)
            if self._looks_low_value(answer_clean):
                confidence = min(confidence, 0.35)

            sources = [{"text": d.page_content[:200], "metadata": d.metadata} for d in docs]
            return {"answer": answer_clean, "sources": sources, "confidence": confidence}
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {"answer": f"Error: {str(e)}", "sources": [], "confidence": 0.0}

    def _estimate_confidence(self, answer: str) -> float:
        answer_l = (answer or "").lower()
        if "not available" in answer_l:
            return 0.2
        if self._looks_low_value(answer):
            return 0.3
        if len(answer.split()) > 20:
            return 0.85
        return 0.65

    def clear_documents(self):
        store = self._get_vectorstore()
        store.delete_collection()
        self.vectorstore = None
        self.chain = None
        self.retriever = None