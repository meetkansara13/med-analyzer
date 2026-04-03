# 🏥 AI Medical Document Analyzer

A production-ready AI system for analyzing medical documents, extracting entities, summarizing reports, and answering questions using RAG.

## Tech Stack

- **HuggingFace Transformers** – Medical NER + Summarization models
- **LangChain + ChromaDB** – RAG pipeline & vector store
- **PyTorch** – Model inference
- **FastAPI** – REST API backend
- **Streamlit** – Frontend UI
- **MLflow** – Experiment tracking & model registry

## Project Structure

```
med-analyzer/
├── app/
│   ├── api/
│   │   ├── main.py          # FastAPI app
│   │   └── routes/
│   │       ├── analyze.py   # Document analysis endpoints
│   │       └── chat.py      # RAG Q&A endpoints
│   ├── core/
│   │   ├── config.py        # Settings & env vars
│   │   ├── ner_pipeline.py  # HuggingFace NER
│   │   ├── summarizer.py    # HuggingFace Summarization
│   │   └── rag_pipeline.py  # LangChain RAG
│   └── mlops/
│       └── tracker.py       # MLflow tracking
├── streamlit_app.py         # Frontend UI
├── requirements.txt
├── .env.example
└── docker-compose.yml
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your HuggingFace token in .env

# Terminal 1 - Start API
uvicorn app.api.main:app --reload --port 8000

# Terminal 2 - Start UI
streamlit run streamlit_app.py

# Terminal 3 - Start mlflow
mlflow ui
```
