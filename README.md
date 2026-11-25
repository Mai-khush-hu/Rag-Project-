# A RAG Application

Ask questions about your own PDF documents. The app retrieves the most relevant
passages and an NVIDIA-hosted LLM answers using only those passages
(RAG = Retrieval-Augmented Generation).

## Workflow
```
PDFs in data/documents
  -> Load (PyPDFLoader)
  -> Chunk (600 characters, 80 overlap)
  -> Embed (FastEmbed, runs locally)
  -> Store (Qdrant, in memory)

Question -> Retrieve top 3 chunks -> Prompt + context -> NVIDIA LLM -> Answer
```

## Folder structure
```
RAG-Project/
├── app.py              # the whole pipeline
├── requirements.txt    # dependencies
├── .env                # NVIDIA API key and model name (keep private)
├── .gitignore          # stops .env being committed
├── README.md
└── data/
    └── documents/      # put your PDFs here
```

## Setup
```bash
cd RAG-Project
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```
1. Check that `.env` contains `NVIDIA_API_KEY=...`
2. Copy your PDF (for example `STATS.pdf`) into `data/documents/`
3. Run `python app.py`

The first run downloads the embedding model, so it needs internet and takes longer.

## Example
```
Question: What are the normal equations for fitting a straight line using the principle of least squares?
```
Type `exit` to quit.

## Tech stack
Python, LangChain, FastEmbed, Qdrant, NVIDIA AI Endpoints
(default model `nvidia/nemotron-3.5-lightning-30b-a3b`, changeable in `.env`)

## Limitations
- The index is kept in memory and rebuilt on every run
- Scanned (image-only) PDFs need OCR first
- Each question is answered on its own, without chat history

## Troubleshooting
| Problem | Fix |
|---|---|
| 401 or 403 from NVIDIA | Create a new key at build.nvidia.com and update `.env` |
| Model not found | Change `NVIDIA_MODEL` in `.env` |
| "No PDF files found" | Add a PDF to `data/documents/` |
