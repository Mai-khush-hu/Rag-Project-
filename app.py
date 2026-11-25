"""A RAG application: ask questions about your own PDFs (LangChain + Qdrant + NVIDIA)."""
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "data" / "documents"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


def load_documents():
    """1. Load every PDF in data/documents (one Document per page)."""
    pdfs = sorted(p for p in DOCS_DIR.glob("*") if p.suffix.lower() == ".pdf")
    if not pdfs:
        sys.exit(f"No PDF files found. Put at least one PDF in: {DOCS_DIR}")
    docs = []
    for pdf in pdfs:
        pages = PyPDFLoader(str(pdf)).load()
        print(f"Loaded {len(pages)} pages from {pdf.name}")
        docs.extend(pages)
    return docs


def split_documents(docs):
    """2. Split pages into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    if not chunks:
        sys.exit("No text could be read from the PDFs. Scanned (image-only) PDFs need OCR first.")
    print(f"Created {len(chunks)} text chunks.")
    return chunks


def build_vector_store(chunks):
    """3. Embed the chunks and index them in an in-memory Qdrant collection."""
    print("Creating embeddings and indexing into Qdrant...")
    store = QdrantVectorStore.from_documents(
        chunks,
        embedding=FastEmbedEmbeddings(),
        location=":memory:",
        collection_name="rag_collection",
    )
    print("Indexing complete!")
    return store


def build_chain():
    """4. Prompt + NVIDIA LLM."""
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("NVIDIA_API_KEY is missing. Add it to the .env file.")
    llm = ChatNVIDIA(
        model=os.getenv("NVIDIA_MODEL", DEFAULT_MODEL),
        nvidia_api_key=api_key,
        temperature=0.2,
        # Nemotron spends tokens on reasoning, so keep this well above 1024
        max_completion_tokens=2500,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer the question accurately based ONLY on the "
         "provided context. If the answer cannot be found in the context, say that you "
         "don't know.\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()


def query_rag(question, retriever, chain):
    """5. Retrieve the most relevant chunks and generate an answer."""
    docs = retriever.invoke(question)
    print(f"\nRetrieved {len(docs)} context chunk(s):")
    for i, doc in enumerate(docs, start=1):
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page")
        page = page + 1 if isinstance(page, int) else "?"  # PyPDFLoader counts from 0
        preview = doc.page_content.strip().replace("\n", " ")[:150]
        print(f" [{i}] {source}, page {page}: {preview}...")

    context = "\n\n".join(doc.page_content for doc in docs)
    print("\nGenerating answer...")
    answer = chain.invoke({"context": context, "question": question})
    print(f"\nAnswer:\n{answer}")
    return answer


def main():
    chain = build_chain()  # first, so a missing key fails before slow indexing
    store = build_vector_store(split_documents(load_documents()))
    retriever = store.as_retriever(search_kwargs={"k": 3})

    print("\nAsk a question about your documents. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            query_rag(question, retriever, chain)
        except Exception as err:
            print(f"Request failed: {err}")
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
