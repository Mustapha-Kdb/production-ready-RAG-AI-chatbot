from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os
import requests
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
import boto3
import tempfile


load_dotenv()
api_key = os.getenv("API_KEY")

EMBED_MODEL = "openai/text-embedding-3-large"
EMBED_DIM = 3072

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""],
)

def load_and_chunk_pdf(path: str):
    if path.startswith("s3://"):
        _, _, rest = path.partition("s3://")
        bucket, _, key = rest.partition("/")
        s3 = boto3.client("s3")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            s3.download_file(bucket, key, tmp.name)
            docs = PDFReader().load_data(file=tmp.name)
    else:
        docs = PDFReader().load_data(file=path)

    texts = [d.text for d in docs if getattr(d, 'text', None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = requests.post(
        url="https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": EMBED_MODEL,
            "input": texts
        })
    )
    data = response.json()
    return [item["embedding"] for item in data["data"]]