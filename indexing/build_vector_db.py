import os
import json
import time
import sys
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, SparseVectorParams
from sentence_transformers import SentenceTransformer

# Add parent directory to path to allow importing from model package
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from ai_models.common import BM25SparseEmbedding

# Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL")
SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL", "BM25")

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
ARTICLES_FILE = os.path.join(DATA_DIR, "sbv_legal_articles.json")
BM25_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "bm25_model.pkl")

def build_vector_db():
    print("Initializing Qdrant client...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # 1. Load Data
    print(f"Loading articles from {ARTICLES_FILE}...")
    with open(ARTICLES_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)
    
    # Filter valid articles
    valid_articles = [a for a in articles if a.get("content") and a.get("id")]
    print(f"Found {len(valid_articles)} valid articles.")
    
    texts = [a["content"] for a in valid_articles]
    
    # 2. Prepare Models
    print(f"Loading Dense Model: {EMBEDDING_MODEL_NAME}...")
    dense_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print(f"Initializing Sparse Model: {SPARSE_MODEL_NAME}...")
    sparse_model = BM25SparseEmbedding()
    sparse_model.fit(texts)
    
    # Save the fitted model
    print(f"Saving BM25 model to {BM25_MODEL_PATH}...")
    sparse_model.save(BM25_MODEL_PATH)
    
    # 3. Check/Create Collection
    print(f"Checking collection '{COLLECTION_NAME}'...")
    
    # Get dense dimension
    sample_dense = dense_model.encode("test")
    dense_dim = len(sample_dense)
    
    collection_exists = False
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        collection_exists = True
        print(f"Collection exists. Will update existing points.")
        print(f"Existing points: {collection_info.points_count}")
    except Exception:
        print(f"Collection does not exist. Creating new collection...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(
                    size=dense_dim,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False,
                    )
                )
            }
        )
    
    # 4. Generate Embeddings and Upload
    print("Generating embeddings and uploading...")
    batch_size = 100
    total = len(valid_articles)
    
    for i in range(0, total, batch_size):
        batch_articles = valid_articles[i : i + batch_size]
        batch_texts = [a["content"] for a in batch_articles]
        
        # Dense
        dense_vectors = dense_model.encode(batch_texts)
        
        # Sparse & Points
        points = []
        for j, article in enumerate(batch_articles):
            # Sparse (Use transform_document for indexing)
            sparse_vector = sparse_model.transform_document(article["content"])
            
            # Payload
            payload = {
                "id": article["id"],
                "law_id": article.get("law_id"),
                "title": article.get("article_title"),
                "content": article.get("content"),
                "chapter": article.get("chapter"),
                "section": article.get("section"),
                "type": "ARTICLE"
            }
            
            points.append(models.PointStruct(
                id=i + j, # Integer ID for Qdrant
                vector={
                    "dense": dense_vectors[j].tolist(),
                    "sparse": sparse_vector
                },
                payload=payload
            ))
            
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"Processed {min(i + batch_size, total)}/{total} articles.")
        
    print("Vector database build complete!")

if __name__ == "__main__":
    build_vector_db()
