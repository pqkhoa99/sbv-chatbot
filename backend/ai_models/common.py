import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from qdrant_client import QdrantClient
from neo4j import GraphDatabase

logger = logging.getLogger("ai_models.common")

# Load environment variables
load_dotenv()

DATA_FILE = 'sbv_legal_articles.json'

# Configuration Constants
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sbv_legal_articles")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 20))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 5))
CONCEPT_PATH_THRESHOLD = float(os.getenv("CONCEPT_PATH_THRESHOLD", 0.5))
RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "flashrank").lower()
RERANK_MODEL = os.getenv("RERANK_MODEL", "ms-marco-MiniLM-L-12-v2")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
logger.debug(f"LLM_PROVIDER is '{LLM_PROVIDER}'")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

# Self-Host / Custom OpenAI Config
SELFHOSTED_API_KEY = os.getenv("SELFHOSTED_API_KEY")
SELFHOSTED_BASE_URL = os.getenv("SELFHOSTED_BASE_URL")
SELFHOSTED_MODEL = os.getenv("SELFHOSTED_MODEL", "gpt-4o")

# Embeddings Config
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Neo4j Config
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def get_llm(temperature: float = 0):
    """Factory to get the configured LLM."""
    if LLM_PROVIDER == "self_host":
        # logger.info(f"Using Self-Hosted LLM ({SELFHOSTED_MODEL})")
        return ChatOpenAI(
            model=SELFHOSTED_MODEL,
            api_key=SELFHOSTED_API_KEY,
            base_url=SELFHOSTED_BASE_URL,
            temperature=temperature
        )
    elif LLM_PROVIDER == "google" or LLM_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set.")
        # logger.info(f"Using Google Gemini ({GEMINI_MODEL})")
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY, temperature=temperature)
    else:
        # Default to OpenAI
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        # logger.info(f"Using OpenAI ({OPENAI_MODEL})")
        # Some newer models or specific endpoints might not support temperature=0
        return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)

from typing import List
from langchain_core.embeddings import Embeddings
from openai import OpenAI

class SelfHostedEmbeddings(Embeddings):
    def __init__(self, api_key, base_url, model):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Embed in batches of 10 to be safe
        batch_size = 10
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = self.client.embeddings.create(input=batch, model=self.model)
                # Sort by index to ensure order matches input
                data = sorted(response.data, key=lambda x: x.index)
                embeddings.extend([d.embedding for d in data])
            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                # Fallback or re-raise? Let's re-raise to stop ingestion
                raise e
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

def get_embeddings():
    """Factory to get the configured Embeddings."""
    if EMBEDDING_PROVIDER == "self_host":
        # logger.info(f"Using Self-Hosted Embeddings (Model: {EMBEDDING_MODEL})")
        return SelfHostedEmbeddings(
            api_key=SELFHOSTED_API_KEY,
            base_url=SELFHOSTED_BASE_URL,
            model=EMBEDDING_MODEL
        )
    elif EMBEDDING_PROVIDER == "google" or EMBEDDING_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set.")
        # logger.info("Using Google Gemini Embeddings")
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    elif EMBEDDING_PROVIDER == "sentence_transformers":
        # logger.info(f"Using Sentence Transformers ({EMBEDDING_MODEL})")
        # Try importing from langchain_huggingface, fallback to community
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
        return HuggingFaceEmbeddings(model_name="minhquan6203/paraphrase-vietnamese-law")
    else:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        # logger.info("Using OpenAI Embeddings")
        return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

def get_embedding_dimension():
    """Get expected dimension for the configured embedding model."""
    if EMBEDDING_PROVIDER == "google" or EMBEDDING_PROVIDER == "gemini":
        return 768
    elif EMBEDDING_PROVIDER == "sentence_transformers":
        return 768
    else:
        # OpenAI text-embedding-3-small is 1536
        # Self-host: depends on model, but let's assume 1536 or user needs to know.
        # UPDATE: Self-hosted gemini-embedding-001 returns 3072.
        return 3072

# Qdrant Config Removed
# QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sbv_legal_articles")

def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def get_qdrant_client():
    """Factory to get Qdrant Client."""
    from qdrant_client import QdrantClient
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

SPARSE_MODEL = os.getenv("SPARSE_MODEL", "BM25")

import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from qdrant_client.http import models

class BM25SparseEmbedding:
    """
    Custom BM25 implementation using sklearn CountVectorizer.
    Generates sparse vectors compatible with Qdrant.
    """
    def __init__(self, k1=1.5, b=0.75):
        self.vectorizer = CountVectorizer()
        self.k1 = k1
        self.b = b
        self.idf = None
        self.avgdl = 0
        self.is_fitted = False

    def fit(self, corpus: List[str]):
        # logger.info("Fitting BM25 model")
        # Get term frequencies
        X = self.vectorizer.fit_transform(corpus)
        self.n_samples, self.n_features = X.shape
        
        # Calculate doc lengths
        doc_lengths = np.array(X.sum(axis=1)).flatten()
        self.avgdl = np.mean(doc_lengths)
        
        # Calculate IDF
        # IDF = log((N - n(q) + 0.5) / (n(q) + 0.5) + 1)
        doc_count = np.array((X > 0).sum(axis=0)).flatten()
        self.idf = np.log((self.n_samples - doc_count + 0.5) / (doc_count + 0.5) + 1)
        
        self.is_fitted = True
        # logger.info(f"BM25 fitted on {self.n_samples} documents. Vocab size: {self.n_features}")
        return self

    def transform(self, text: str) -> models.SparseVector:
        if not self.is_fitted:
            raise ValueError("BM25 model is not fitted.")
            
        # Transform single text
        vec = self.vectorizer.transform([text])
        indices = vec.indices
        data = vec.data
        
        # Calculate BM25 weights
        # score = IDF * (f * (k1 + 1)) / (f + k1 * (1 - b + b * |D| / avgdl))
        doc_len = vec.sum() # Use query length
        # Note: For query, we usually use average doc length or just query length?
        # Standard BM25 for query usually just sums the IDFs of matching terms, 
        # but for sparse vector retrieval, we want to weight the query terms.
        # Qdrant will compute dot product: sum(q_i * d_i).
        # If d_i is BM25 weight, q_i should be binary or term frequency?
        # Usually q_i is 1 (binary) for simple BM25.
        # But here we are generating a sparse vector for the query.
        # Let's stick to the same formula but maybe simplified for query?
        # Actually, if we put BM25 weights in the Document vector, the Query vector should just be TF (or binary).
        # Qdrant's sparse search is dot product.
        # BM25(q, d) = sum( IDF(q_i) * ... )
        # So Document vector should contain the complex term:
        #   IDF * (f * (k1 + 1)) / (f + k1 * (1 - b + b * |D| / avgdl))
        # And Query vector should just be 1 (or q_tf).
        # WAIT. My previous implementation in build_vector_db.py put the FULL BM25 score into the document vector.
        # So for the query, I should just return the indices with value 1?
        # If I put weights in query, I get weighted sum of BM25 scores? That might be wrong.
        # Let's check how Qdrant recommends BM25.
        # Usually: Doc = TF-IDF or BM25-weighted. Query = Binary or TF.
        # Let's assume Query = Binary (1.0) for now to match standard BM25 behavior where we just sum scores of matching terms.
        
        weights = []
        for idx in indices:
            # Just use 1.0 for query terms so dot product = sum(doc_weights)
            weights.append(1.0)
            
        return models.SparseVector(
            indices=indices.tolist(),
            values=weights
        )
        
    def transform_document(self, text: str) -> models.SparseVector:
        """Transform document text (for indexing)."""
        if not self.is_fitted:
            raise ValueError("BM25 model is not fitted.")
            
        vec = self.vectorizer.transform([text])
        indices = vec.indices
        data = vec.data
        doc_len = vec.sum()
        
        weights = []
        for idx, freq in zip(indices, data):
            idf_val = self.idf[idx]
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score = idf_val * numerator / denominator
            weights.append(float(score))
            
        return models.SparseVector(
            indices=indices.tolist(),
            values=weights
        )

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

BM25_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bm25_model.pkl")

def get_sparse_model():
    """Load the fitted BM25 model."""
    if not os.path.exists(BM25_MODEL_PATH):
        raise FileNotFoundError(f"BM25 model not found at {BM25_MODEL_PATH}. Run build_vector_db.py first.")
    return BM25SparseEmbedding.load(BM25_MODEL_PATH)
