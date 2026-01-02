import json
import os
import uuid
import logging
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

from ai_models.common import get_embeddings, get_qdrant_client, get_embedding_dimension, COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY

logger = logging.getLogger("ingest")

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSON file."""
    logger.info(f"Loading data from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} records")
    return data

def prepare_documents(data: List[Dict[str, Any]]) -> List[Document]:
    """Convert raw data to LangChain Documents."""
    documents = []
    for item in data:
        # Construct content for embedding
        # You might want to customize what goes into page_content vs metadata
        content = item.get("content", "")
        if not content or not content.strip():
            continue
            
        # Metadata
        metadata = {
            "id": item.get("id"),
            "type": item.get("type"),
            "document_id": item.get("document_id"),
            "vbpl_id": item.get("vbpl_id"),
            "document_title": item.get("document_title"),
            "article_title": item.get("article_title"),
            "document_status": item.get("document_status"),
            "effective_date": item.get("effective_date"),
        }
        
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    logger.info(f"Prepared {len(documents)} documents (filtered empty)")
    return documents

def setup_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int = 1536):
    """Create or recreate Qdrant collection with hybrid search configuration."""
    if client.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' exists. Recreating")
        client.delete_collection(collection_name)
    
    logger.info(f"Creating collection '{collection_name}'")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        ),
        # Enable sparse vectors for hybrid search if needed, 
        # but LangChain's Qdrant wrapper often handles this via add_documents with specific config
        # or we can use the 'sparse_vector_config' if we want to use Qdrant's sparse search.
        # For simplicity with standard LangChain usage, we'll focus on dense vectors first,
        # but to support "Hybrid" properly in Qdrant, we usually need a sparse vector model too.
        # Here we will configure it to support sparse vectors as well.
        sparse_vectors_config={
            "text-sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                )
            )
        }
    )
    logger.info("Collection created")

def main():
    # Initialize Qdrant Client
    client = get_qdrant_client()
    
    # Initialize Embeddings
    embeddings = get_embeddings()
    vector_size = get_embedding_dimension()
    
    # Test Embedding
    try:
        logger.info("Testing embedding connection")
        test_emb = embeddings.embed_query("test")
        vector_size = len(test_emb)
        logger.info(f"Test embedding successful. Vector length: {vector_size}")
    except Exception as e:
        logger.error(f"Test embedding failed: {e}")
        return
    
    # Load and Prepare Data
    data_file = "sbv_legal_articles.json"
    if not os.path.exists(data_file):
        logger.error(f"Error: {data_file} not found")
        return

    raw_data = load_data(data_file)
    documents = prepare_documents(raw_data)
    
    # Setup Collection
    setup_qdrant_collection(client, COLLECTION_NAME, vector_size=vector_size)
    
    # Initialize Vector Store
    # We use retrieval_mode="HYBRID" which requires sparse embeddings.
    # LangChain's QdrantVectorStore can handle sparse embeddings if we provide a sparse_embedding_model.
    # For this example, we will use FastEmbed for sparse embeddings (default in some Qdrant setups) 
    # or we can rely on Qdrant's server-side sparse vectors if configured.
    # To keep it simple and robust, we will use the 'FastEmbedSparse' from langchain-qdrant if available,
    # or just stick to dense if the user environment is limited, but the requirement was "Hybrid".
    
    try:
        from langchain_qdrant import FastEmbedSparse
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        logger.info("Success: FastEmbedSparse initialized with Qdrant/bm25")
    except ImportError:
        logger.warning("Could not import FastEmbedSparse. Installing fastembed might be required")
        logger.warning("Falling back to Dense only for now, or ensure 'fastembed' is installed")
        sparse_embeddings = None

    logger.info("Ingesting documents")
    
    QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        sparse_vector_name="text-sparse",
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,

        force_recreate=False # We already recreated it manually above
    )
    
    logger.info("Ingestion complete!")

if __name__ == "__main__":
    main()
