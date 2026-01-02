import os
import sys
import logging
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.documents import Document
from operator import itemgetter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_models.common import get_llm, get_embeddings, get_sparse_model, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, RETRIEVAL_TOP_K

logger = logging.getLogger("sbv_advancedrag")

# Load environment variables
load_dotenv()

# LangSmith Configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "sbv-advancedrag"

def normalize_scores(scores):
    """
    Min-Max normalization of scores to [0, 1] range.
    """
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [1.0] * len(scores) if max_score > 0 else [0.0] * len(scores)
        
    return [(s - min_score) / (max_score - min_score) for s in scores]

def create_advanced_rag_chain():
    """
    Creates an Advanced RAG chain using Hybrid Search (Dense + Sparse) and Weighted Fusion.
    Weights: 0.75 Sparse + 0.25 Dense.
    """
    # 1. Initialize Models
    embeddings = get_embeddings()
    sparse_model = get_sparse_model()

    # 2. Initialize Qdrant Client
    logger.info("Step 1: Initializing Qdrant Client for Hybrid Search")
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        logger.info("Step 1 success with Qdrant client initialized")
    except Exception as e:
        logger.error(f"Step 1 got error: Qdrant initialization failed - {e}")
        raise e

    # Custom Hybrid Retrieval with Weighted Fusion
    def retrieve_hybrid(inputs: dict):
        question = inputs["question"]
        top_k = inputs.get("top_k", RETRIEVAL_TOP_K)
        logger.info(f"Step 2: Hybrid Retrieval (question_length={len(question)}, top_k={top_k})")
        
        # --- 1. Dense Search ---
        query_dense = embeddings.embed_query(question)
        try:
            dense_results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_dense,
                using="dense",
                limit=top_k,
                with_payload=True
            ).points
        except Exception as e:
            logger.error(f"  Error: Dense search failed - {e}")
            dense_results = []

        # --- 2. Sparse Search ---
        query_sparse = sparse_model.transform(question)
        try:
            sparse_results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_sparse,
                using="sparse",
                limit=top_k,
                with_payload=True
            ).points
        except Exception as e:
            logger.error(f"  Error: Sparse search failed - {e}")
            sparse_results = []

        # --- 3. Weighted Fusion ---
        # Map: doc_id -> {doc, dense_score, sparse_score}
        # Using point.id (integer) as key
        fused_results = {}
        
        # Process Dense
        dense_scores = [p.score for p in dense_results]
        norm_dense_scores = normalize_scores(dense_scores)
        
        for i, point in enumerate(dense_results):
            pid = point.id
            if pid not in fused_results:
                fused_results[pid] = {
                    "point": point,
                    "dense_score": norm_dense_scores[i],
                    "sparse_score": 0.0
                }
            else:
                fused_results[pid]["dense_score"] = norm_dense_scores[i]

        # Process Sparse
        sparse_scores = [p.score for p in sparse_results]
        norm_sparse_scores = normalize_scores(sparse_scores)
        
        for i, point in enumerate(sparse_results):
            pid = point.id
            if pid not in fused_results:
                fused_results[pid] = {
                    "point": point,
                    "dense_score": 0.0,
                    "sparse_score": norm_sparse_scores[i]
                }
            else:
                fused_results[pid]["sparse_score"] = norm_sparse_scores[i]
                # Prefer the point object from sparse if it exists (should be same payload)
        
        # Calculate Final Score
        # Weight: 0.75 Sparse + 0.25 Dense
        final_docs = []
        for pid, data in fused_results.items():
            final_score = (0.75 * data["sparse_score"]) + (0.25 * data["dense_score"])
            
            point = data["point"]
            content = point.payload.get("content", "")
            metadata = point.payload if point.payload else {}
            
            # Store scores in metadata for debugging/LangSmith
            metadata["score"] = final_score
            metadata["dense_score_norm"] = data["dense_score"]
            metadata["sparse_score_norm"] = data["sparse_score"]
            
            doc = Document(page_content=content, metadata=metadata)
            final_docs.append(doc)
            
        # Sort by final score descending
        final_docs.sort(key=lambda x: x.metadata["score"], reverse=True)
        
        # Return Top K
        return final_docs[:top_k]

    # 3. Initialize LLM
    llm = get_llm()

    # 4. Create Prompt
    template = """Bạn là trợ lý pháp lý hữu ích. Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp dưới đây.
    Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy nói rằng bạn không biết.
    
    Ngữ cảnh (kèm điểm số phù hợp):
    {context}
    
    Câu hỏi: {question}
    
    Trả lời:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # 5. Create LCEL Chain
    # 5. Create LCEL Chain
    
    def format_docs(docs):
        return "\n\n".join([f"Content: {d.page_content}\nScore: {d.metadata.get('score')}" for d in docs])

    rag_chain = (
        RunnableParallel({
            "context": RunnableLambda(retrieve_hybrid),
            "question": itemgetter("question")
        })
        | RunnableParallel({
            "answer": (
                RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
                | prompt 
                | llm 
                | StrOutputParser()
            ),
            "documents": itemgetter("context")
        })
    )

    return rag_chain

def main():
    logger.info("Initializing Advanced RAG Pipeline (Hybrid Search + Weighted Fusion)")
    try:
        chain = create_advanced_rag_chain()
        
        question = "Ngân hàng có bắt buộc phải công khai thông tin về Open API trên trang thông tin điện tử không?"
        logger.info(f"Test Question: {question}")
        
        # Invoke the chain
        response = chain.invoke(question)
        
        logger.info("Answer:")
        logger.info(response)
        
        # We can't easily access the intermediate docs from the chain invoke unless we modify the chain to return them or use a callback.
        # But for verification, we can just call the retrieval function directly if we want, or trust the metadata in LangSmith.
        # Actually, let's modify the chain to return source documents or just run retrieval separately for debug print.
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
