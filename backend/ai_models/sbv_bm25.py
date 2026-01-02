import os
import sys
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from operator import itemgetter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_models.common import get_llm, get_sparse_model, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, RETRIEVAL_TOP_K

logger = logging.getLogger("sbv_bm25")

# Load environment variables
load_dotenv()

# LangSmith Configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "sbv-bm25"

def create_bm25_chain():
    """
    Creates a BM25 RAG chain using Qdrant (Sparse) and LCEL.
    """
    # 1. Initialize Sparse Model
    sparse_model = get_sparse_model()

    # 2. Initialize Qdrant Client
    logger.info("Step 1: Initializing Qdrant Client for BM25 Sparse Search")
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        logger.info("Step 1: Success with Qdrant client initialized")
    except Exception as e:
        logger.error(f"Step 1: Error: Qdrant initialization failed - {e}")
        raise e

    # Custom Retrieval with Scores in Metadata (Direct Qdrant Client)
    def retrieve_with_scores(inputs: dict):
        question = inputs["question"]
        top_k = inputs.get("top_k", RETRIEVAL_TOP_K)
        logger.info(f"Step 2: BM25 Sparse Retrieval (question_length={len(question)}, top_k={top_k})")
        
        # 1. Embed query (Sparse)
        query_vector = sparse_model.transform(question)
        logger.info(f"Step 2: Generated sparse query vector with {len(query_vector.indices)} terms")
        logger.debug(f"Query indices: {query_vector.indices[:min(10, len(query_vector.indices))]}")
        logger.debug(f"Query values: {query_vector.values[:min(10, len(query_vector.values))]}")
        
        # 2. Search Qdrant (using query_points)
        # Note: Using named vector "sparse" as configured in build_vector_db.py
        try:
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                using="sparse",
                limit=top_k,
                with_payload=True
            ).points
        except Exception as e:
            logger.error(f"Step 2: Error: Query failed - {e}")
            raise e
        
        docs = []
        from langchain_core.documents import Document
        for point in results:
            content = point.payload.get("content", "")
            # Create metadata dict with score
            metadata = point.payload if point.payload else {}
            metadata["score"] = point.score
            
            doc = Document(page_content=content, metadata=metadata)
            docs.append(doc)
        
        # Log success with document IDs and scores
        doc_ids = [doc.metadata.get('id', 'unknown') for doc in docs]
        doc_scores = {doc.metadata.get('id', 'unknown'): doc.metadata.get('score') for doc in docs}
        logger.info(f"Step 2: Success with {len(docs)} documents retrieved: {doc_ids}")
        logger.info(f"Step 2: Document scores: {doc_scores}")
            
        return docs
        
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
    # Input to chain is now a dict: {"question": "...", "top_k": ...}
    # We pass the whole dict to retrieve_with_scores
    # 5. Create LCEL Chain
    # Input to chain is now a dict: {"question": "...", "top_k": ...}
    
    def format_docs(docs):
        return "\n\n".join([f"Content: {d.page_content}\nScore: {d.metadata.get('score')}" for d in docs])

    rag_chain = (
        RunnableParallel({
            "context": RunnableLambda(retrieve_with_scores),
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
    logger.info("Initializing BM25 RAG Pipeline (Qdrant Sparse)")
    try:
        chain = create_bm25_chain()
        
        question = "Ngân hàng có bắt buộc phải công khai thông tin về Open API trên trang thông tin điện tử không?"
        logger.info(f"Test Question: {question}")
        
        # Invoke the chain
        answer = chain.invoke(question)
        
        logger.info("Answer:")
        logger.info(answer)
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
