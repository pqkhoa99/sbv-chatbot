import os
import sys
import json
from typing import TypedDict, List, Dict, Any, Optional
import numpy as np
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from flashrank import Ranker, RerankRequest

# Add parent directory to path to allow importing from pipeline package if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.common import get_llm, get_embeddings, get_neo4j_driver, RETRIEVAL_TOP_K, RERANK_TOP_K, CONCEPT_PATH_THRESHOLD, RERANK_PROVIDER, RERANK_MODEL

# Load environment variables
load_dotenv()

# LangSmith Configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "sbv-lawgraph"

# Initialize Components
embeddings = get_embeddings()
print(f"DEBUG: embeddings type: {type(embeddings)}")
llm = get_llm()

# Initialize Reranker
print(f"Initializing Reranker ({RERANK_PROVIDER}: {RERANK_MODEL})...")
reranker = Ranker(model_name=RERANK_MODEL, cache_dir="./flashrank_cache")

# --- State Definition ---
class GraphState(TypedDict):
    question: str
    documents: List[Document]  # Initial retrieval results
    reranked_docs: List[Document] # After reranking
    graph_context: List[Document] # Additional context from Graph
    concepts: List[str] # Relevant concepts
    concept_context: str # Expanded context for concepts
    fused_context: str # Final context string for LLM
    answer: str
    top_k: Optional[int] # Dynamic top_k for evaluation
    threshold: Optional[float] # Dynamic threshold for filtering
    sparse_weight: Optional[float] # Dynamic sparse weight (0.0 to 1.0) - Unused in Rerank Model

# --- Nodes ---

def hybrid_search(state: GraphState):
    """
    Step 1: Hybrid Search (Neo4j Vector Index + Graph-based Sparse Search)
    Combines dense vector search with sparse graph traversal.
    """
    print("---HYBRID SEARCH (Neo4j Dense + Graph Sparse)---")
    question = state["question"]
    
    driver = get_neo4j_driver()
    
    # Initialize Sparse Model for Query Embedding
    from model.common import get_sparse_model
    try:
        sparse_model = get_sparse_model()
    except Exception as e:
        print(f"Failed to load Sparse Model: {e}")
        return {"documents": []}

    documents = []
    
    try:
        # Determine K
        # For Reranking, we want a larger pool of candidates.
        # If user asks for top_k=5, we should fetch maybe 20 or 50.
        # But let's respect the input k for the final output, and fetch more here.
        current_k = state.get("top_k")
        final_k = current_k if current_k else RETRIEVAL_TOP_K
        
        # Fetch 3x candidates for reranking, or at least 20
        fetch_k = final_k * 2
        print(f"Fetching {fetch_k} candidates for reranking (Final K: {final_k})...")
        
    # --- 1. Dense Search (Neo4j Vector Index) ---
        print(f"Executing Dense Search (k={fetch_k})...")
        query_embedding = embeddings.embed_query(question)
        
        vector_query = """
        CALL db.index.vector.queryNodes('article_embedding_index', $k, $embedding)
        YIELD node, score
        RETURN node.id as id, node.content as content, node.title as title, score
        """
        
        with driver.session() as session:
            vec_results = list(session.run(vector_query, k=fetch_k, embedding=query_embedding))
            
        # --- 2. Sparse Search (Neo4j Fulltext + Sparse Vector Re-scoring) ---
        print(f"Executing Sparse Search (Neo4j Fulltext + Sparse Vector Re-scoring)...")
        
        # A. Generate Query Sparse Vector
        try:
            # transform() returns Binary Query Vector (indices with weight 1.0)
            query_sparse_vec = sparse_model.transform(question)
            query_indices = set(query_sparse_vec.indices)
        except Exception as e:
            print(f"Failed to generate query sparse vector: {e}")
            query_indices = set()

        # B. Candidate Generation via Fulltext
        # Fetch more candidates for re-ranking
        candidate_k = fetch_k * 2
        
        # Clean question for Lucene
        import re
        clean_question = re.sub(r'[^\w\s]', ' ', question)
        
        fulltext_query = """
        CALL db.index.fulltext.queryNodes("article_fulltext_index", $q, {limit: $k})
        YIELD node, score
        RETURN node.id as id, node.content as content, node.title as title, node.sparse_embedding as sparse_embedding
        """
        
        with driver.session() as session:
            try:
                ft_results = list(session.run(fulltext_query, q=clean_question, k=candidate_k))
            except Exception as e:
                print(f"Fulltext search failed: {e}")
                ft_results = []
            
        # C. Re-score using Sparse Vector (Dot Product)
        sparse_results = []
        for record in ft_results:
            try:
                # Parse stored sparse vector: {"index": score}
                doc_sparse_json = record["sparse_embedding"]
                if not doc_sparse_json:
                    final_score = 0.0
                else:
                    doc_sparse_vec = json.loads(doc_sparse_json)
                    # Dot Product (Query weights are 1.0)
                    final_score = sum(doc_sparse_vec.get(str(idx), 0.0) for idx in query_indices)
                
                sparse_results.append({
                    "id": record["id"],
                    "content": record["content"],
                    "title": record["title"],
                    "score": final_score
                })
            except Exception as e:
                print(f"Error scoring document {record['id']}: {e}")
                continue
                
        # D. Sort and Top K
        sparse_results.sort(key=lambda x: x["score"], reverse=True)
        sparse_results = sparse_results[:fetch_k]
        
        print(f"DEBUG: Found {len(vec_results)} dense results and {len(sparse_results)} sparse results.")
        
        # --- 3. Combine Results (Union) ---
        combined = {}
        
        # Process Vector Results (Dense)
        for record in vec_results:
            did = record["id"]
            if did not in combined:
                combined[did] = {
                    "doc": Document(
                        page_content=record["content"] if record["content"] else "",
                        metadata={
                            "id": record["id"],
                            "title": record["title"]
                        }
                    ),
                    "dense_score": record["score"],
                    "sparse_score": None
                }
            else:
                combined[did]["dense_score"] = record["score"]
                
        # Process Sparse Results (Fulltext)
        for record in sparse_results:
            did = record["id"]
            score = record["score"]
            
            if did not in combined:
                combined[did] = {
                    "doc": Document(
                        page_content=record["content"] if record["content"] else "",
                        metadata={
                            "id": did,
                            "title": record["title"]
                        }
                    ),
                    "dense_score": None,
                    "sparse_score": score
                }
            else:
                combined[did]["sparse_score"] = score
        
        # 4. Create Documents List
        documents = []
        for info in combined.values():
            doc = info["doc"]
            doc.metadata["dense_score"] = info["dense_score"]
            doc.metadata["sparse_score"] = info["sparse_score"]
            documents.append(doc)
            
        # Optional: Sort by ID for deterministic output before reranking
        documents.sort(key=lambda x: x.metadata["id"])
        
        print(f"Retrieved {len(documents)} unique documents for reranking.")
            
    except Exception as e:
        print(f"Retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        documents = []
    finally:
        driver.close()
        
    return {"documents": documents, "question": question}

def threshold_validation(state: GraphState):
    """
    Step 3: Threshold Validation
    Filter out documents with low relevance scores from reranker.
    """
    print("---THRESHOLD VALIDATION---")
    reranked_docs = state["reranked_docs"]
    
    if not reranked_docs:
        print("No documents found after reranking.")
        return {"reranked_docs": []}
        
    # Example filtering (can be enhanced with actual score check)
    # filtered_docs = [doc for doc in reranked_docs if doc.metadata.get("rerank_score", 0) > THRESHOLD]
    # For now, just pass through, but logically this is where we'd filter.
    
    return {"reranked_docs": reranked_docs}

def rerank(state: GraphState):
    """
    Step 2: Reranking (FlashRank)
    Use Cross-Encoder to re-score documents.
    """
    print(f"---RERANK (FlashRank: {RERANK_MODEL})---")
    documents = state["documents"]
    question = state["question"]
    
    if not documents:
        return {"reranked_docs": []}
    
    # Use dynamic top_k if provided, else default RERANK_TOP_K
    top_k = state.get("top_k", RERANK_TOP_K)
    if top_k is None: top_k = RERANK_TOP_K
    
    try:
        # Prepare requests for FlashRank
        pass_docs = []
        for doc in documents:
            pass_docs.append({
                "id": doc.metadata.get("id"),
                "text": doc.page_content,
                "meta": doc.metadata
            })
            
        rerank_request = RerankRequest(query=question, passages=pass_docs)
        results = reranker.rerank(rerank_request)
        
        # Sort results by score
        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_k]
        
        reranked_docs = []
        for res in top_results:
            # Reconstruct Document
            meta = res["meta"]
            meta["rerank_score"] = res["score"]
            doc = Document(page_content=res["text"], metadata=meta)
            reranked_docs.append(doc)
            
        print(f"Reranked top {len(reranked_docs)} documents using FlashRank.")
        for doc in reranked_docs:
            print(f"  - {doc.metadata.get('id')}: {doc.metadata.get('rerank_score'):.4f}")
            
        return {"reranked_docs": reranked_docs}

    except Exception as e:
        print(f"FlashRank Reranking failed: {e}. Fallback to top {top_k}.")
        import traceback
        traceback.print_exc()
        return {"reranked_docs": documents[:top_k]}

def relationship_retrieval(state: GraphState):
    """
    Step 4: Relationship Retrieval (Legal Context Expansion)
    For each retrieved article, find related articles in Graph.
    """
    print("---RELATIONSHIP RETRIEVAL (Neo4j)---")
    reranked_docs = state["reranked_docs"]
    question = state["question"]
    
    driver = get_neo4j_driver()
    graph_context_docs = []
    
    try:
        # 1. Related Articles Logic
        article_ids = [doc.metadata.get("id") for doc in reranked_docs if doc.metadata.get("id")]
        
        if article_ids:
            query = """
            MATCH (a:ARTICLE) WHERE a.id IN $article_ids
            
            // 1. Get Parent Document
            OPTIONAL MATCH (d:DOCUMENT)-[:CONTAIN]->(a)
            
            // 2. Get Modifications (Incoming & Outgoing)
            OPTIONAL MATCH (modifier:ARTICLE)-[r_in:AMEND|REPLACE|REPEAL]->(a)
            OPTIONAL MATCH (a)-[r_out:AMEND|REPLACE|REPEAL]->(modified:ARTICLE)
            
            RETURN a.id as article_id, 
                   d.title as doc_title, d.id as doc_id, d.status as doc_status, 
                   d.effective_date as effective_date, d.expired_date as expired_date,
                   collect(DISTINCT {rel: type(r_in), id: modifier.id, title: modifier.title, content: modifier.content, direction: "incoming"}) as incoming_mods,
                   collect(DISTINCT {rel: type(r_out), id: modified.id, title: modified.title, content: modified.content, direction: "outgoing"}) as outgoing_mods
            """
            
            with driver.session() as session:
                result = session.run(query, article_ids=article_ids)
                
                for record in result:
                    a_id = record["article_id"]
                    
                    # Format Graph Note
                    notes = []
                    if record['doc_title']:
                        status_str = f"Trạng thái: {record['doc_status']}"
                        if record['effective_date']:
                            status_str += f" | Hiệu lực: {record['effective_date']}"
                        if record['expired_date']:
                            status_str += f" | Hết hạn: {record['expired_date']}"
                            
                        notes.append(f"- Thuộc văn bản: {record['doc_title']} (Số hiệu: {record['doc_id']}).\n  {status_str}")
                    
                    for mod in record['incoming_mods']:
                        if not mod['id']: continue
                        rel_vn = mod['rel']
                        if rel_vn == "AMEND": rel_vn = "SỬA ĐỔI"
                        elif rel_vn == "REPLACE": rel_vn = "THAY THẾ"
                        elif rel_vn == "REPEAL": rel_vn = "BÃI BỎ"
                        
                        content_full = mod['content'] if mod['content'] else "Không có nội dung"
                        notes.append(f"\n[CẢNH BÁO]: Điều này bị {rel_vn} bởi Điều {mod['id']}.\n[NỘI DUNG ĐIỀU {mod['id']}]:\n{content_full}")
                        
                    for mod in record['outgoing_mods']:
                        if not mod['id']: continue
                        rel_vn = mod['rel']
                        if rel_vn == "AMEND": rel_vn = "SỬA ĐỔI"
                        elif rel_vn == "REPLACE": rel_vn = "THAY THẾ"
                        elif rel_vn == "REPEAL": rel_vn = "BÃI BỎ"
 
                        content_full = mod['content'] if mod['content'] else "Không có nội dung"
                        notes.append(f"\n[LƯU Ý]: Điều này {rel_vn} Điều {mod['id']}.\n[NỘI DUNG ĐIỀU {mod['id']}]:\n{content_full}")
                        
                    if notes:
                        graph_note = "\n".join(notes)
                        new_doc = Document(
                            page_content=graph_note,
                            metadata={"type": "graph_context", "related_to": a_id}
                        )
                        graph_context_docs.append(new_doc)

    except Exception as e:
        print(f"Graph retrieval failed: {e}")
    finally:
        driver.close()
        
    return {"graph_context": graph_context_docs}

def context_fusion(state: GraphState):
    """
    Step 5: Context Fusion (Deterministic)
    Synthesize retrieved documents and graph context into a coherent knowledge base.
    """
    print("---CONTEXT FUSION (Deterministic)---")
    reranked_docs = state["reranked_docs"]
    graph_context = state.get("graph_context", [])
    question = state["question"]
    
    # Map graph context to article IDs
    graph_map = {}
    for g_doc in graph_context:
        related_id = g_doc.metadata.get("related_to")
        if related_id:
            graph_map[related_id] = g_doc.page_content
            
    # Build Structured Context
    context_parts = []
    
    # Part 1: Legal Articles & Relationships
    context_parts.append("# VĂN BẢN PHÁP LUẬT LIÊN QUAN")
    
    if not reranked_docs:
        context_parts.append("(Không tìm thấy văn bản phù hợp)")
    else:
        for i, doc in enumerate(reranked_docs):
            a_id = doc.metadata.get("id", "Không xác định")
            title = doc.metadata.get("title", "")
            content = doc.page_content
            
            # Scores
            r_score = doc.metadata.get("rerank_score")
            score_info = f"(Rerank Score: {r_score:.4f})" if r_score is not None else ""
            
            # Graph Info
            graph_note = graph_map.get(a_id)
            
            # Format Article Block
            article_block = f"""
## Văn bản {i+1}: {a_id}
**Tiêu đề**: {title}
**Điểm số**: {score_info}
**Nội dung**:
{content}
"""
            if graph_note:
                article_block += f"""
**Thông tin bổ sung (Graph)**:
{graph_note}
"""
            context_parts.append(article_block)
            
    fused_context = "\n".join(context_parts)
    
    # Debug print first 2000 chars
    print(f"Fused Context Preview:\n{fused_context[:2000]}...")
        
    return {"fused_context": fused_context}

def generate(state: GraphState):
    """
    Step 6: Generation
    Synthesize answer using LLM.
    """
    print("---GENERATE---")
    fused_context = state["fused_context"]
    question = state["question"]
    
    if not fused_context:
        return {"answer": "I cannot find relevant legal documents to answer your question."}
        
    prompt = ChatPromptTemplate.from_template(
        """Bạn là trợ lý pháp lý hữu ích cho Luật Ngân hàng Việt Nam.
        Trả lời câu hỏi của người dùng dựa *nghiêm ngặt* vào ngữ cảnh được cung cấp.
        
        Câu trả lời của bạn phải tuân theo cấu trúc chính xác sau:
        
        1. **Câu trả lời trực tiếp**: 
           - Trả lời ngắn gọn, súc tích cho câu hỏi.
           - BẮT BUỘC phải trích dẫn văn bản ngay trong câu trả lời (ví dụ: "Theo Điều 1 Thông tư 12/2025/NHNN...").
           - Nếu có thông tin về mối quan hệ (sửa đổi, bổ sung, thay thế), phải đề cập ngay (ví dụ: "Tuy nhiên, nội dung trên đã bị thay thế bởi...").
           
        2. **Căn cứ pháp lý**: 
           - Liệt kê các Điều khoản và Văn bản cụ thể được sử dụng.
           - Với mỗi căn cứ, hãy tóm tắt ngắn gọn nội dung chính của điều khoản đó (không trích nguyên văn dài dòng).
           
        3. **Đồ thị pháp lý**: 
           - Liệt kê tất cả các mối quan hệ văn bản được tìm thấy trong ngữ cảnh.
           - Với mỗi mối quan hệ, hãy mô tả tóm tắt thông tin liên quan (ví dụ: "Thông tư A sửa đổi Điều X của Thông tư B về vấn đề Y").
        
        Quy tắc:
        - Ưu tiên thông tin từ các văn bản CÒN HIỆU LỰC (VALID).
        - Nếu một điều khoản bị THAY THẾ (REPLACE) hoặc BÃI BỎ (REPEAL), hãy cảnh báo người dùng rõ ràng.
        - Trả lời hoàn toàn bằng Tiếng Việt.
        
        Ngữ cảnh:
        {context}
        
        Câu hỏi: {question}
        
        Trả lời:"""
    )
    
    llm = get_llm()
    chain = prompt | llm
    try:
        response = chain.invoke({"context": fused_context, "question": question})
        answer = response.content
    except Exception as e:
        answer = f"Error generating answer: {e}"
        
    return {"answer": answer}

# --- Workflow Construction ---
workflow = StateGraph(GraphState)

workflow.add_node("hybrid_search", hybrid_search)
# workflow.add_node("threshold_validation", threshold_validation)
workflow.add_node("rerank", rerank)
workflow.add_node("relationship_retrieval", relationship_retrieval)
workflow.add_node("context_fusion", context_fusion)
workflow.add_node("generate", generate)

workflow.set_entry_point("hybrid_search")
workflow.add_edge("hybrid_search", "rerank")
# workflow.add_edge("rerank", "threshold_validation")
# workflow.add_edge("threshold_validation", "relationship_retrieval")
workflow.add_edge("rerank", "relationship_retrieval")
workflow.add_edge("relationship_retrieval", "context_fusion")
# workflow.add_edge("context_fusion", END)

workflow.add_edge("context_fusion", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

def generate_graph_image(output_path="rag_graph_rerank.png"):
    """Generate and save the graph visualization."""
    try:
        print("Generating graph image...")
        graph_png = app.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(graph_png)
        print(f"Graph saved to {output_path}")
    except Exception as e:
        print(f"Failed to generate graph image: {e}")
        print("Ensure you have internet access or mermaid-cli installed if running locally.")

if __name__ == "__main__":
    # Generate Visualization
    generate_graph_image()

    # Test Run
    test_question = "Ngân hàng có bắt buộc phải công khai thông tin về Open API trên trang thông tin điện tử không?"
    print(f"Testing with question: {test_question}")
    
    inputs = {"question": test_question}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Finished Node: {key}")
            
    print("\nFinal Answer:")
    if "answer" in value:
        print(value["answer"])
    else:
        print("Generation node disabled. Outputting fused context preview:")
        # print(value.get("fused_context", "")[:500])
