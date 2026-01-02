import os
import json
import csv
import time
import numpy as np
from typing import List, Dict, Any, Set
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL")
SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL", "BM25")

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
DOCUMENTS_FILE = os.path.join(DATA_DIR, "sbv_legal_documents.json")
ARTICLES_FILE = os.path.join(DATA_DIR, "sbv_legal_articles.json")
CSV_FILE = os.path.join(DATA_DIR, "graph_extraction_final.csv")

# BM25SparseEmbedding class removed (using model.common)

def clear_database(driver):
    print("Clearing Neo4j database...")
    with driver.session() as session:
        # Use client-side iterative deletion to avoid OOM
        print("Deleting relationships...")
        total_rels = 0
        while True:
            result = session.run("MATCH ()-[r]->() WITH r LIMIT 1000 DELETE r RETURN count(r) as count")
            count = result.single()["count"]
            total_rels += count
            print(f"Deleted {count} relationships (Total: {total_rels})...")
            if count == 0:
                break
        
        print("Deleting nodes...")
        total_nodes = 0
        while True:
            result = session.run("MATCH (n) WITH n LIMIT 1000 DELETE n RETURN count(n) as count")
            count = result.single()["count"]
            total_nodes += count
            print(f"Deleted {count} nodes (Total: {total_nodes})...")
            if count == 0:
                break
        
        # Drop indexes to be safe
        try:
            session.run("DROP INDEX article_embedding_index IF EXISTS")
            session.run("DROP INDEX concept_embedding_index IF EXISTS")
            session.run("DROP INDEX article_fulltext_index IF EXISTS")
            session.run("DROP INDEX concept_fulltext_index IF EXISTS")
            # Also drop constraints
            session.run("DROP CONSTRAINT ON (d:DOCUMENT) ASSERT d.id IS UNIQUE IF EXISTS") # Old syntax fallback or ignore if not exists
            # Newer syntax usually: DROP CONSTRAINT constraint_name
            # But we can just rely on the create if not exists later.
            # Ideally we should list and drop constraints, but for now the node deletion is the critical part.
        except Exception as e:
            print(f"Warning dropping indexes: {e}")

def setup_constraints_and_indexes(driver, dense_dim):
    print("Setting up constraints and indexes...")
    with driver.session() as session:
        # Constraints
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:DOCUMENT) REQUIRE d.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:ARTICLE) REQUIRE a.id IS UNIQUE")
        
        # Vector Indexes (Dense)
        print(f"Creating vector index for ARTICLE with dim {dense_dim}...")
        session.run(f"""
        CREATE VECTOR INDEX article_embedding_index IF NOT EXISTS
        FOR (a:ARTICLE) ON (a.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {dense_dim},
            `vector.similarity_function`: 'cosine'
        }}}}
        """)
        
        # Fulltext Indexes (for hybrid/keyword search)
        print("Creating fulltext indexes...")
        session.run("""
        CREATE FULLTEXT INDEX article_fulltext_index IF NOT EXISTS
        FOR (a:ARTICLE) ON EACH [a.content, a.title]
        """)

def build_graph():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    try:
        # 1. Clear DB
        clear_database(driver)
        
        # 2. Load Data
        print("Loading data...")
        with open(DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
            documents_data = json.load(f)
        
        with open(ARTICLES_FILE, 'r', encoding='utf-8') as f:
            articles_data = json.load(f)
            
        # Map documents by law_id
        documents_map = {d.get('law_id'): d for d in documents_data if d.get('law_id')}
        
        # 3. Create Document and Article Nodes
        print(f"Creating {len(documents_map)} Documents and {len(articles_data)} Articles...")
        with driver.session() as session:
            # Create Documents
            for law_id, doc in documents_map.items():
                session.run("""
                MERGE (d:DOCUMENT {id: $id})
                SET d.title = $title,
                    d.vbpl_id = $vbpl_id,
                    d.status = $status,
                    d.effective_date = $effective_date,
                    d.expired_date = $expired_date
                """, {
                    "id": law_id,
                    "title": doc.get("document_title"),
                    "vbpl_id": doc.get("vbpl_id"),
                    "status": doc.get("document_status"),
                    "effective_date": doc.get("effective_date"),
                    "expired_date": doc.get("expired_date")
                })
                
            # Create Articles and link to Documents
            for art in articles_data:
                if not art.get("id") or not art.get("law_id"):
                    continue
                
                if len(art.get("id")) > 512:
                    print(f"Skipping long article ID: {art.get('id')[:50]}...")
                    continue
                
                session.run("""
                MERGE (a:ARTICLE {id: $id})
                SET a.title = $title,
                    a.content = $content,
                    a.chapter = $chapter,
                    a.section = $section,
                    a.law_id = $law_id
                WITH a
                MATCH (d:DOCUMENT {id: $law_id})
                MERGE (a)-[:BELONGS_TO]->(d)
                """, {
                    "id": art.get("id"),
                    "title": art.get("article_title"),
                    "content": art.get("content"),
                    "chapter": art.get("chapter"),
                    "section": art.get("section"),
                    "law_id": art.get("law_id")
                })

        # 4. Process CSV for Relationships (Article-Article only)
        print(f"Processing {CSV_FILE}...")
        relationships = []
        
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                source_id = row.get('source_id')
                target_id = row.get('target_id')
                target_type = row.get('target_type')
                relation_type = row.get('relation_type')
                
                if not source_id or not target_id:
                    continue
                
                # Only process Article-Article relationships
                if target_type == 'ARTICLE':
                    relationships.append(row)
        
        # Create Relationships
        print(f"Creating {len(relationships)} relationships...")
        with driver.session() as session:
            batch_size = 1000
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i:i+batch_size]
                for rel in batch:
                    source = rel['source_id']
                    target = rel['target_id']
                    rtype = rel['relation_type']
                    
                    query = f"""
                    MATCH (s:ARTICLE {{id: $source}})
                    MATCH (t:ARTICLE {{id: $target}})
                    MERGE (s)-[:{rtype}]->(t)
                    """
                    
                    session.run(query, {"source": source, "target": target})
                print(f"Processed {min(i+batch_size, len(relationships))}/{len(relationships)} rels")

        # 5. Embeddings (Dense + Sparse)
        print("Generating embeddings...")
        
        # --- ARTICLES ---
        print("Processing Articles Embeddings...")
        
        # Load shared BM25 model (created by build_vector_db.py)
        print(f"Loading Shared BM25 Model...")
        from ai_models.common import get_sparse_model
        try:
            sparse_model_articles = get_sparse_model()
            print("BM25 Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load BM25 model: {e}")
            print("Please run build_vector_db.py first to generate the model.")
            return
        
        print(f"Loading Dense Model: {EMBEDDING_MODEL_NAME}...")
        dense_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Get dense dimension
        sample_dense = dense_model.encode("test")
        dense_dim = len(sample_dense)
        
        # Setup Indexes
        setup_constraints_and_indexes(driver, dense_dim)
        
        # Update Articles
        print("Updating Articles with embeddings...")
        with driver.session() as session:
            # Fetch all articles with content
            result = session.run("MATCH (a:ARTICLE) WHERE a.content IS NOT NULL RETURN a.id AS id, a.content AS content")
            articles_to_update = [{"id": r["id"], "content": r["content"]} for r in result]
            
            # Batch process
            for i in range(0, len(articles_to_update), 100):
                batch = articles_to_update[i:i+100]
                texts = [b["content"] for b in batch]
                ids = [b["id"] for b in batch]
                
                dense_vecs = dense_model.encode(texts)
                
                for j, text in enumerate(texts):
                    # Transform using Article BM25 Model
                    # model.common.BM25SparseEmbedding.transform_document returns Qdrant SparseVector
                    sparse_vec_obj = sparse_model_articles.transform_document(text)
                    
                    # Convert to dict for JSON storage: {index: score}
                    sparse_vec_dict = {str(idx): float(val) for idx, val in zip(sparse_vec_obj.indices, sparse_vec_obj.values)}
                    sparse_json = json.dumps(sparse_vec_dict)
                    
                    session.run("""
                    MATCH (a:ARTICLE {id: $id})
                    CALL db.create.setNodeVectorProperty(a, 'embedding', $dense)
                    SET a.sparse_embedding = $sparse
                    """, {
                        "id": ids[j],
                        "dense": dense_vecs[j].tolist(),
                        "sparse": sparse_json
                    })
                print(f"Updated {min(i+100, len(articles_to_update))} Articles")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        driver.close()
        print("Done.")

if __name__ == "__main__":
    build_graph()
