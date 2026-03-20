from typing import List, Dict
from fastapi import HTTPException

from langchain_core.messages import SystemMessage, HumanMessage

from src.services.llm import openAI
from src.services.supabase import supabase

from src.rag.retrieval.utils import (
    load_project_settings,
    get_document_ids,
    build_context,
    rrf_rank_and_fuse
)
from src.models.index import QueryVariations


def retrieve_context(project_id, user_query):
    try:
        # Load project settings
        # to know: chunk size, similarity threshold, etc.
        settings = load_project_settings(project_id)

        # Get document IDs for this project
        document_ids = get_document_ids(project_id)
        print("Found document IDs: ", len(document_ids))

        strategy = settings['rag_strategy']
        print(f"\nRAG STRATEGY: {strategy.upper()}")
        
        # Perform search using PostgreSQL functions 
        if strategy == 'basic':
            chunks = vector_search(user_query, document_ids, settings) 
            print(f"Retrieved {len(chunks)} relevant chunks from vector search") 

        elif strategy == 'hybrid':
            print("Executing: Hybrid Search (Vector + Keyword)") 
            chunks = hybrid_search(user_query, document_ids, settings)
            print(f"Hybrid search returned: {len(chunks)} chunks")

        # Step 6: Multi-query vector search
        elif strategy == "multi-query-vector":
            print(f"Executing: Multi-Query Vector Search ({settings['number_of_queries']} queries)")
            queries = generate_query_variations(user_query, settings['number_of_queries'])
            print(f"Generated queries: {queries}")
            all_results = []
            for i, q in enumerate(queries):
                results = vector_search(q, document_ids, settings)
                print(f"Query {i+1} '{q}' returned: {len(results)} chunks")
                all_results.append(results)
            chunks = rrf_rank_and_fuse(all_results)
            print(f"RRF fusion returned: {len(chunks)} chunks")

        # Step 7: Multi-query hybrid search
        elif strategy == 'multi-query-hybrid':
            print(f"Executing: Multi-Query Hybrid Search ({settings['number_of_queries']} queries, Vector + Keyword)")
            queries = generate_query_variations(user_query, settings['number_of_queries'])
            print(f"Generated queries: {queries}")
            
            # Stage 1: Per-query hybrid fusion
            all_hybrid_results = []
            for i, q in enumerate(queries):
                print(f"\n  Query {i+1}: '{q}'")
                
                # Use the existing hybrid_search function which handles weights
                hybrid_results = hybrid_search(q, document_ids, settings)
                
                print(f"Hybrid fusion returned: {len(hybrid_results)} chunks")
                
                all_hybrid_results.append(hybrid_results)
            
            # Stage 2: Cross-query fusion (equal weights across queries by default)
            print(f"\nFinal RRF fusion across {len(all_hybrid_results)} queries")
            chunks = rrf_rank_and_fuse(all_hybrid_results)
            print(f"Final result: {len(chunks)} chunks")


        # Step 8: Selecting top k chunks
        chunks = chunks[: settings["final_context_size"]]

        # Step 9: Build the context from the retrieved chunks and format them into a structured context with citations.
        texts, images, tables, citations = build_context(chunks)

        return texts, images, tables, citations
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed in RAG's Retrieval: {str(e)}"
        )


def vector_search(query: str, document_ids: List[str], settings: dict) -> List[Dict]:
    query_embedding = openAI["embeddings"].embed_query(query)
    
    result = supabase.rpc('vector_search_document_chunks', {
        'query_embedding': query_embedding,
        'filter_document_ids': document_ids,
        'match_threshold': settings['similarity_threshold'],
        'chunks_per_search': settings['chunks_per_search']
    }).execute()
    
    return result.data if result.data else []


def hybrid_search(query: str, document_ids: List[str], settings: dict) -> List[Dict]:
    """Execute hybrid search by combining vector and keyword results"""

    # Get results from both search methods
    vector_results = vector_search(query, document_ids, settings)
    keyword_results = keyword_search(query, document_ids, settings)

    print(f"📈 Vector search returned: {len(vector_results)} chunks")
    print(f"📈 Keyword search returned: {len(keyword_results)} chunks")
    
    # Combine using RRF with configured weights
    return rrf_rank_and_fuse(
        [vector_results, keyword_results], 
        [settings['vector_weight'], settings['keyword_weight']]
    )


def keyword_search(query: str, document_ids: List[str], settings: dict) -> List[Dict]:
    result = supabase.rpc('keyword_search_document_chunks', {
        'query_text': query,
        'filter_document_ids': document_ids,
        'chunks_per_search': settings['chunks_per_search']
    }).execute()
    
    return result.data if result.data else []


def generate_query_variations(query: str, num_queries: int = 3) -> List[str]:
    system_prompt = f"""Generate {num_queries-1} alternative ways to phrase this question for document search. 
    Use different keywords and synonyms while maintaining the same intent. Return exactly {num_queries-1} variations."""

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Original query: {query}")
        ]
        
        structured_llm = openAI["chat_llm"].with_structured_output(QueryVariations)
        result = structured_llm.invoke(messages)
        
        return [query] + result.queries[:num_queries-1]
    except Exception:
        return [query]
