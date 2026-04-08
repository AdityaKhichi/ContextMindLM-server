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

from src.config.logging import get_logger, set_project_id

logger = get_logger(__name__)


def retrieve_context(project_id, user_query):
    set_project_id(project_id)
    try:
        # Load project settings
        # to know: chunk size, similarity threshold, etc.
        settings = load_project_settings(project_id)

        # Get document IDs for this project
        document_ids = get_document_ids(project_id)
        logger.info("documents_found", document_count=len(document_ids))

        strategy = settings['rag_strategy']
        logger.info("project_settings_retrieved", strategy=strategy, final_context_size=settings["final_context_size"])
        
        # Perform search using PostgreSQL functions 
        if strategy == 'basic':
            chunks = vector_search(user_query, document_ids, settings) 
            logger.info("vector_search_completed", chunks_found=len(chunks))

        elif strategy == 'hybrid':
            chunks = hybrid_search(user_query, document_ids, settings)
            logger.info("hybrid_search_completed", chunks_found=len(chunks))

        # Step 6: Multi-query vector search
        elif strategy == "multi-query-vector":
            chunks = multi_query_vector_search(user_query, document_ids, settings)
            logger.info("multi_query_vector_search_completed", chunks_found=len(chunks))

        # Step 7: Multi-query hybrid search
        elif strategy == 'multi-query-hybrid':
            chunks = multi_query_hybrid_search(user_query, document_ids, settings)
            logger.info("multi_query_hybrid_search_completed", chunks_found=len(chunks))


        # Step 8: Selecting top k chunks
        chunks = chunks[: settings["final_context_size"]]
        logger.info("chunks_limited", final_chunk_count=len(chunks))

        # Step 9: Build the context from the retrieved chunks and format them into a structured context with citations.
        texts, images, tables, citations = build_context(chunks)
        logger.info("retrieval_completed", texts_count=len(texts), images_count=len(images), tables_count=len(tables), citations_count=len(citations))

        return texts, images, tables, citations
    except Exception as e:
        logger.error("retrieval_failed", error=str(e), exc_info=True)
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

    logger.info("hybrid_search_results", vector_count=len(vector_results), keyword_count=len(keyword_results))
    
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


def multi_query_vector_search(user_query, document_ids, project_settings):
    queries = generate_query_variations(user_query, project_settings["number_of_queries"])
    logger.info("query_variations_generated", query_count=len(queries))

    all_chunks = []
    for index, query in enumerate(queries):
        chunks = vector_search(query, document_ids, project_settings)
        all_chunks.append(chunks)
        logger.info("query_variation_search", query_num=f"{index+1}/{len(queries)}", query=query, chunks_found=len(chunks))

    final_chunks = rrf_rank_and_fuse(all_chunks)
    logger.info("rrf_fusion_completed", final_chunks_count=len(final_chunks))
    return final_chunks


def multi_query_hybrid_search(user_query, document_ids, project_settings):
    queries = generate_query_variations(user_query, project_settings["number_of_queries"])
    logger.info("query_variations_generated_hybrid", query_count=len(queries))

    all_chunks = []
    for index, query in enumerate(queries):
        chunks = hybrid_search(query, document_ids, project_settings)
        all_chunks.append(chunks)
        logger.info("hybrid_query_variation_search", query_num=f"{index+1}/{len(queries)}", query=query, chunks_found=len(chunks))

    final_chunks = rrf_rank_and_fuse(all_chunks)
    logger.info("rrf_fusion_completed_hybrid", final_chunks_count=len(final_chunks))
    return final_chunks