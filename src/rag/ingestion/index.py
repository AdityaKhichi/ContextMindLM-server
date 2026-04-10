import os
import tempfile
import traceback
import time

from src.models.index import ProcessingStatus
from src.config.index import appConfig

from src.services.supabase import supabase
from src.services.webScrapper import firecrawl_client
from src.services.llm import openAI
from src.services.awsS3 import s3_client

from src.rag.ingestion.utils import (
    partition_document,
    analyze_elements,
    separate_content_types,
    get_page_number,
    create_ai_summary,
)

from unstructured.chunking.title import chunk_by_title

from src.config.logging import get_logger, set_project_id

logger = get_logger(__name__)


def update_status(document_id: str, status: str, details: dict = None):
    logger.info(
        "updating_document_status",
        document_id=document_id,
        status=status.value,
        has_details=details is not None
    )

    try:
        # Get current document 
        result = supabase.table("project_documents").select("processing_details").eq("id", document_id).execute()

        if not result.data:
            logger.error(
                "document_not_found",
                document_id=document_id,
                status=status.value
            )
            raise Exception(
                f"Failed to get project document record with id: {document_id}"
            )

        # Start with existing details or empty dict
        current_details = {}

        if result.data and result.data[0]["processing_details"]:
            current_details = result.data[0]["processing_details"]

        
        # Add new details if provided
        if details: 
            current_details.update(details)
            logger.debug(
                "merged_processing_details",
                document_id=document_id,
                details_keys=list(details.keys())
            )
        

        # Update document 
        update_result = supabase.table("project_documents").update({
            "processing_status": status, 
            "processing_details": current_details
        }).eq("id", document_id).execute()

        if not update_result.data:
            logger.error(
                "status_update_failed",
                document_id=document_id,
                status=status.value
            )
            raise Exception(
                f"Failed to update project document record with id: {document_id}"
            )

        logger.info(
            "document_status_updated_successfully",
            document_id=document_id,
            status=status.value,
            details_count=len(current_details)
        )
    except Exception as e:
        logger.error(
            "update_status_error",
            document_id=document_id,
            status=status.value,
            error=str(e),
            exc_info=True
        )
        raise Exception(f"Failed to update status in database: {str(e)}")



def process_document(document_id: str):
    logger.info("document_processing_started", document_id=document_id)

    try:
        update_status(document_id, ProcessingStatus.PROCESSING)

        doc_result = supabase.table("project_documents").select("*").eq("id", document_id).execute()
        if not doc_result.data:
            logger.error("document_not_found", document_id=document_id)
            raise Exception(
                f"Failed to get project document record with id: {document_id}"
            )
        
        document = doc_result.data[0]
        set_project_id(document["project_id"])
        logger.info("document_retrieved", document_id=document_id, source_type=document.get("source_type"))

        # step 1: Download and partition 
        update_status(document_id, ProcessingStatus.PARTITIONING)
        elements, elements_summary = download_and_partition(document_id, document)
        logger.info("partitioning_completed", document_id=document_id, elements_summary=elements_summary)

        update_status(document_id, ProcessingStatus.CHUNKING, {
            "partitioning": {
                "elements_found": elements_summary
            }
        })

        tables = sum(1 for e in elements if e.category == "Table")
        images = sum(1 for e in elements if e.category == "Image")
        text_elements = sum(1 for e in elements if e.category in ["NarrativeText", "Title", "Text"])
        print(f"Extracted: {tables} tables, {images} images, {text_elements} text elements")


        # step 2: Chunk elements
        chunks, chunking_metrics = chunk_elements(elements)
        logger.info("chunking_completed", document_id=document_id, total_chunks=chunking_metrics["total_chunks"])
        update_status(document_id, ProcessingStatus.SUMMARISING, {
            "chunking": chunking_metrics
        })

        # step 3: Summarising chunks
        processed_chunks = summarise_chunks(chunks, document_id, source_type)
        logger.info("summarization_completed", document_id=document_id, chunks_count=len(processed_chunks))

        # step 4: Vectorization & storing
        update_status(document_id, ProcessingStatus.VECTORIZATION)
        stored_chunk_ids = store_chunks_with_embeddings(document_id, processed_chunks)
        logger.info("vectorization_completed", document_id=document_id, stored_chunks=len(stored_chunk_ids))

        # Mark as completed
        update_status(document_id, ProcessingStatus.COMPLETED)
        logger.info("document_processing_completed", document_id=document_id, chunks_created=len(processed_chunks))
        print(f"Celery task completed for document: {document_id} with {len(stored_chunk_ids)} chunks")
        

        return {
            "message": "success",
            "document_id": document_id
        }
    
    except Exception as e:
        logger.error("document_processing_failed", document_id=document_id, error=str(e), exc_info=True)
        traceback.print_exc()
        raise e


def download_and_partition(document_id: str, document: dict):
    
    try:
        source_type = document.get("source_type", "file")

        if source_type == "url":
            # Crawl URL 
            url = document["source_url"] 
            
            # Fetch content with ScrapingBee
            logger.info("crawling_url", document_id=document_id, url=url)
            response = firecrawl_client.scrape(url, formats=["html"])
            
            # Save to temp file
            temp_file = os.path.join( tempfile.gettempdir(), f"{document_id}.html" )
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(response.html)
            logger.info("url_crawl_completed", document_id=document_id)
            
            elements = partition_document(temp_file, "html", source_type="url")
            logger.info("elements_analyzed", document_id=document_id, elements_count=len(elements))

        else:
            # Handle file processing
            s3_key = document["s3_key"]
            filename = document["filename"]
            file_type = filename.split(".")[-1].lower()

            #  Download to a temporary location 
            # temp_file = f"/tmp/{document_id}.{file_type}"
            temp_file = os.path.join( tempfile.gettempdir(), f"{document_id}.{file_type}" )

            logger.info("downloading_from_s3", document_id=document_id, s3_key=s3_key, file_type=file_type)
            s3_client.download_file(appConfig["s3_bucket_name"], s3_key, temp_file)
            logger.info("s3_download_completed", document_id=document_id)

            elements = partition_document(temp_file, file_type, source_type="file")
            
        elements_summary = analyze_elements(elements)

        os.remove(temp_file)

        return elements, elements_summary

    except Exception as e:
        logger.error("download_and_partition_failed", document_id=document_id, error=str(e), exc_info=True)
        raise Exception(f"Failed in Step 1 to download content and partition elements: {str(e)}")


def chunk_elements(elements):

    print("Creating smart chunks...")
    
    chunks = chunk_by_title(
        elements, # The parsed PDF elements from previous step
        max_characters=3000, # Hard limit - never exceed 3000 characters per chunk
        new_after_n_chars=2400, # Try to start a new chunk after 2400 characters
        combine_text_under_n_chars=500 # Merge tiny chunks under 500 chars with neighbors
    )

    # Collect chunking metrics 
    total_chunks = len(chunks)

    chunking_metrics = {
        "total_chunks": total_chunks
    }

    print(f"Created {total_chunks} chunks from {len(elements)} elements")

    return chunks, chunking_metrics


def summarise_chunks(chunks, document_id, source_type="file"):
    try:
        processed_chunks = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            current_chunk = i + 1
            
            # Update progress directly
            update_status(document_id, ProcessingStatus.SUMMARISING, {
                "summarising": {
                    "current_chunk": current_chunk,
                    "total_chunks": total_chunks
                }
            })
            
            # Extract content from the chunk
            content_data = separate_content_types(chunk, source_type)

            # Debug prints
            print(f"     Types found: {content_data['types']}")
            print(f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")
            
            # Decide if we need AI summarisation
            if content_data['tables'] or content_data['images']:
                enhanced_content = create_ai_summary( 
                    content_data['text'], 
                    content_data['tables'], 
                    content_data['images']
                )
            else:
                enhanced_content = content_data['text']
            
            # Build the original_content structure
            original_content = {'text': content_data['text']}
            if content_data['tables']:
                original_content['tables'] = content_data['tables']
            if content_data['images']:
                original_content['images'] = content_data['images']
            
            # Create processed chunk with all data
            processed_chunk = {
                'content': enhanced_content,
                'original_content': original_content, 
                'type': content_data['types'],
                'page_number': get_page_number(chunk, i),
                'char_count': len(enhanced_content)
            }
            
            processed_chunks.append(processed_chunk)
            
        return processed_chunks
    except Exception as e:
        raise Exception(f"Failed to summarise chunks: {str(e)}")


def store_chunks_with_embeddings(document_id: str, processed_chunks: list):
    print("Generating embeddings and storing chunks...")
    
    if not processed_chunks:
        print(" No chunks to process")
        return []
    
    # Generate embeddings for all chunks
    print(f"Generating embeddings for {len(processed_chunks)} chunks...")
    
    try:
        # Extract content for embedding generation
        texts = [chunk_data['content'] for chunk_data in processed_chunks]
        
        # Generate embeddings in batches to avoid API limits
        batch_size = 10
        all_embeddings = []
        logger.info("vectorization_started", document_id=document_id, total_chunks=len(texts), batch_size=batch_size)
        
        for i in range(0, len(texts), batch_size):
            end = i + batch_size
            batch_texts = texts[i:end]
            batch_num = (i // batch_size) + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size

         # Retry with exponential backoff
        attempt = 0
        while True:
            try:
                batch_embeddings = openAI["embeddings"].embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
                logger.info("batch_vectorized", document_id=document_id, batch=f"{batch_num}/{total_batches}", chunks_in_batch=len(batch_texts))
                break
            except Exception as e:
                attempt += 1
                if attempt >= 3:
                    logger.error("vectorization_batch_failed", document_id=document_id, batch=batch_num, attempt=attempt, error=str(e), exc_info=True)
                    raise e
                logger.warning("vectorization_retry", document_id=document_id, batch=batch_num, attempt=attempt, wait_seconds=2 ** attempt)
                time.sleep(2 ** attempt)
        
        # Store chunks with embeddings
        stored_chunk_ids = []
        chunk_embedding_pairs = list(zip(processed_chunks, all_embeddings))
        logger.info("storing_chunks_started", document_id=document_id, total_chunks=len(chunk_embedding_pairs))
        
        for i, (chunk_data, embedding) in enumerate(chunk_embedding_pairs):
            # Add document_id, chunk_index, and embedding
            chunk_data_with_embedding = {
                **chunk_data,
                'document_id': document_id,
                'chunk_index': i,
                'embedding': embedding
            }
            
            result = supabase.table('document_chunks').insert(chunk_data_with_embedding).execute()
            stored_chunk_ids.append(result.data[0]['id'])
        
        logger.info("chunks_stored_successfully", document_id=document_id, stored_count=len(stored_chunk_ids))
        return stored_chunk_ids
    
    except Exception as e:
        logger.error("vectorization_and_storage_failed", document_id=document_id, error=str(e), exc_info=True)
        raise Exception(f"Failed to vectorize chunks to store in database: {str(e)}")