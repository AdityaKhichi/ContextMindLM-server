import os
import tempfile
import traceback
import time

from src.models.index import ProcessingStatus
from src.config.index import appConfig

from src.services.supabase import supabase
from src.services.webScrapper import scrapingbee_client
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


def update_status(document_id: str, status: str, details: dict = None):

    # Get current document 
    result = supabase.table("project_documents").select("processing_details").eq("id", document_id).execute()

    # Start with existing details or empty dict
    current_details = {}

    if result.data and result.data[0]["processing_details"]:
        current_details = result.data[0]["processing_details"]

    
    # Add new details if provided
    if details: 
        current_details.update(details)
    

    # Update document 
    supabase.table("project_documents").update({
        "processing_status": status, 
        "processing_details": current_details
    }).eq("id", document_id).execute()


def process_document(document_id: str):
    try:
        update_status(document_id, ProcessingStatus.PROCESSING)

        doc_result = supabase.table("project_documents").select("*").eq("id", document_id).execute()
        if not doc_result.data:
            raise Exception(
                f"Failed to get project document record with id: {document_id}"
            )
        
        document = doc_result.data[0]
        source_type = document.get('source_type', 'file')

        # step 1: Download and partition 
        update_status(document_id, ProcessingStatus.PARTITIONING)
        elements, elements_summary = download_and_partition(document_id, document)

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
        update_status(document_id, ProcessingStatus.SUMMARISING, {
            "chunking": chunking_metrics
        })

        # step 3: Summarising chunks
        processed_chunks = summarise_chunks(chunks, document_id, source_type)

        # step 4: Vectorization & storing
        update_status(document_id, ProcessingStatus.VECTORIZATION)
        stored_chunk_ids = store_chunks_with_embeddings(document_id, processed_chunks)

        # Mark as completed
        update_status(document_id, ProcessingStatus.COMPLETED)
        print(f"Celery task completed for document: {document_id} with {len(stored_chunk_ids)} chunks")
        

        return {
            "message": "success",
            "document_id": document_id
        }
    
    except Exception as e:
        traceback.print_exc()
        raise e


def download_and_partition(document_id: str, document: dict):
    
    print(f"Downloading and partitioning document {document_id}")

    source_type = document.get("source_type", "file")

    if source_type == "url":
        # Crawl URL 
        url = document["source_url"] 
        
        # Fetch content with ScrapingBee
        response = scrapingbee_client.get(url)
        
        # Save to temp file
        temp_file = os.path.join( tempfile.gettempdir(), f"{document_id}.html" )
        with open(temp_file, 'wb') as f:
            f.write(response.content)
        
        elements = partition_document(temp_file, "html", source_type="url")

    else:
        # Handle file processing
        
        s3_key = document["s3_key"]
        filename = document["filename"]
        file_type = filename.split(".")[-1].lower()

        #  Download to a temporary location 
        # temp_file = f"/tmp/{document_id}.{file_type}"
        temp_file = os.path.join( tempfile.gettempdir(), f"{document_id}.{file_type}" )
        s3_client.download_file(appConfig["s3_bucket_name"], s3_key, temp_file)

        elements = partition_document(temp_file, file_type, source_type="file")
        
    elements_summary = analyze_elements(elements)

    os.remove(temp_file)

    return elements, elements_summary


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
    print("Processing chunks with AI Summarisation...")
    
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
            print(f"     Creating AI summary for mixed content...")
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
    
    print(f"Processed {len(processed_chunks)} chunks")
    return processed_chunks


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
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

         # Retry with exponential backoff
        attempt = 0
        while True:
            try:
                batch_embeddings = openAI["embeddings"].embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                attempt += 1
                print(f"Retry {attempt} for batch {i//batch_size + 1}")
                if attempt >= 3:
                    raise e
                time.sleep(2 ** attempt)
        
        # Store chunks with embeddings
        print("Storing chunks with embeddings in database...")
        stored_chunk_ids = []
        
        for i, (chunk_data, embedding) in enumerate(zip(processed_chunks, all_embeddings)):
            # Add document_id, chunk_index, and embedding
            chunk_data_with_embedding = {
                **chunk_data,
                'document_id': document_id,
                'chunk_index': i,
                'embedding': embedding
            }
            
            result = supabase.table('document_chunks').insert(chunk_data_with_embedding).execute()
            stored_chunk_ids.append(result.data[0]['id'])
        
        print(f"Successfully stored {len(processed_chunks)} chunks with embeddings")
        return stored_chunk_ids
    
    except Exception as e:
        raise Exception(f"Failed to vectorize chunks to store in database: {str(e)}")