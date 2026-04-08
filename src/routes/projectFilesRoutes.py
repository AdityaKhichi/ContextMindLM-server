import uuid

from fastapi import APIRouter, Depends, HTTPException

from src.services.clerkAuth import get_current_user
from src.services.supabase import supabase
from src.services.awsS3 import s3_client
from src.services.celery import process_document_ingestion

from src.models.index import FileUploadRequest, ProcessingStatus, UrlAddRequest
from src.config.index import appConfig
from src.utils.index import validate_url

from src.config.logging import get_logger, set_project_id, set_user_id

logger = get_logger(__name__)

router = APIRouter(
    tags=["files"]
)


@router.get("/{project_id}/files")
async def get_project_files(
    project_id: str, 
    clerk_id: str = Depends(get_current_user)
):
    set_project_id(project_id)
    set_user_id(clerk_id)

    try:
        logger.info("fetching_project_files")
        result = supabase.table("project_documents").select("*").eq("project_id", project_id).eq("clerk_id", clerk_id).order("created_at", desc=True).execute()

        logger.info("project_files_retrieved", file_count=len(result.data or []))
        return {
            "message": "Project files retrieved successfully", 
            "data": result.data or []
        }

    except Exception as e:
        logger.error("project_files_retrieval_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail = f"An internal server error occurred while retrieving project {project_id} files: {str(e)}")


@router.post("/{project_id}/files/upload-url")
async def get_upload_presigned_url(
    project_id: str, 
    file_request: FileUploadRequest, 
    clerk_id: str = Depends(get_current_user)
):
    set_project_id(project_id)
    set_user_id(clerk_id)

    try:
        logger.info("generating_upload_url", filename=file_request.filename, file_size=file_request.file_size)
        # Verify project exists and belongs to the user 
        projects_result = supabase.table("projects").select("id").eq("id", project_id).eq("clerk_id", clerk_id).execute()

        if not projects_result.data:
            logger.warning("project_not_found_for_upload")
            raise HTTPException(status_code=400, detail="Project not found or you don't have permission to upload files to this project")
        
        # Generate unique S3 key
        file_extension = file_request.filename.split('.')[-1] if '.' in file_request.filename else ''
        unique_id = str(uuid.uuid4())
        s3_key = f"projects/{project_id}/documents/{unique_id}.{file_extension}"

        # Generate presigned URL (expire in 1 hour)
        presigned_url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': appConfig["s3_bucket_name"],
                'Key': s3_key,
                'ContentType': file_request.file_type
            },
            ExpiresIn=3600  # 1 hour
        )

        if not presigned_url:
            logger.error("presigned_url_generation_failed", s3_key=s3_key)
            raise HTTPException(
                status_code=422,
                detail="Failed to generate upload presigned url",
            )

        # Create database record with pending status
        document_result = supabase.table("project_documents").insert({
            "project_id": project_id,
            'filename': file_request.filename,
            's3_key': s3_key,
            'file_size': file_request.file_size,
            'file_type': file_request.file_type,
            'processing_status': 'uploading',
            'clerk_id': clerk_id 
        }).execute()

        if not document_result.data:
            logger.error("document_record_creation_failed", filename=file_request.filename, reason="no_data_returned")
            raise HTTPException(status_code=500, detail="Failed to create document record - invalid data provided")
        
        logger.info("upload_url_generated_successfully", document_id=document_result.data[0]["id"], s3_key=s3_key)
        return {
            "message": "Upload presigned URL generated successfully",
            "data": {
                "upload_url": presigned_url,
                "s3_key": s3_key,
                "document": document_result.data[0]
            }
        }

    except Exception as e:
        logger.error("upload_url_generation_error", filename=file_request.filename, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail = f"An internal server error occurred while generating upload presigned url for {project_id}: {str(e)}")


@router.post("/{project_id}/files/confirm")
async def confirm_file_upload_to_s3(
    project_id: str, 
    confirm_request: dict, 
    clerk_id: str = Depends(get_current_user)
):
    set_project_id(project_id)
    set_user_id(clerk_id)

    try:
        s3_key = confirm_request.get("s3_key")
        logger.info("confirming_file_upload", s3_key=s3_key)

        if not s3_key:
            logger.warning("s3_key_missing")
            raise HTTPException(status_code=400, detail="s3_key is required")
        
        
        document_verification_result = (
            supabase.table("project_documents")
            .select("id")
            .eq("s3_key", s3_key)
            .eq("project_id", project_id)
            .eq("clerk_id", clerk_id)
            .execute()
        )

        if not document_verification_result.data:
            logger.warning("file_not_found_for_confirmation", s3_key=s3_key)
            raise HTTPException(
                status_code=404,
                detail="File not found or you don't have permission to confirm upload to S3 for this file",
            )

        # Update document status
        result = supabase.table("project_documents").update({
            "processing_status": ProcessingStatus.QUEUED
        }).eq("s3_key", s3_key).execute()

        document_id = result.data[0]["id"]

        # Start background preprocessing of the current file with Celery
        task_result = process_document_ingestion.delay(document_id)
        task_id = task_result.id
        logger.info("rag_ingestion_task_queued", document_id=document_id, task_id=task_id)

        #Store this task ID so that we can track later if needed
        document_update_result = supabase.table("project_documents").update({
            "task_id": task_id
        }).eq("id", document_id).execute()

        if not document_update_result.data:
            logger.error("task_id_update_failed", document_id=document_id, task_id=task_id, reason="no_data_returned")
            raise HTTPException(
                status_code=422,
                detail="Failed to update project document record with task_id",
            )


        # Return JSON 
        logger.info("file_upload_confirmed_successfully", document_id=document_id, task_id=task_id)
        return {
            "message": "Upload to S3 confirmed, background Pre-Processing started", 
            "data": document_update_result.data[0]
        }

    except Exception as e:
        logger.error("file_confirmation_error", s3_key=s3_key, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail = f"An internal server error occurred while confirming upload to S3 for {project_id}: {str(e)}")
    

@router.post("/{project_id}/urls")
async def add_website_url(
    project_id: str, 
    url_request: UrlAddRequest, 
    clerk_id: str = Depends(get_current_user)
):
    set_project_id(project_id)
    set_user_id(clerk_id)

    try:
        # URL validation
        url = url_request.url.strip()

        if not url.startswith(('http://', 'https://')):
            url = "https://" + url

        logger.info("processing_url", url=url)
        if not validate_url(url):
            logger.warning("invalid_url", url=url)
            raise HTTPException(
                status_code=400,
                detail="Invalid URL",
            )


        result = supabase.table("project_documents").insert({
            "project_id": project_id,
            'filename': url,
            's3_key': "",
            'file_size': 0,
            'file_type': 'text/html',
            'processing_status': ProcessingStatus.QUEUED,
            'clerk_id': clerk_id, 
            "source_url": url, 
            "source_type": "url"
        }).execute()

        if not result.data:
            logger.error("url_document_creation_failed", url=url, reason="no_data_returned")
            raise HTTPException(status_code=500, detail="Failed to create URL record - invalid data provided")
        
        document_id = result.data[0]["id"]

        #Start background preprocessing of the current file with celery
        task_result = process_document_ingestion.delay(document_id)
        task_id = task_result.id
        logger.info("url_ingestion_task_queued", document_id=document_id, task_id=task_id, url=url)

        #Store this task ID so that we can track later if needed
        document_update_result = supabase.table("project_documents").update({
            "task_id": task_id
        }).eq("id", document_id).execute()

        if not document_update_result.data:
            logger.error("url_task_id_update_failed", document_id=document_id, task_id=task_id, reason="no_data_returned")
            raise HTTPException(
                status_code=422,
                detail="Failed to update project document record with task_id",
            )


        logger.info("url_processed_successfully", document_id=document_id, url=url, task_id=task_id)
        return {
            "message": "URL added successfully, background Pre-Processing started", 
            "data": result.data[0]
        }
        
    except Exception as e:
        logger.error("url_processing_error", url=url, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred while processing urls for {project_id}: {str(e)}")


@router.delete("/{project_id}/files/{file_id}")
async def delete_file(
    project_id: str, 
    file_id: str, 
    clerk_id: str = Depends(get_current_user)
):
    set_project_id(project_id)
    set_user_id(clerk_id)

    try:
        logger.info("deleting_document", file_id=file_id)
        # Get the file record (this also verifies ownership via clerk_id)
        file_result = supabase.table("project_documents").select("*").eq("id", file_id).eq("clerk_id", clerk_id).eq("project_id", project_id).execute()

        if not file_result.data:
            logger.warning("document_not_found_for_deletion", file_id=file_id)
            raise HTTPException(status_code=404, detail="File not found or access denied")
        
        
        s3_key = file_result.data[0]["s3_key"]


        # Delete from S3 (only for actual files, not URLs)
        if s3_key:
            s3_client.delete_object(Bucket=appConfig["s3_bucket_name"], Key=s3_key)
            logger.info("deleting_from_s3", file_id=file_id, s3_key=s3_key)


        # Delete document record from DB 
        delete_result = (
            supabase.table("project_documents")
            .delete()
            .eq("id", file_id)
            .execute()
        )

        if not delete_result.data:
            logger.error("document_deletion_failed", file_id=file_id, reason="no_data_returned")
            raise HTTPException(status_code=500, detail="Failed to delete file")

        logger.info("document_deleted_successfully", file_id=file_id)
        return {
            "message": "File deleted successfully", 
            "data": delete_result.data[0]
        }    

    except Exception as e:
        logger.error("document_deletion_error", file_id=file_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred while deleting project document {file_id} for {project_id}: {str(e)}")
    

@router.get("/{project_id}/files/{file_id}/chunks")
async def get_document_chunks(
    project_id: str,
    file_id: str,
    clerk_id: str = Depends(get_current_user)
):
    set_project_id(project_id)
    set_user_id(clerk_id)

    try:
        logger.info("fetching_document_chunks", file_id=file_id)
        project_result = supabase.table('projects').select('id').eq('id', project_id).eq('clerk_id', clerk_id).execute()
        
        if not project_result.data:
            logger.warning("document_not_found_for_chunks", file_id=file_id)
            raise HTTPException(status_code=404, detail="Project not found or access denied")
        
        doc_result = supabase.table('project_documents').select('id').eq('id', file_id).eq('project_id', project_id).execute()
        
        if not doc_result.data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        chunks_result = supabase.table('document_chunks').select('*').eq('document_id', file_id).order('chunk_index').execute()
        
        logger.info("document_chunks_retrieved", file_id=file_id, chunk_count=len(chunks_result.data or []))
        return {
            "message": "Document chunks retrieved successfully",
            "data": chunks_result.data or []
        }

    except Exception as e:
        logger.error("document_chunks_retrieval_error", file_id=file_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred while getting project document chunks for {file_id} for {project_id}: {str(e)}")
    
    