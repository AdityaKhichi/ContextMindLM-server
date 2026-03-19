from celery import Celery

from src.config.index import appConfig
from src.rag.ingestion.index import process_document


celery_app = Celery(
    "document_processor", #Name of celery app
    broker=appConfig["redis_url"], #Where tasks are queued
    backend=appConfig["redis_url"] #Where results are stored
)


@celery_app.task
def process_document_ingestion(document_id: str):
    try:
        process_document_result = process_document(document_id)
        return (
            f"Document {process_document_result['document_id']} processed successfully"
        )
    except Exception as e:
        return f"Failed to process document {document_id}: {str(e)}"