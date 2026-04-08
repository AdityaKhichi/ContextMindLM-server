from fastapi import APIRouter, Depends, HTTPException

from src.models.index import ChatCreate

from src.services.supabase import supabase
from src.services.clerkAuth import get_current_user

from src.config.logging import get_logger, set_project_id, set_user_id

logger = get_logger(__name__)


router = APIRouter(
    tags=["chats"]
)  


@router.post("/")
async def create_chat(
    chat: ChatCreate, 
    clerk_id: str = Depends(get_current_user)
):
    set_project_id(chat.project_id)
    set_user_id(clerk_id)
    logger.info("creating_chat", title=chat.title)

    try:
        result = supabase.table("chats").insert({
            "title": chat.title, 
            "project_id": chat.project_id, 
            "clerk_id": clerk_id
        }).execute()

        if not result.data:
            logger.warning( "chat_creation_failed", reason="invalid_data")
            raise HTTPException(
                status_code=422, detail="Failed to create chat - invalid data provided"
            )
        
        chat_id = result.data[0].get("id")
        logger.info("chat_created_successfully", chat_id=chat_id)

        return {
            "message": "Chat created successfully", 
            "data": result.data[0]
        }

    except Exception as e:
        logger.error("chat_creation_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail = f"An internal server error occurred while creating chat: {str(e)}")


@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: str, 
    clerk_id: str = Depends(get_current_user)
):
    set_user_id(clerk_id)
    logger.info("deleting_chat", chat_id=chat_id)

    try:
        # First get the chat to retrieve project_id
        chat_result = (
            supabase.table("chats")
            .select("project_id")
            .eq("id", chat_id)
            .eq("clerk_id", clerk_id)
            .execute()
        )

        if chat_result.data:
            set_project_id(chat_result.data[0].get("project_id"))

        deleted_result = supabase.table("chats").delete().eq("id", chat_id).eq("clerk_id", clerk_id).execute()

        if not deleted_result.data: 
            logger.warning("chat_deletion_failed", chat_id=chat_id, reason="not_found_or_unauthorized")
            raise HTTPException(status_code=404, detail="Chat not found or access denied")

        logger.info("chat_deleted_successfully", chat_id=chat_id)
        return {
            "message": "Chat Deleted Successfully", 
            "data": deleted_result.data[0]
        }

    except Exception as e:
        logger.error("chat_deletion_error", chat_id=chat_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail = f"An internal server error occurred while deleting chat {chat_id}: {str(e)}")
    

@router.get("/{chat_id}")
async def get_chat(
    chat_id: str,
    clerk_id: str = Depends(get_current_user)
):
    set_user_id(clerk_id)
    try:
        result = supabase.table('chats').select('*').eq('id', chat_id).eq('clerk_id', clerk_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Chat not found or you don't have permission to access it")
        
        chat = result.data[0]
        set_project_id(chat.get("project_id"))
        
        # Get messages for this chat
        messages_result = supabase.table('messages').select('*').eq('chat_id', chat_id).order('created_at', desc=False).execute()

        chat['messages'] = messages_result.data or []
        
        return {
            "message": "Chat retrieved successfully",
            "data": chat
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred while getting chat {chat_id}: {str(e)}")

