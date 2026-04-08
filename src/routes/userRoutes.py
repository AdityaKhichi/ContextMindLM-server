from fastapi import APIRouter, HTTPException
from src.services.supabase import supabase

from src.config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    tags=["users"]
)

@router.post("/create-user")
async def create_user_from_clerk_webhook(clerk_webhook_data: dict):
    try:
        logger.info("webhook_received", event_type=clerk_webhook_data.get("type") if isinstance(clerk_webhook_data, dict) else None)
        # Step 1: Validate webhook payload structure
        if not isinstance(clerk_webhook_data, dict):
            logger.warning("invalid_webhook_payload", payload_type=type(clerk_webhook_data).__name__)
            raise HTTPException(
                status_code=400, 
                detail="Invalid webhook payload format"
            )
        
        # Step 2: Check event type
        event_type = clerk_webhook_data.get("type")
        if event_type != "user.created":
            logger.info("event_type_ignored", event_type=event_type)
            # Return success for other events (don't retry)
            return {
                "success": True,
                "message": f"Event type '{event_type}' ignored"
            }
        
        # Step 3: Extract and validate user data
        user_data = clerk_webhook_data.get("data")
        if not user_data or not isinstance(user_data, dict):
            logger.warning("invalid_user_data", has_data=bool(user_data), data_type=type(user_data).__name__ if user_data else None)
            raise HTTPException(
                status_code=400,
                detail="Missing or invalid user data in webhook payload"
            )
        
        # Step 4: Extract and validate clerk_id
        clerk_id = user_data.get("id")
        if not clerk_id or not isinstance(clerk_id, str):
            logger.warning("invalid_clerk_id", has_id=bool(clerk_id), id_type=type(clerk_id).__name__ if clerk_id else None)
            raise HTTPException(
                status_code=400, 
                detail="Missing or invalid clerk_id in user data"
            )
        
        logger.info("creating_user", user_id=clerk_id)
        # Step 5: Check if user already exists (webhook idempotency)
        existing_user = (
            supabase.table("users")
            .select("clerk_id")
            .eq("clerk_id", clerk_id)
            .execute()
        )
        
        if existing_user.data:
            logger.info("user_already_exists", user_id=clerk_id)
            # User already exists - return success (don't retry webhook)
            return {
                "success": True,
                "message": "User already exists",
                "clerk_id": clerk_id
            }
        
        # Step 6: Create new user in database
        result = supabase.table("users").insert({
            "clerk_id": clerk_id
        }).execute()
        
        # Step 7: Verify insertion was successful
        if not result.data:
            logger.error("user_creation_failed", user_id=clerk_id, reason="no_data_returned")
            raise HTTPException(
                status_code=500, 
                detail="Failed to create user in database"
            )
        
        logger.info("user_created_successfully", user_id=clerk_id, db_user_id=result.data[0].get("id"))
        return {
            "success": True,
            "message": "User created successfully",
            "user": result.data[0]
        }
    
    except Exception as e:
        logger.error("webhook_processing_error", error=str(e), exc_info=True)
        # Only catch unexpected exceptions (database errors, network errors, etc.)
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while processing webhook: {str(e)}"
        )