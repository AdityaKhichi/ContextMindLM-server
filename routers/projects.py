from fastapi import APIRouter, Depends, HTTPException
from database import supabase
from auth import get_current_user
from pydantic import BaseModel

router = APIRouter(
    tags=["projects"]
)

class ProjectCreate(BaseModel):
    name: str
    description: str = ""

@router.get("/api/projects") 
def get_projects(clerk_id: str = Depends(get_current_user)):
    try:
        result = supabase.table('projects').select('*').eq('clerk_id', clerk_id).execute()
 
        return { 
            "success": True,
            "message": "Projects retrieved successfully",
            "data": result.data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get projects: {str(e)}")
    
    
@router.post("/api/projects") 
def create_project(project: ProjectCreate, clerk_id: str = Depends(get_current_user)):
    try:
        # Step 1: Insert new project into database
        project_result = supabase.table("projects").insert({
            "name": project.name, 
            "description": project.description,
            "clerk_id": clerk_id
        }).execute()

        if not project_result.data:
            raise HTTPException(
                status_code=422, 
                detail="Failed to create project - invalid data provided"
            )

        created_project = project_result.data[0]
        project_id = created_project["id"]

        # Step 2: Create default settings for the project 
        settings_result = supabase.table("project_settings").insert({
            "project_id": project_id, 
            "embedding_model": "text-embedding-3-large",
            "rag_strategy": "basic",
            "agent_type": "agentic",
            "chunks_per_search": 10,
            "final_context_size": 5,
            "similarity_threshold": 0.3,
            "number_of_queries": 5,
            "reranking_enabled": True,
            "reranking_model": "rerank-english-v3.0",
            "vector_weight": 0.7,
            "keyword_weight": 0.3,
        }).execute()

        if not settings_result.data:
            # Step 3: Rollback - Delete the project if settings creation fails
            supabase.table("projects").delete().eq("id", project_id).execute()
            raise HTTPException(
                status_code=422, 
                detail="Failed to create project settings - project creation rolled back"
            )

        return {
            "success": True,
            "message": "Project created successfully", 
            "data": created_project 
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An internal server error occurred while creating project: {str(e)}"
        )
    
    
@router.delete("/api/projects/{project_id}")
def delete_project(
    project_id: str, 
    clerk_id: str = Depends(get_current_user)
):
    try:
        # Step 1: Verify project exists and belongs to user 
        project_result = supabase.table("projects").select("*").eq("id", project_id).eq("clerk_id", clerk_id).execute()

        if not project_result.data: 
            raise HTTPException(
                status_code=404, 
                detail="Project not found or you don't have permission to delete it"
            )

        # Step 2: Delete project (CASCADE handles all related data)
        deleted_result = supabase.table("projects").delete().eq("id", project_id).eq("clerk_id", clerk_id).execute()

        if not deleted_result.data: 
            raise HTTPException(
                status_code=404, 
                detail="Failed to delete project - project not found"
            )

        return {
            "success": True,
            "message": "Project deleted successfully", 
            "data": deleted_result.data[0]  
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An internal server error occurred while deleting project: {str(e)}"
        )


@router.get("/api/projects/{project_id}")
async def get_project(
    project_id: str, 
    clerk_id: str = Depends(get_current_user)
):
    try:
        result = supabase.table("projects").select("*").eq("id", project_id).eq("clerk_id", clerk_id).execute()

        if not result.data:
            raise HTTPException(
                status_code=404, 
                detail="Project not found or you don't have permission to access it"
            )

        return {
            "success": True,
            "message": "Project retrieved successfully", 
            "data": result.data[0]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An internal server error occurred while retrieving project: {str(e)}"
        )


@router.get("/api/projects/{project_id}/chats")
async def get_project_chats(
    project_id: str, 
    clerk_id: str = Depends(get_current_user)
):
    try:
        result = supabase.table("chats").select("*").eq("project_id", project_id).eq("clerk_id", clerk_id).order("created_at", desc=True).execute()

        return {
            "success": True,
            "message": "Project chats retrieved successfully", 
            "data": result.data or []
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An internal server error occurred while retrieving project chats: {str(e)}"
        )
    

@router.get("/api/projects/{project_id}/settings")
async def get_project_settings(
    project_id: str, 
    clerk_id: str = Depends(get_current_user)
):
    try:
        settings_result = supabase.table("project_settings").select("*").eq("project_id", project_id).execute()

        if not settings_result.data:
            raise HTTPException(
                status_code=404, 
                detail="Project settings not found"
            )

        return {
            "success": True,
            "message": "Project settings retrieved successfully", 
            "data": settings_result.data[0]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An internal server error occurred while retrieving project settings: {str(e)}"
        )

