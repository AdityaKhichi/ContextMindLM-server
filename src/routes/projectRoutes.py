import json

from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException

from src.services.supabase import supabase
from src.services.clerkAuth import get_current_user

from src.models.index import ProjectCreate, ProjectSettings, SendMessageRequest, MessageRole

from src.agents.simple_agent.agent import create_simple_rag_agent
from src.agents.supervisor_agent.agent import create_supervisor_agent

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse


router = APIRouter(
    tags=["projects"]
)


@router.get("/") 
def get_projects(clerk_id: str = Depends(get_current_user)):
    try:
        result = supabase.table('projects').select('*').eq('clerk_id', clerk_id).execute()
 
        return { 
            "success": True,
            "message": "Projects retrieved successfully",
            "data": result.data or []
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching projects: {str(e)}")
    
    
@router.post("/") 
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

        project_id = project_result.data[0]["id"]

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
            "data": project_result.data[0] 
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An internal server error occurred while creating project: {str(e)}"
        )
    
    
@router.delete("/{project_id}")
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


@router.get("/{project_id}")
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


@router.get("/{project_id}/chats")
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
    

@router.get("/{project_id}/settings")
async def get_project_settings(
    project_id: str
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
    

@router.put("/{project_id}/settings")
async def update_project_settings(
    project_id: str, 
    settings: ProjectSettings, 
    clerk_id: str = Depends(get_current_user)
):
    try: 
        # Verify the project exists and belongs to the user
        project_result = supabase.table("projects").select("id").eq("id", project_id).eq("clerk_id", clerk_id).execute()    

        if not project_result.data:
            raise HTTPException(
                status_code=404, 
                detail="Project not found or you don't have permission to update its settings"
            )

        # Perform the update
        result = supabase.table("project_settings").update(settings.model_dump()).eq("project_id", project_id).execute()

        if not result.data:
            raise HTTPException(
                status_code=422, 
                detail="Failed to update project settings - invalid data provided"
            )

        return {
            "success": True,
            "message": "Project settings updated successfully", 
            "data": result.data[0]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An internal server error occurred while updating project settings: {str(e)}"
        )
    

@router.post("/{project_id}/chats/{chat_id}/messages")
async def send_message(
    chat_id: str,
    project_id: str,
    request: SendMessageRequest,
    clerk_id: str = Depends(get_current_user)
):
    try:
        message = request.content
        
        # Save user message
        user_message_result = supabase.table('messages').insert({
            "chat_id": chat_id,
            "content": message,
            "role": MessageRole.USER.value,
            "clerk_id": clerk_id
        }).execute()

        if not user_message_result.data:
            raise HTTPException(status_code=422, detail="Failed to create message")
        
        user_message = user_message_result.data[0]

        # Get project settings to retrieve agent_type
        try:
            project_settings = await get_project_settings(project_id)
            agent_type = project_settings["data"].get("agent_type", "simple")
        except Exception as e:
            agent_type = "simple"

        print(agent_type, "agent_type")

        # Get chat history (excluding current message)
        chat_history = get_chat_history(chat_id, exclude_message_id=user_message["id"])

        # Invoke the appropriate agent based on agent_type
        if agent_type == "simple":
            agent = create_simple_rag_agent(
                project_id=project_id,
                model="gpt-4o",
                chat_history=chat_history
            )
        elif agent_type == "agentic":
            agent = create_supervisor_agent(
                project_id=project_id,
                model="gpt-4o",
                chat_history=chat_history
            )

        # Invoke the agent with the user's message
        result = agent.invoke({
            "messages": [{"role": "user", "content": message}]
        })

        # Extract the final response and citations from the result
        final_response = result["messages"][-1].content
        citations = result.get("citations", [])

         # Insert the AI Response into the database.
        ai_message_result = supabase.table('messages').insert({
            "chat_id": chat_id,
            "content": final_response,
            "role": MessageRole.ASSISTANT.value,
            "clerk_id": clerk_id,
            "citations": citations
        }).execute()

        if not ai_message_result.data:
            raise HTTPException(status_code=422, detail="Failed to create AI response")
        
        ai_message = ai_message_result.data[0]
        
        #Return data
        return {
            "message": "Messages sent successfully",
            "data": {
                "userMessage": user_message,
                "aiMessage": ai_message
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred while creating message: {str(e)}")
    

def get_chat_history(chat_id: str, exclude_message_id: str = None) -> List[Dict[str, str]]:
    """
    Retrieves the last 10 messages (5 user + 5 assistant) from the chat,
    excluding the current message being processed.
    """
    try:
        query = (
            supabase.table("messages")
            .select("id, role, content")
            .eq("chat_id", chat_id)
            .order("created_at", desc=False)
        )
        
        # Exclude current message if provided
        if exclude_message_id:
            query = query.neq("id", exclude_message_id)
        
        messages_result = query.execute()
        
        if not messages_result.data:
            return []
        
        # Get last 10 messages (limit to 10 total messages)
        recent_messages = messages_result.data[-10:]
        
        # Format messages for agent
        formatted_history = []
        for msg in recent_messages:
            formatted_history.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        return formatted_history
    except Exception:
        # If history retrieval fails, return empty list
        return []
    

@router.post("/{project_id}/chats/{chat_id}/messages/stream")
async def stream_message(
    project_id: str,
    chat_id: str,
    message: SendMessageRequest,
    clerk_id: str = Query(..., description="Clerk user ID"),
):
    async def event_generator():
        try:

            # Insert user message into database
            message_content = message.content
            message_insert_data = {
                "content": message_content,
                "chat_id": chat_id,
                "clerk_id": clerk_id,
                "role": MessageRole.USER.value,
            }
            message_creation_result = (
                supabase.table("messages").insert(message_insert_data).execute()
            )
            if not message_creation_result.data:
                yield f"event: error\ndata: {json.dumps({'message': 'Failed to create message'})}\n\n"
                return
            
            user_message_data = message_creation_result.data[0]
            current_message_id = user_message_data["id"]
            
            # Get project settings for agent_type
            try:
                project_settings = await get_project_settings(project_id)
                agent_type = project_settings["data"].get("agent_type", "simple")
            except Exception as e:
                agent_type = "simple"

            
            # Get chat history
            chat_history = get_chat_history(chat_id, exclude_message_id=current_message_id)
            
            # Create the appropriate agent
            if agent_type == "simple":
                agent = create_simple_rag_agent(
                    project_id=project_id,
                    model="gpt-4o",
                    chat_history=chat_history
                )
            else:  # agentic
                agent = create_supervisor_agent(
                    project_id=project_id,
                    model="gpt-4o",
                    chat_history=chat_history
                )

            
            # Stream the agent response
            full_response = ""
            citations = []
            
            # Track state to know when we're in the final response
            tool_called = False
            is_final_response = False
            
            async for event in agent.astream_events(
                {"messages": [{"role": "user", "content": message_content}]},
                version="v2"
            ):
                kind = event["event"]
                tags = event.get("tags", [])
                name = event.get("name", "")
                

                # Status updates for tool calls
                if kind == "on_tool_start":
                    tool_called = True
                    tool_name = name
                    if tool_name == "rag_search":
                        yield f"event: status\ndata: {json.dumps({'status': 'Searching documents...'})}\n\n"
                    elif tool_name == "search_web":
                        yield f"event: status\ndata: {json.dumps({'status': 'Searching the web...'})}\n\n"
                
                # Detect when tool ends - next model call will be the final response
                elif kind == "on_tool_end":
                    is_final_response = True
                    yield f"event: status\ndata: {json.dumps({'status': 'Generating response...'})}\n\n"
                
                # Stream tokens from the model
                elif kind == "on_chat_model_stream":
                    # Stream if:
                    # Either tool finished OR no tool was called yet AND
                    # Has the seq:step:1 tag (part of main agent flow, not nested LLM)
                    if (is_final_response or not tool_called) and 'seq:step:1' in tags:
                        chunk = event["data"].get("chunk")
                        if chunk:
                            content = chunk.content if hasattr(chunk, 'content') else ""
                            if content:
                                full_response += content
                                yield f"event: token\ndata: {json.dumps({'content': content})}\n\n"
                
                # Capture citations from the final state
                elif kind == "on_chain_end" and name == "LangGraph" and tags == []:
                    # This is the outermost LangGraph ending
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict) and "citations" in output:
                        citations = output["citations"]
            
            
            # Insert AI response into database
            ai_response_insert_data = {
                "content": full_response,
                "chat_id": chat_id,
                "clerk_id": clerk_id,
                "role": MessageRole.ASSISTANT.value,
                "citations": citations,
            }
            ai_response_creation_result = (
                supabase.table("messages").insert(ai_response_insert_data).execute()
            )
            
            if not ai_response_creation_result.data:
                yield f"event: error\ndata: {json.dumps({'message': 'Failed to save AI response'})}\n\n"
                return
            
            ai_message_data = ai_response_creation_result.data[0]
            
            # Send done event
            yield f"event: done\ndata: {json.dumps({'userMessage': user_message_data, 'aiMessage': ai_message_data})}\n\n"
            
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )