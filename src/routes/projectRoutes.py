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

from src.config.logging import get_logger, set_project_id, set_user_id

logger = get_logger(__name__)


router = APIRouter(
    tags=["projects"]
)


@router.get("/") 
def get_projects(clerk_id: str = Depends(get_current_user)):
    set_user_id(clerk_id)

    try:
        logger.info("fetching_projects")
        result = supabase.table('projects').select('*').eq('clerk_id', clerk_id).execute()
 
        logger.info("projects_retrieved", project_count=len(result.data or []))
        return { 
            "success": True,
            "message": "Projects retrieved successfully",
            "data": result.data or []
        }

    except Exception as e:
        logger.error("projects_fetch_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching projects: {str(e)}")
    
    
@router.post("/") 
def create_project(project: ProjectCreate, clerk_id: str = Depends(get_current_user)):
    set_user_id(clerk_id)
    
    try:
        logger.info("creating_project", name=project.name)
        # Step 1: Insert new project into database
        project_result = supabase.table("projects").insert({
            "name": project.name, 
            "description": project.description,
            "clerk_id": clerk_id
        }).execute()

        if not project_result.data:
            logger.error("project_creation_failed", name=project.name, reason="no_data_returned")
            raise HTTPException(
                status_code=422, 
                detail="Failed to create project - invalid data provided"
            )

        project_id = project_result.data[0]["id"]
        logger.info("project_created", name=project.name)

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
            logger.error("project_settings_creation_failed", reason="no_data_returned")
            # Step 3: Rollback - Delete the project if settings creation fails
            supabase.table("projects").delete().eq("id", project_id).execute()
            raise HTTPException(
                status_code=422, 
                detail="Failed to create project settings - project creation rolled back"
            )

        logger.info("project_created_successfully", name=project.name)
        return {
            "success": True,
            "message": "Project created successfully", 
            "data": project_result.data[0] 
        }

    except Exception as e:
        logger.error("project_creation_error", name=project.name, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An internal server error occurred while creating project: {str(e)}"
        )
    
    
@router.delete("/{project_id}")
def delete_project(
    project_id: str, 
    clerk_id: str = Depends(get_current_user)
):
    set_project_id(project_id)
    set_user_id(clerk_id)

    try:
        logger.info("deleting_project")
        # Step 1: Verify project exists and belongs to user 
        project_result = supabase.table("projects").select("*").eq("id", project_id).eq("clerk_id", clerk_id).execute()

        if not project_result.data: 
            logger.warning("project_not_found_or_unauthorized")
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

        logger.info("project_deleted_successfully")
        return {
            "success": True,
            "message": "Project deleted successfully", 
            "data": deleted_result.data[0]  
        }
    
    except Exception as e:
        logger.error("project_deletion_error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An internal server error occurred while deleting project: {str(e)}"
        )


@router.get("/{project_id}")
async def get_project(
    project_id: str, 
    clerk_id: str = Depends(get_current_user)
):
    set_project_id(project_id)
    set_user_id(clerk_id)

    try:
        logger.info("fetching_project")
        result = supabase.table("projects").select("*").eq("id", project_id).eq("clerk_id", clerk_id).execute()

        if not result.data:
            logger.warning("project_not_found")
            raise HTTPException(
                status_code=404, 
                detail="Project not found or you don't have permission to access it"
            )

        logger.info("project_retrieved")
        return {
            "success": True,
            "message": "Project retrieved successfully", 
            "data": result.data[0]
        }
    
    except Exception as e:
        logger.error("project_retrieval_error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An internal server error occurred while retrieving project: {str(e)}"
        )


@router.get("/{project_id}/chats")
async def get_project_chats(
    project_id: str, 
    clerk_id: str = Depends(get_current_user)
):
    set_project_id(project_id)
    set_user_id(clerk_id)

    try:
        logger.info("fetching_project_chats")
        result = supabase.table("chats").select("*").eq("project_id", project_id).eq("clerk_id", clerk_id).order("created_at", desc=True).execute()

        logger.info("project_chats_retrieved", chat_count=len(result.data or []))
        return {
            "success": True,
            "message": "Project chats retrieved successfully", 
            "data": result.data or []
        }

    except Exception as e:
        logger.error("project_chats_retrieval_error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An internal server error occurred while retrieving project chats: {str(e)}"
        )
    

@router.get("/{project_id}/settings")
async def get_project_settings(
    project_id: str,
    clerk_id: str = Depends(get_current_user)
):
    set_project_id(project_id)
    set_user_id(clerk_id)

    try:
        logger.info("fetching_project_settings")
        settings_result = supabase.table("project_settings").select("*").eq("project_id", project_id).execute()

        if not settings_result.data:
            logger.warning("project_settings_not_found")
            raise HTTPException(
                status_code=404, 
                detail="Project settings not found"
            )

        logger.info("project_settings_retrieved",
            rag_strategy=settings_result.data[0].get("rag_strategy"),
            agent_type=settings_result.data[0].get("agent_type"),
            embedding_model=settings_result.data[0].get("embedding_model"),
            final_context_size=settings_result.data[0].get("final_context_size"),
            reranking_enabled=settings_result.data[0].get("reranking_enabled")
        )
        return {
            "success": True,
            "message": "Project settings retrieved successfully", 
            "data": settings_result.data[0]
        }

    except Exception as e:
        logger.error("project_settings_retrieval_error", error=str(e), exc_info=True)
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
        logger.info("updating_project_settings",
            rag_strategy=settings.rag_strategy,
            agent_type=settings.agent_type,
            embedding_model=settings.embedding_model,
            final_context_size=settings.final_context_size,
            reranking_enabled=settings.reranking_enabled
        )
        # Verify the project exists and belongs to the user
        project_result = supabase.table("projects").select("id").eq("id", project_id).eq("clerk_id", clerk_id).execute()    

        if not project_result.data:
            logger.warning("project_not_found_for_settings_update")
            raise HTTPException(
                status_code=404, 
                detail="Project not found or you don't have permission to update its settings"
            )

        # Perform the update
        result = supabase.table("project_settings").update(settings.model_dump()).eq("project_id", project_id).execute()

        if not result.data:
            logger.error("project_settings_update_failed", reason="no_data_returned")
            raise HTTPException(
                status_code=422, 
                detail="Failed to update project settings - invalid data provided"
            )

        logger.info("project_settings_updated_successfully",
            rag_strategy=settings.rag_strategy,
            agent_type=settings.agent_type,
            embedding_model=settings.embedding_model,
            final_context_size=settings.final_context_size,
            reranking_enabled=settings.reranking_enabled
        )
        return {
            "success": True,
            "message": "Project settings updated successfully", 
            "data": result.data[0]
        }

    except Exception as e:
        logger.error("project_settings_update_error", error=str(e), exc_info=True)
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
    set_project_id(project_id)
    set_user_id(clerk_id)

    try:
        logger.info("sending_message", chat_id=chat_id)
        message = request.content
        
        # Save user message
        user_message_result = supabase.table('messages').insert({
            "chat_id": chat_id,
            "content": message,
            "role": MessageRole.USER.value,
            "clerk_id": clerk_id
        }).execute()

        if not user_message_result.data:
            logger.error("message_creation_failed", chat_id=chat_id, reason="no_data_returned")
            raise HTTPException(status_code=422, detail="Failed to create message")
        
        user_message = user_message_result.data[0]
        logger.info("user_message_created", message_id=user_message["id"], chat_id=chat_id)

        # Get project settings to retrieve agent_type
        try:
            project_settings = await get_project_settings(project_id)
            agent_type = project_settings["data"].get("agent_type", "simple")
        except Exception as e:
            logger.warning("settings_retrieval_failed_defaulting_to_simple", error=str(e))
            agent_type = "simple"

        logger.info("agent_type_determined", agent_type=agent_type)

        # Get chat history (excluding current message)
        chat_history = get_chat_history(chat_id, exclude_message_id=user_message["id"])
        logger.info("chat_history_retrieved", chat_id=chat_id, history_length=len(chat_history))

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

        logger.info("invoking_agent", chat_id=chat_id, agent_type=agent_type)
        # Invoke the agent with the user's message
        result = agent.invoke({
            "messages": [{"role": "user", "content": message}]
        })

        # Extract the final response and citations from the result
        final_response = result["messages"][-1].content
        citations = result.get("citations", [])
        logger.info("agent_invocation_completed", chat_id=chat_id, response_length=len(final_response), citations_count=len(citations))

         # Insert the AI Response into the database.
        ai_message_result = supabase.table('messages').insert({
            "chat_id": chat_id,
            "content": final_response,
            "role": MessageRole.ASSISTANT.value,
            "clerk_id": clerk_id,
            "citations": citations
        }).execute()

        if not ai_message_result.data:
            logger.error("ai_response_creation_failed", chat_id=chat_id, reason="no_data_returned")
            raise HTTPException(status_code=422, detail="Failed to create AI response")
        
        ai_message = ai_message_result.data[0]
        
        logger.info("message_sent_successfully", chat_id=chat_id, ai_message_id=ai_message["id"])
        #Return data
        return {
            "message": "Messages sent successfully",
            "data": {
                "userMessage": user_message,
                "aiMessage": ai_message
            }
        }
        
    except Exception as e:
        logger.error("send_message_error", chat_id=chat_id, error=str(e), exc_info=True)
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
    set_project_id(project_id)  
    set_user_id(clerk_id)  

    async def event_generator():
        try:
            logger.info("sending_message", chat_id=chat_id)

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
                logger.error("message_creation_failed", chat_id=chat_id, reason="no_data_returned")
                yield f"event: error\ndata: {json.dumps({'message': 'Failed to create message'})}\n\n"
                return
            
            user_message_data = message_creation_result.data[0]
            current_message_id = user_message_data["id"]
            logger.info("user_message_created", message_id=current_message_id, chat_id=chat_id)
            
            # Get project settings for agent_type
            try:
                project_settings = await get_project_settings(project_id)
                agent_type = project_settings["data"].get("agent_type", "simple")
            except Exception as e:
                logger.warning("settings_retrieval_failed_defaulting_to_simple", error=str(e))
                agent_type = "simple"

            logger.info("agent_type_determined", agent_type=agent_type)

            # Get chat history
            chat_history = get_chat_history(chat_id, exclude_message_id=current_message_id)
            logger.info("chat_history_retrieved", chat_id=chat_id, history_length=len(chat_history))
            
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

            logger.info("invoking_agent", chat_id=chat_id, agent_type=agent_type)
            
            # Stream the agent response
            full_response = ""
            citations = []
            
            # Track state to know when we're in the final response
            passed_guardrail = False
            tool_called = False
            is_final_response = False
            
            async for event in agent.astream_events(
                {"messages": [{"role": "user", "content": message_content}]},
                version="v2"
            ):
                kind = event["event"]
                tags = event.get("tags", [])
                name = event.get("name", "")
                
                # Detect guardrail completion
                if kind == "on_chain_end" and name == "guardrail":
                    # Check if guardrail rejected the input
                    output = event.get("data", {}).get("output", {})
                    if output.get("guardrail_passed") == False:
                        # Stream the rejection message
                        messages = output.get("messages", [])
                        if messages:
                            rejection_content = messages[0].content if hasattr(messages[0], 'content') else str(messages[0])
                            full_response = rejection_content
                            yield f"event: token\ndata: {json.dumps({'content': rejection_content})}\n\n"
                    else:
                        passed_guardrail = True
                        yield f"event: status\ndata: {json.dumps({'status': 'Thinking...'})}\n\n"

                # Status updates for tool calls
                elif kind == "on_tool_start":
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
                    # Guardrail passed
                    # Either tool finished OR no tool was called yet AND
                    # Has the seq:step:1 tag (part of main agent flow, not nested LLM)
                    if passed_guardrail and (is_final_response or not tool_called) and 'seq:step:1' in tags:
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
            
            logger.info("agent_invocation_completed", chat_id=chat_id, response_length=len(full_response), citations_count=len(citations))
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
                logger.error("ai_response_creation_failed", chat_id=chat_id, reason="no_data_returned")
                yield f"event: error\ndata: {json.dumps({'message': 'Failed to save AI response'})}\n\n"
                return
            
            ai_message_data = ai_response_creation_result.data[0]
            logger.info("message_sent_successfully", chat_id=chat_id, ai_message_id=ai_message_data["id"])
            
            # Send done event
            yield f"event: done\ndata: {json.dumps({'userMessage': user_message_data, 'aiMessage': ai_message_data})}\n\n"
            
        except Exception as e:
            logger.error("send_message_error", chat_id=chat_id, error=str(e), exc_info=True)
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )