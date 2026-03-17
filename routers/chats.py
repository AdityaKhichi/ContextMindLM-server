from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from database import supabase
from auth import get_current_user
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import List, Dict, Tuple


# Initialize LLM for summarization
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1536
)

router = APIRouter(
    tags=["chats"]
)  

class ChatCreate(BaseModel):
    title: str
    project_id: str

class SendMessageRequest(BaseModel):
    content: str


@router.post("/api/chats")
async def create_chat(
    chat: ChatCreate, 
    clerk_id: str = Depends(get_current_user)
):
    try:
        result = supabase.table("chats").insert({
            "title": chat.title, 
            "project_id": chat.project_id, 
            "clerk_id": clerk_id
        }).execute()

        return {
            "message": "Chat created successfully", 
            "data": result.data[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail = f"Failed to create chat: {str(e)}")


@router.delete("/api/chats/{chat_id}")
async def delete_chat(
    chat_id: str, 
    clerk_id: str = Depends(get_current_user)
):
    try:
        deleted_result = supabase.table("chats").delete().eq("id", chat_id).eq("clerk_id", clerk_id).execute()

        if not deleted_result.data: 
            raise HTTPException(status_code=404, detail="Chat not found or access denied")

        return {
            "message": "Chat Deleted Successfully", 
            "data": deleted_result.data[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail = f"Failed to delete chat: {str(e)}")
    

@router.get("/api/chats/{chat_id}")
async def get_chat(
    chat_id: str,
    clerk_id: str = Depends(get_current_user)
):
    try:
        result = supabase.table('chats').select('*').eq('id', chat_id).eq('clerk_id', clerk_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Chat not found or access denied")
        
        chat = result.data[0]
        
        # Get messages for this chat
        messages_result = supabase.table('messages').select('*').eq('chat_id', chat_id).order('created_at', desc=False).execute()

        chat['messages'] = messages_result.data or []
        
        return {
            "message": "Chat retrieved successfully",
            "data": chat
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chat: {str(e)}")


@router.post("/api/projects/{project_id}/chats/{chat_id}/messages")
async def send_message(
    chat_id: str,
    project_id: str,
    request: SendMessageRequest,
    clerk_id: str = Depends(get_current_user)
):
    try:
        message = request.content
        
        print(f"💬 New message: {message[:50]}...")
        
        # Save user message
        print(f"💾 Saving user message...")
        user_message_result = supabase.table('messages').insert({
            "chat_id": chat_id,
            "content": message,
            "role": "user",
            "clerk_id": clerk_id
        }).execute()
        
        user_message = user_message_result.data[0]
        print(f"✅ User message saved: {user_message['id']}")
        
        # Load project settings
        # We need settings to know: chunk size, similarity threshold, etc.
        settings = load_project_settings(project_id)
        
        # Get document IDs for this project
        # This narrows our search scope to only documents uploaded to this specific project
        document_ids = get_document_ids(project_id)

        
        # Generate query embedding
        # Perform vector search using the RPC function 
        chunks = vector_search(message, document_ids, settings)
        print(f"✅ Retrieved {len(chunks)} relevant chunks from vector search")


        # Build context from retrieved chunks
        # Format the retrieved chunks into a structured context with citations
        texts, images, tables, citations = build_context(chunks)
        validate_context(texts, images, tables, citations)
        
        # Build system prompt with injected context
        # Add the retrieved document context to the system prompt so the LLM can answer based on the documents
        print(f"🤖 Preparing context and calling LLM...")
        ai_response = prepare_prompt_and_invoke_llm(
            user_query=message,
            texts=texts,
            images=images,
            tables=tables
        )
        
        # Save AI message with citations to database
        # Store the AI's response along with citations
        
        print(f"💾 Saving AI message...")

        ai_message_result = supabase.table('messages').insert({
            "chat_id": chat_id,
            "content": ai_response,
            "role": "assistant",
            "clerk_id": clerk_id,
            "citations": citations
        }).execute()
        
        ai_message = ai_message_result.data[0]
        print(f"✅ AI message saved: {ai_message['id']}")
        
        #Return data
        return {
            "message": "Messages sent successfully",
            "data": {
                "userMessage": user_message,
                "aiMessage": ai_message
            }
        }
        
    except Exception as e:
        print(f"❌ Error in send_message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

def load_project_settings(project_id: str) -> dict:
    print(f"⚙️ Fetching project settings...")
    settings_result = supabase.table('project_settings').select('*').eq('project_id', project_id).execute()
    
    if not settings_result.data:
        raise HTTPException(status_code=404, detail="Project settings not found")
    
    settings = settings_result.data[0]
    print(f"✅ Settings retrieved")
    return settings

def get_document_ids(project_id: str) -> List[str]:
    print(f"📄 Fetching project documents...")
    documents_result = supabase.table('project_documents').select('id').eq('project_id', project_id).execute()
    
    document_ids = [doc['id'] for doc in documents_result.data]
    print(f"✅ Found {len(document_ids)} documents")
    return document_ids

def vector_search(query: str, document_ids: List[str], settings: dict) -> List[Dict]:
    query_embedding = embeddings_model.embed_query(query)
    
    result = supabase.rpc('vector_search_document_chunks', {
        'query_embedding': query_embedding,
        'filter_document_ids': document_ids,
        'match_threshold': settings['similarity_threshold'],
        'chunks_per_search': settings['chunks_per_search']
    }).execute()
    
    return result.data if result.data else []


def build_context(chunks: List[Dict]) -> Tuple[List[str], List[str], List[str], List[Dict]]:
    """
    Returns:
        Tuple of (texts, images, tables, citations)
    """
    if not chunks:
        return [], [], [], []
    
    texts = []
    images = []
    tables = []
    citations = [] 
    
    # Batch fetch all filenames in ONE query
    doc_ids = [chunk['document_id'] for chunk in chunks if chunk.get('document_id')]
    unique_doc_ids: List[str] = list(set(doc_ids))  # ✅ Fixed syntax
    
    filename_map = {}
    
    if unique_doc_ids:
        result = supabase.table('project_documents')\
            .select('id, filename')\
            .in_('id', unique_doc_ids)\
            .execute()
        filename_map = {doc['id']: doc['filename'] for doc in result.data}
    
    # Process each chunk
    for chunk in chunks:
        original_content = chunk.get('original_content', {})
        
        # Extract content from chunk
        chunk_text = original_content.get('text', '')
        chunk_images = original_content.get('images', [])
        chunk_tables = original_content.get('tables', [])

        # Collect content
        if chunk_text:  # ✅ Add this check back
            texts.append(chunk_text)
        images.extend(chunk_images)
        tables.extend(chunk_tables)
        
        # Add citation for every chunk
        doc_id = chunk.get('document_id')
        if doc_id:
            citations.append({
                "chunk_id": chunk.get('id'),
                "document_id": doc_id,
                "filename": filename_map.get(doc_id, 'Unknown Document'),
                "page": chunk.get('page_number', 'Unknown')
            })
    
    return texts, images, tables, citations

def validate_context(texts: List[str], images: List[str], tables: List[str], citations: List[Dict]) -> None:
    print("\n" + "="*80)
    print("📦 CONTEXT VALIDATION")
    print("="*80)
    
    # Texts - SHOW FULL TEXT
    print(f"\n📝 TEXTS: {len(texts)} chunks")
    for i, text in enumerate(texts, 1):
        print(f"\n{'='*80}")
        print(f"CHUNK [{i}] - {len(text)} characters")
        print(f"{'='*80}")
        print(text)  # ✅ Full text, no truncation
        print(f"{'='*80}\n")
    
    # Images
    print(f"\n🖼️  IMAGES: {len(images)}")
    for i, img in enumerate(images, 1):
        img_preview = str(img)[:60] + ('...' if len(str(img)) > 60 else '')
        print(f"  [{i}] {img_preview}")
    
    # Tables
    print(f"\n📊 TABLES: {len(tables)}")
    for i, table in enumerate(tables, 1):
        if isinstance(table, dict):
            rows = len(table.get('rows', []))
            cols = len(table.get('headers', []))
            print(f"  [{i}] {rows} rows × {cols} cols")
        else:
            print(f"  [{i}] Type: {type(table).__name__}")
    
    # Citations
    print(f"\n📚 CITATIONS: {len(citations)}")
    for i, cite in enumerate(citations, 1):
        chunk_id = cite['chunk_id'][:8] if cite.get('chunk_id') else 'N/A'
        print(f"  [{i}] {cite['filename']} (pg.{cite['page']}) | chunk: {chunk_id}...")
    
    # Summary
    total_chars = sum(len(text) for text in texts)
    print(f"\n{'='*80}")
    print(f"✅ Total: {len(texts)} texts ({total_chars:,} chars), {len(images)} images, {len(tables)} tables, {len(citations)} citations")
    print("="*80 + "\n")


def prepare_prompt_and_invoke_llm(
    user_query: str,
    texts: List[str],
    images: List[str],
    tables: List[str]
) -> str:
    # Build system prompt parts
    prompt_parts = []
    
    # Main instruction
    prompt_parts.append(
        "You are a helpful AI assistant that answers questions based solely on the provided context. "
        "Your task is to provide accurate, detailed answers using ONLY the information available in the context below.\n\n"
        "IMPORTANT RULES:\n"
        "- Only answer based on the provided context (texts, tables, and images)\n"
        "- If the answer cannot be found in the context, respond with: 'I don't have enough information in the provided context to answer that question.'\n"
        "- Do not use external knowledge or make assumptions beyond what's explicitly stated\n"
        "- When referencing information, be specific and cite relevant parts of the context\n"
        "- Synthesize information from texts, tables, and images to provide comprehensive answers\n\n"
    )
    
    # Add text contexts
    if texts:
        prompt_parts.append("=" * 80)
        prompt_parts.append("CONTEXT DOCUMENTS")
        prompt_parts.append("=" * 80 + "\n")
        
        for i, text in enumerate(texts, 1):
            prompt_parts.append(f"--- Document Chunk {i} ---")
            prompt_parts.append(text.strip())
            prompt_parts.append("")
    
    # Add tables if present
    if tables:
        prompt_parts.append("\n" + "=" * 80)
        prompt_parts.append("RELATED TABLES")
        prompt_parts.append("=" * 80)
        prompt_parts.append(
            "The following tables contain structured data that may be relevant to your answer. "
            "Analyze the table contents carefully.\n"
        )
        
        for i, table_html in enumerate(tables, 1):
            prompt_parts.append(f"--- Table {i} ---")
            prompt_parts.append(table_html)
            prompt_parts.append("")
    
    # Reference images if present
    if images:
        prompt_parts.append("\n" + "=" * 80)
        prompt_parts.append("RELATED IMAGES")
        prompt_parts.append("=" * 80)
        prompt_parts.append(
            f"{len(images)} image(s) will be provided alongside the user's question. "
            "These images may contain diagrams, charts, figures, formulas, or other visual information. "
            "Carefully analyze the visual content when formulating your response. "
            "The images are part of the retrieved context and should be used to answer the question.\n"
        )
    
    # Final instruction
    prompt_parts.append("=" * 80)
    prompt_parts.append(
        "Based on all the context provided above (documents, tables, and images), "
        "please answer the user's question accurately and comprehensively."
    )
    prompt_parts.append("=" * 80)
    
    system_prompt = "\n".join(prompt_parts)
    
    # Build messages for LLM
    messages = [SystemMessage(content=system_prompt)]
    
    # Create human message with user query and images
    if images:
        # Multi-modal message: text + images
        content_parts = [{"type": "text", "text": user_query}]
        
        # Add each image to the content array
        for img_base64 in images:
            # Clean base64 string if it has data URI prefix
            if img_base64.startswith('data:image'):
                img_base64 = img_base64.split(',', 1)[1]
            
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })
        
        messages.append(HumanMessage(content=content_parts))
    else:
        # Text-only message
        messages.append(HumanMessage(content=user_query))
    
    # Invoke LLM and return response
    print(f"🤖 Invoking LLM with {len(messages)} messages ({len(texts)} texts, {len(tables)} tables, {len(images)} images)...")
    response = llm.invoke(messages)
    
    return response.content
