from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.text import partition_text
from unstructured.partition.md import partition_md

from src.services.llm import openAI
from langchain_core.messages import HumanMessage



def get_page_number(chunk, chunk_index):
    if hasattr(chunk, 'metadata'):
        page_number = getattr(chunk.metadata, 'page_number', None)
        if page_number is not None:
            return page_number
    
    # Fallback: use chunk index as page number
    return chunk_index + 1


def separate_content_types(chunk, source_type="file"):
    """Analyze what types of content are in a chunk"""
    is_url_source = source_type == 'url'
    
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    
    # Check for tables and images in original elements
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            
            # Handle tables
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            
            # Handle images (skip for URL sources)
            elif element_type == 'Image' and not is_url_source:
                if (hasattr(element, 'metadata') and 
                    hasattr(element.metadata, 'image_base64') and 
                    element.metadata.image_base64 is not None
                ):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    
    content_data['types'] = list(set(content_data['types']))
    return content_data


def create_ai_summary(text, tables_html, images_base64):
    """Create AI-enhanced summary for mixed content"""

    try:
        # Build the text prompt with more efficient instructions
        prompt_text = f"""Create a searchable index for this document content.
        CONTENT:
        {text}
        """
        
        # Add tables if present
        if tables_html:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables_html):
                prompt_text += f"Table {i+1}:\n{table}\n\n"
        
        # More concise but effective prompt
        prompt_text += """
        Generate a structured search index (aim for 250-400 words):

        QUESTIONS: List 5-7 key questions this content answers (use what/how/why/when/who variations)

        KEYWORDS: Include:
        - Specific data (numbers, dates, percentages, amounts)
        - Core concepts and themes
        - Technical terms and casual alternatives
        - Industry terminology

        VISUALS (if images present):
        - Chart/graph types and what they show
        - Trends and patterns visible
        - Key insights from visualizations

        DATA RELATIONSHIPS (if tables present):
        - Column headers and their meaning
        - Key metrics and relationships
        - Notable values or patterns

        Focus on terms users would actually search for. Be specific and comprehensive.

        SEARCH INDEX:"""
        
        # Build message content starting with the text prompt
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add images to the message
        for i, image_base64 in enumerate(images_base64):
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        message = HumanMessage(content=message_content)
        
        response = openAI["embeddings_llm"].invoke([message])
        
        return response.content
        
    except Exception as e:
        print(f" AI summary failed: {e}")


def partition_document(temp_file: str, file_type: str, source_type: str = "file"):

    if source_type == "url": 
        return partition_html(
            filename=temp_file
        )

    elif file_type == "pdf":
        return partition_pdf(
            filename=temp_file,  # Path to your PDF file
            strategy="hi_res", # Use the most accurate (but slower) processing method of extraction
            infer_table_structure=True, # Keep tables as structured HTML, not jumbled text
            extract_image_block_types=["Image"], # Grab images found in the PDF
            extract_image_block_to_payload=True # Store images as base64 data you can actually use
        )
    
    elif file_type == 'docx':
        return partition_docx(
            filename=temp_file,
            strategy="hi_res",
            infer_table_structure=True
        )

    elif file_type == 'pptx':
        return partition_pptx(
            filename=temp_file,
            strategy="hi_res",
            infer_table_structure=True, 
        )

    elif file_type == "txt":
        return partition_text(
            filename=temp_file
        )
    
    elif file_type == "md":
        return partition_md(
            filename=temp_file
        )


def analyze_elements(elements):
    text_count = 0
    table_count = 0
    image_count = 0
    title_count = 0
    other_count = 0

    # Go through each element and count what type it is 
    for element in elements: 
        element_name = type(element).__name__ #Get the class names

        if element_name == "Table":
            table_count += 1
        elif element_name == "Image": 
            image_count += 1
        elif element_name in ["Title", "Header"]:
            title_count += 1
        elif element_name in ["NarrativeText", "Text", "ListItem", "FigureCaption"]:
            text_count += 1
        else:
            other_count += 1

    # Return a simple dictionary
    return {
        "text": text_count,
        "tables": table_count,
        "images": image_count,
        "titles": title_count,
        "other": other_count
    }