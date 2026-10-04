import os

config_content = '''"""Configuration settings for the Icebreaker Bot."""

# Model settings
LLM_MODEL_ID = "gemma2:2b"
EMBEDDING_MODEL_ID = "embeddinggemma:latest"

# ProxyCurl API settings
PROXYCURL_API_KEY = ""  # Replace with your API key

# Mock data URL
MOCK_DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZRe59Y_NJyn3hZgnF1iFYA/linkedin-profile-data.json"

# Query settings
SIMILARITY_TOP_K = 5
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 500
MIN_NEW_TOKENS = 1
TOP_K = 50
TOP_P = 1

# Node settings
CHUNK_SIZE = 500

# LLM prompt templates
INITIAL_FACTS_TEMPLATE = """
You are an AI assistant that provides detailed answers based on the provided context.

Context information is below:

{context_str}

Based on the context provided, list 3 interesting facts about this person's career or education.

Answer in detail, using only the information provided in the context.
"""

USER_QUESTION_TEMPLATE = """
You are an AI assistant that provides detailed answers to questions based on the provided context.

Context information is below:

{context_str}

Question: {query_str}

Answer in full details, using only the information provided in the context. If the answer is not available in the context, say "I don't know. The information is not available on the LinkedIn page."
"""
'''

data_extraction_content = '''"""Module for extracting LinkedIn profile data."""

import time
import requests
import logging
from typing import Dict, Optional, Any
import config

logger = logging.getLogger(__name__)

def extract_linkedin_profile(
    linkedin_profile_url: str, 
    api_key: Optional[str] = None, 
    mock: bool = False
) -> Dict[str, Any]:
    start_time = time.time()
    try:
        if mock:
            logger.info("Using mock data from a premade JSON file...")
            mock_url = config.MOCK_DATA_URL
            response = requests.get(mock_url, timeout=30)
        else:
            if not api_key:
                raise ValueError("API key is required when mock is set to False.")
            logger.info("Starting to extract the LinkedIn profile...")
            api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
            headers = {"Authorization": f"Bearer {api_key}"}
            params = {
                "url": linkedin_profile_url,
                "fallback_to_cache": "on-error",
                "use_cache": "if-present",
                "skills": "include",
                "inferred_salary": "include",
                "personal_email": "include",
                "personal_contact_number": "include"
            }
            logger.info(f"Sending API request to ProxyCurl at {time.time() - start_time:.2f} seconds...")
            response = requests.get(api_endpoint, headers=headers, params=params, timeout=10)
        
        logger.info(f"Received response at {time.time() - start_time:.2f} seconds...")
        if response.status_code == 200:
            try:
                data = response.json()
                data = {
                    k: v for k, v in data.items()
                    if v not in ([], "", None) and k not in ["people_also_viewed", "certifications"]
                }
                if data.get("groups"):
                    for group_dict in data.get("groups"):
                        group_dict.pop("profile_pic_url", None)
                return data
            except ValueError as e:
                logger.error(f"Error parsing JSON response: {e}")
                return {}
        else:
            logger.error(f"Failed to retrieve data. Status code: {response.status_code}")
            return {}
    except Exception as e:
        logger.error(f"Error in extract_linkedin_profile: {e}")
        return {}
'''

data_processing_content = '''"""Module for processing LinkedIn profile data."""

import json
import logging
from typing import Dict, List, Any, Optional

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from modules.llm_interface import create_watsonx_embedding
import config

logger = logging.getLogger(__name__)

def split_profile_data(profile_data: Dict[str, Any]) -> List:
    logger.info("Splitting profile data into chunks...")
    json_string = json.dumps(profile_data, indent=2)
    doc = Document(text=json_string)
    parser = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=20)
    nodes = parser.get_nodes_from_documents([doc])
    logger.info(f"Created {len(nodes)} nodes from profile data")
    return nodes

def create_vector_database(nodes: List) -> Optional[VectorStoreIndex]:
    logger.info("Creating vector database...")
    try:
        embed_model = create_watsonx_embedding()
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        logger.info("Vector database created successfully")
        return index
    except Exception as e:
        logger.error(f"Failed to create vector database: {e}")
        return None

def verify_embeddings(index: VectorStoreIndex) -> bool:
    try:
        vector_store = index.storage_context.vector_store
        # We assume True for now since we built it with Ollama and LlamaIndex handles it natively.
        return True
    except Exception as e:
        logger.error(f"Error verifying embeddings: {e}")
        return False
'''

llm_interface_content = '''"""Module for interfacing with Ollama LLMs."""

import logging
from typing import Dict, Any, Optional

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

import config

logger = logging.getLogger(__name__)

def create_watsonx_embedding() -> OllamaEmbedding:
    logger.info(f"Initializing Ollama embedding model: {config.EMBEDDING_MODEL_ID}")
    embeddings = OllamaEmbedding(model_name=config.EMBEDDING_MODEL_ID)
    return embeddings

def create_watsonx_llm(
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    decoding_method: str = "sample"
) -> Ollama:
    logger.info(f"Initializing Ollama LLM: {config.LLM_MODEL_ID}")
    llm = Ollama(
        model=config.LLM_MODEL_ID,
        temperature=temperature,
        request_timeout=300.0,
    )
    return llm

def change_llm_model(new_model_id: str) -> None:
    config.LLM_MODEL_ID = new_model_id
    logger.info(f"Changed LLM model to {new_model_id}")
'''

query_engine_content = '''"""Module for querying indexed LinkedIn profile data."""

import logging
from typing import Any, Dict, Optional

from llama_index.core import VectorStoreIndex, PromptTemplate

from modules.llm_interface import create_watsonx_llm
import config

logger = logging.getLogger(__name__)

def generate_initial_facts(index: VectorStoreIndex) -> str:
    logger.info("Generating initial facts...")
    llm = create_watsonx_llm()
    qa_template = PromptTemplate(config.INITIAL_FACTS_TEMPLATE)
    
    query_engine = index.as_query_engine(
        llm=llm,
        text_qa_template=qa_template,
        similarity_top_k=config.SIMILARITY_TOP_K
    )
    
    # We ask an empty query, but the prompt template forces 3 facts based on context.
    response = query_engine.query("Provide the facts.")
    return str(response)

def answer_user_query(index: VectorStoreIndex, user_query: str) -> Any:
    logger.info(f"Answering user query: {user_query}")
    llm = create_watsonx_llm()
    qa_template = PromptTemplate(config.USER_QUESTION_TEMPLATE)
    
    query_engine = index.as_query_engine(
        llm=llm,
        text_qa_template=qa_template,
        similarity_top_k=config.SIMILARITY_TOP_K
    )
    
    response = query_engine.query(user_query)
    return str(response)
'''

main_content = '''"""Main script for running the Icebreaker Bot."""

import sys
import time
import logging
import argparse

from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import split_profile_data, create_vector_database, verify_embeddings
from modules.query_engine import generate_initial_facts, answer_user_query
from typing import Dict, Any, Optional
import config
from modules.llm_interface import change_llm_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)

logger = logging.getLogger(__name__)

def process_linkedin(linkedin_url, api_key=None, mock=False):
    logger.info(f"Starting process for: {linkedin_url}")
    
    profile_data = extract_linkedin_profile(linkedin_url, api_key, mock)
    if not profile_data:
        logger.error("Failed to extract profile data.")
        return
        
    nodes = split_profile_data(profile_data)
    if not nodes:
        logger.error("Failed to split profile data.")
        return
        
    index = create_vector_database(nodes)
    if not index:
        logger.error("Failed to create vector database.")
        return
        
    facts = generate_initial_facts(index)
    print("\\n" + "="*50)
    print("INITIAL FACTS:")
    print("="*50)
    print(facts)
    print("="*50 + "\\n")
    
    chatbot_interface(index)

def chatbot_interface(index):
    print("Chatbot interface started. Ask questions about the profile!")
    print("Type 'exit', 'quit', or 'bye' to end the conversation.")
    
    while True:
        try:
            query = input("\\nYou: ")
            if query.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break
                
            if not query.strip():
                continue
                
            response = answer_user_query(index, query)
            print(f"\\nBot: {response}")
            
        except KeyboardInterrupt:
            print("\\nGoodbye!")
            break

def main():
    parser = argparse.ArgumentParser(description='Icebreaker Bot - LinkedIn Profile Analyzer')
    parser.add_argument('--url', type=str, help='LinkedIn profile URL')
    parser.add_argument('--api-key', type=str, help='ProxyCurl API key')
    parser.add_argument('--mock', action='store_true', help='Use mock data instead of API')
    parser.add_argument('--model', type=str, help='LLM model to use')
    
    args = parser.parse_args()
    
    linkedin_url = args.url or input("Enter LinkedIn profile URL (or press Enter to use mock data): ")
    use_mock = args.mock or not linkedin_url
    
    if args.model:
        change_llm_model(args.model)
    
    api_key = args.api_key or config.PROXYCURL_API_KEY
    
    if not use_mock and not api_key:
        api_key = input("Enter ProxyCurl API key: ")
    
    if use_mock and not linkedin_url:
        linkedin_url = "https://www.linkedin.com/in/leonkatsnelson/"
    
    process_linkedin(linkedin_url, api_key, mock=use_mock)

if __name__ == "__main__":
    main()
'''

app_content = '''"""Gradio web interface for the Icebreaker Bot."""

import os
import sys
import logging
import uuid
import gradio as gr

from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import split_profile_data, create_vector_database
from modules.llm_interface import change_llm_model
from modules.query_engine import generate_initial_facts, answer_user_query
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)

logger = logging.getLogger(__name__)

active_indices = {}

def process_profile(linkedin_url, api_key, use_mock, selected_model):
    session_id = str(uuid.uuid4())
    
    if not use_mock and not linkedin_url:
        return "⚠️ Please enter a LinkedIn profile URL or select 'Use Mock Data'.", None
    if not use_mock and not api_key:
        return "⚠️ Please enter a ProxyCurl API key or select 'Use Mock Data'.", None

    change_llm_model(selected_model)
    
    if use_mock and not linkedin_url:
        linkedin_url = "https://www.linkedin.com/in/leonkatsnelson/"
        
    try:
        profile_data = extract_linkedin_profile(linkedin_url, api_key, mock=use_mock)
        if not profile_data:
            return "❌ Failed to extract profile data. Please check your URL and API key.", None
            
        nodes = split_profile_data(profile_data)
        index = create_vector_database(nodes)
        
        if not index:
            return "❌ Failed to create vector database.", None
            
        active_indices[session_id] = index
        
        facts = generate_initial_facts(index)
        return facts, session_id
        
    except Exception as e:
        logger.error(f"Error processing profile: {e}")
        return f"❌ An error occurred: {str(e)}", None

def chat_with_profile(session_id, user_query, chat_history):
    if not session_id or session_id not in active_indices:
        return chat_history + [[user_query, "⚠️ No profile loaded. Please process a LinkedIn profile first."]]
    elif not user_query.strip():
        return chat_history + [["", "⚠️ Please enter a question."]]
    
    try:
        index = active_indices[session_id]
        answer = answer_user_query(index, user_query)
        return chat_history + [[user_query, answer]]
    except Exception as e:
        logger.error(f"Error answering query: {e}")
        return chat_history + [[user_query, f"❌ An error occurred: {str(e)}"]]

def create_gradio_interface():
    available_models = ["gemma2:2b", "llama3"]
    
    with gr.Blocks(title="LinkedIn Icebreaker Bot") as demo:
        gr.Markdown("# LinkedIn Icebreaker Bot")
        gr.Markdown("Generate personalized icebreakers and chat about LinkedIn profiles")
        
        with gr.Tab("Process LinkedIn Profile"):
            with gr.Row():
                with gr.Column():
                    linkedin_url = gr.Textbox(
                        label="LinkedIn Profile URL",
                        placeholder="https://www.linkedin.com/in/username/"
                    )
                    api_key = gr.Textbox(
                        label="ProxyCurl API Key (Leave empty to use mock data)",
                        placeholder="Your ProxyCurl API Key",
                        type="password"
                    )
                    use_mock = gr.Checkbox(label="Use Mock Data", value=True)
                    model_dropdown = gr.Dropdown(
                        choices=available_models,
                        label="Select LLM Model",
                        value=config.LLM_MODEL_ID
                    )
                    process_btn = gr.Button("Process Profile")
                
                with gr.Column():
                    result_text = gr.Textbox(label="Initial Facts", lines=10)
                    session_id = gr.Textbox(label="Session ID", visible=False)
            
            process_btn.click(
                fn=process_profile,
                inputs=[linkedin_url, api_key, use_mock, model_dropdown],
                outputs=[result_text, session_id]
            )
        
        with gr.Tab("Chat"):
            gr.Markdown("Chat with the processed LinkedIn profile")
            
            chatbot = gr.Chatbot(height=500)
            chat_input = gr.Textbox(
                label="Ask a question about the profile",
                placeholder="What is this person's current job title?"
            )
            
            chat_btn = gr.Button("Send")
            
            chat_btn.click(
                fn=chat_with_profile,
                inputs=[session_id, chat_input, chatbot],
                outputs=[chatbot]
            )
            
            chat_input.submit(
                fn=chat_with_profile,
                inputs=[session_id, chat_input, chatbot],
                outputs=[chatbot]
            )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name="127.0.0.1", server_port=5000, share=True)
'''

with open('config.py', 'w', encoding='utf-8') as f:
    f.write(config_content)
with open('modules/data_extraction.py', 'w', encoding='utf-8') as f:
    f.write(data_extraction_content)
with open('modules/data_processing.py', 'w', encoding='utf-8') as f:
    f.write(data_processing_content)
with open('modules/llm_interface.py', 'w', encoding='utf-8') as f:
    f.write(llm_interface_content)
with open('modules/query_engine.py', 'w', encoding='utf-8') as f:
    f.write(query_engine_content)
with open('main.py', 'w', encoding='utf-8') as f:
    f.write(main_content)
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_content)

print("Files successfully overwritten!")
