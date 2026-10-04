"""Gradio web interface for the Icebreaker Bot."""

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
        return chat_history + [{"role": "user", "content": user_query}, {"role": "assistant", "content": "⚠️ No profile loaded. Please process a LinkedIn profile first."}]
    elif not user_query.strip():
        return chat_history + [{"role": "assistant", "content": "⚠️ Please enter a question."}]
    
    try:
        index = active_indices[session_id]
        answer = answer_user_query(index, user_query)
        return chat_history + [{"role": "user", "content": user_query}, {"role": "assistant", "content": str(answer)}]
    except Exception as e:
        logger.error(f"Error answering query: {e}")
        return chat_history + [{"role": "user", "content": user_query}, {"role": "assistant", "content": f"❌ An error occurred: {str(e)}"}]

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
