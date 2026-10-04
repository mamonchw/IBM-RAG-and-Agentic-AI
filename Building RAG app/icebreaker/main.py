"""Main script for running the Icebreaker Bot."""

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
    print("\n" + "="*50)
    print("INITIAL FACTS:")
    print("="*50)
    print(facts)
    print("="*50 + "\n")
    
    chatbot_interface(index)

def chatbot_interface(index):
    print("Chatbot interface started. Ask questions about the profile!")
    print("Type 'exit', 'quit', or 'bye' to end the conversation.")
    
    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break
                
            if not query.strip():
                continue
                
            response = answer_user_query(index, query)
            print(f"\nBot: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
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
