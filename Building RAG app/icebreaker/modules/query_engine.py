"""Module for querying indexed LinkedIn profile data."""

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
