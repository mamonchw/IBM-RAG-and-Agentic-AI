"""Module for interfacing with Ollama LLMs."""

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
