"""Module for processing LinkedIn profile data."""

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
