"""
This module handles the configuration settings for the Longchain environment.
It includes a data class `ChainConf` that holds all necessary configuration variables.
"""

import os
from dataclasses import dataclass

@dataclass
class ChainConf:
    """
    Data class holding the Longchain environment configuration variables.

    Attributes:
        local_file_directory (str): The local directory where transcript files are stored.
        activeloop_org_id (str): The organization ID for the DeepLake dataset.
        activelooop_dataset_name (str): The name of the DeepLake dataset.
        embedding_model (str): The model name for text embeddings.
        llm_model (str): The model name for language processing (LLM).
    """

    local_file_directory: str = "./transcripts"
    
    activeloop_org_id: str = "mahmoudfazeli"
    activelooop_dataset_name: str = "WorkShops_VectorDB"
    
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "text-davinci-003"

    chat_model: str = "gpt-4-1106-preview"

# Additional environment-specific configurations can be added here as needed
