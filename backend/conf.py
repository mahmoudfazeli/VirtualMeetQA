"""
Environment variables handling.
"""
import os
from dataclasses import dataclass
from typing import Dict


@dataclass
class ChainConf:
    """
    Data class holding the Longchain environment configuration variables
    """

    #gcp_project_id: str = os.environ.get("GCP_PROJECT_ID")
    #gcs_bucket_data: str = os.environ.get("GCS_BUCKET_DATA")
    local_file_directory: str = "./transcripts"
    
    activeloop_org_id: str = "##################"
    activelooop_dataset_name: str = "###############"
    #path_to_local_db: str = "/data_base"
    #local_dataset_name: str = "WorkShops_VectorDB"
    
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "text-davinci-003"

    #def __post_init__(self):
     #   if not isinstance(self.gcp_project_id, str) or not self.gcp_project_id:
     #       raise ValueError("Invalid gcp_project_id value")
     #   if not isinstance(self.gcs_bucket_data, str) or not self.gcs_bucket_data:
     #       raise ValueError("Invalid gcs_bucket_data value")