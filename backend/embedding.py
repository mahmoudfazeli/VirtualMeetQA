import logging

#from langchain.document_loaders import GCSDirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake
from .conf import ChainConf

def load_retriever(chain_conf: ChainConf):
    """
    Load retriever from Deeplake.
    """
    if (
        not isinstance(chain_conf.activeloop_org_id, str)
        or not chain_conf.activeloop_org_id
    ):
        raise ValueError("Invalid activeloop_org_id value")
    if (
        not isinstance(chain_conf.activelooop_dataset_name, str)
        or not chain_conf.activelooop_dataset_name
    ):
        raise ValueError("Invalid activelooop_dataset_name value")
    if (
        not isinstance(chain_conf.embedding_model, str)
        or not chain_conf.embedding_model
    ):
        raise ValueError("Invalid embedding_model value")
    dataset_path = (
        f"hub://{chain_conf.activeloop_org_id}/{chain_conf.activelooop_dataset_name}"
    )
    embeddings = OpenAIEmbeddings(model=chain_conf.embedding_model)
    db = DeepLake(dataset_path=dataset_path, embedding=embeddings, read_only = True)
    retriever = db.as_retriever()
    return retriever