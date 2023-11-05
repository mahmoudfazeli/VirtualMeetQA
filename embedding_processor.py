import logging

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

from backend.conf import ChainConf

def generate_embedding(chain_conf: ChainConf):
    """
    Function generates embeddings for a list of documents and adds them to a Deep Lake dataset
    """
    if not isinstance(chain_conf, ChainConf):
        raise ValueError("Invalid chain_conf value. Expected an instance of ChainConf.")

    logging.info("Starting generate_embedding function")

    # load files

    loader = DirectoryLoader(
        chain_conf.local_file_directory,
        '*.pdf'
        )

    documents = loader.load()
    logging.info(f"Loaded {len(documents)} documents")

    # generate embeddings
    text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)

    # Update Deep Lake dataset
    dataset_path = (
        f"hub://{chain_conf.activeloop_org_id}/{chain_conf.activelooop_dataset_name}"
    )
    
    embeddings = OpenAIEmbeddings(model=chain_conf.embedding_model)

    db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

    # add documents to our Deep Lake dataset
    db.add_documents(documents)
    logging.info(f"Added {len(documents)} documents to Deep Lake dataset")
    logging.info("Finished generate_embedding function")
    return db

if __name__ == "__main__":
    chain_conf = ChainConf()
    generate_embedding(chain_conf)