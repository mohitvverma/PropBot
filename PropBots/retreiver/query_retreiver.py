from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader


def retreiver(text_similarity_top_k, image_similarity_top_k, index):
    retreiver = index.as_retriever(similarity_top_k=text_similarity_top_k, image_similarity_top_k=image_similarity_top_k)
    return retreiver


