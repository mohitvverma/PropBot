import unittest
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from PropBots.constants import Config
from PropBots.CustomPropException import PropBotException
from PropBots.logger import logging
import sys


def PropBotVectorStore(client_path):
    """
    Initializes Qdrant vector stores for text and image collections.

    Args:
        client_path (str): The file path to the Qdrant client.

    Returns:
        StorageContext: A storage context containing vector stores for text and image collections.
    """
    try:
        client = qdrant_client.QdrantClient(path=client_path)

        # Initialize vector stores for text and image collections
        text_store = QdrantVectorStore(client=client, collection_name=Config.TEXT_COLLECTION_NAME)
        image_store = QdrantVectorStore(client=client, collection_name=Config.IMAGE_COLLECTION_NAME)

        # Create storage context with the vector stores
        storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)
        return storage_context

    except Exception as e:
        logging.error(f"Error initializing vector store: {e}")
        raise PropBotException(f"Failed to initialize vector store: {str(e)}", sys)

def add_documents_to_store(dir_path, storage_context):
    """
    Adds documents from a directory to the vector store index.

    Args:
        dir_path (str): The directory path containing documents.
        storage_context (StorageContext): The storage context with initialized vector stores.

    Returns:
        MultiModalVectorStoreIndex: An index containing the added documents.
    """
    try:
        # Load documents from the specified directory
        documents = SimpleDirectoryReader(dir_path).load_data()

        # Create an index from the documents and the provided storage context
        index = MultiModalVectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context
        )
        return index

    except Exception as e:
        logging.error(f"Error adding documents to the store: {e}")
        raise PropBotException(f"Failed to add documents to the store: {str(e)}", sys)


def retreiver(text_similarity_top_k, image_similarity_top_k, index):
    """
    Retrieves similar documents from the index based on text and image similarity.

    Args:
        text_similarity_top_k (int): The number of top similar text results to retrieve.
        image_similarity_top_k (int): The number of top similar image results to retrieve.
        index (MultiModalVectorStoreIndex): The index containing the documents.

    Returns:
        Retriever: A retriever for fetching similar documents.
    """
    try:
        retriever = index.as_retriever(
            similarity_top_k=text_similarity_top_k,
            image_similarity_top_k=image_similarity_top_k
        )
        return retriever

    except Exception as e:
        logging.error(f"Error creating retriever: {e}")
        raise PropBotException(f"Failed to create retriever: {str(e)}", sys)



# Unit Tests
class PropBotVectorStoreTests(unittest.TestCase):
    def setUp(self):
        # Initialize with a dummy client path for testing purposes
        self.client_path = "dummy_path"
        self.storage_context = PropBotVectorStore(self.client_path)
        self.test_dir_path = "/path/to/test/documents"  # Modify with a valid path for actual tests

    def test_vector_store_initialization(self):
        """Test the initialization of the vector store."""
        storage_context = PropBotVectorStore(self.client_path)
        self.assertIsNotNone(storage_context, "Storage context should not be None")

    def test_add_documents_to_store(self):
        """Test adding documents to the vector store."""
        index = add_documents_to_store(self.test_dir_path, self.storage_context)
        self.assertIsNotNone(index, "Index should not be None")

    def test_retriever_creation(self):
        """Test retriever creation from the index."""
        index = add_documents_to_store(self.test_dir_path, self.storage_context)
        retriever = retreiver(5, 5, index)
        self.assertIsNotNone(retriever, "Retriever should not be None")


if __name__ == '__main__':
    unittest.main()
