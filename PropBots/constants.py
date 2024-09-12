import os
from pathlib import Path

os.chdir('../')

class Config:
    PARENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    MAIN_FILE_PATH = project_path = Path(__file__).parent.parent


    # Directories and Paths
    OUTPUT_DIR_NAME = "Artifacts"
    OUTPUT_PATH = "/Users/mohitverma/Documents/PropBot"
    FULL_DOC_FILE_PATH = "/Users/mohitverma/Documents/RealState_Chatbot/Data"
    EXTRACTED_IMAGE_FOLDER_NAME = 'extracted_images'
    EXTRACTED_OUTPUT_JSON_FILE_NAME = 'full_doc_data.json'
    EXTRACTED_IMAGE_SUMMARY_DIR_NAME = 'extracted_images_summary'
    COMBINED_IMAGE_TEXT_DIR_NAME = 'full_retreived_docs'
    VECTORSTORE_DIR_NAME = 'PropVectorStore'

    # Constants related to models
    MODEL_ID = "HuggingFaceH4/zephyr-7b-alpha"
    VISION_MODEL_NAME = "gpt-4-vision-preview"
    GPT_MODEL_NAME = 'gpt-4o'

    # Token and Model settings
    TEMPERATURE = 0.01
    DO_SAMPLE = False
    MAX_LENGTH = 4096
    MAX_NEW_TOKEN = 4096
    MAX_TOKENS_VISION = 1024
    DEVICE = 'cuda'

    # Supported file types
    SUPPORTED_FILE_TYPES = ['pdf', 'docx', 'xlsx', 'png', 'csv']

    # Agent configurations
    MAX_ITERATIONS = 5000
    AGENT_MAX_ITERATION_LIMIT = 100000
    AGENT_MAX_VALUE_LENGTH = 50000

    # Miscellaneous settings
    TEXT_COLLECTION_NAME = "text_collection_0"
    IMAGE_COLLECTION_NAME = "image_collection_0"
    IMAGE_SYSTEM_MESSAGE = "You are a bot that is good at analyzing images."
    VERBOSE = False

    @staticmethod
    def initialize_directories():
        # Ensure required directories exist
        os.makedirs(Config.OUTPUT_PATH, exist_ok=True)

# Initialize directories on script load
Config.initialize_directories()
