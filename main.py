import os
import sys
from PIL import Image
import streamlit as st

from PropBots.CustomPropException import PropBotException
# from PropBots import Config
# from PropBots.constants import AGENT_MAX_ITERATION_LIMIT
from PropBots.utils import (
    convert_jpeg_to_png_in_folder, move_files_to_new_folder,
    ensure_serializable, save_json, plot_images, open_json
)
from PropBots.constants import Config
from PropBots.Preprocessing.Processor import process_property_docs
from PropBot.PropVectorStore.vector_store import (
    PropBotVectorStore, add_documents_to_store, retreiver
)
from PropBots.Preprocessing.ImageProcessing.ImageExtractor import ImageDataExtractor
from PropBots.AgenticRetreival.JSON_retreiver import AgentExecuter, ProcessQuery
from PropBots.Model.MultiModel import PropBotModels
from PropBots.logger import logging
from PropBots.prompt import CustomPrompt

models = PropBotModels()


class PropBotInitiater:
    def __init__(self):
        logging.info("Initializing PropBotInitiater class")

    def pdfprocessor(self):
        try:
            logging.info("Starting PDF processing")
            json_path = os.path.join(Config.OUTPUT_PATH, Config.OUTPUT_DIR_NAME, Config.EXTRACTED_OUTPUT_JSON_FILE_NAME)
            logging.info(f"Created JSON path {json_path}")

            if not os.path.exists(json_path):
                os.makedirs(json_path, exist_ok=True)
                logging.info("Successfully Created JSON directory")
                data_dict = process_property_docs(base_dir=Config.FULL_DOC_FILE_PATH)
                serialize_json_data = ensure_serializable(data_dict)
                json_file_name = os.path.join(Config.OUTPUT_PATH, Config.EXTRACTED_OUTPUT_JSON_FILE_NAME)
                save_json(json_output_path=json_file_name, property_data_serializable=serialize_json_data)
                logging.info("PDF processing completed and JSON saved.")
                print(json_path)
                return json_path

            else:
                print("JSON file already exists")
                logging.info("JSON Directory already exists at the path. No need to create directory.")
                return json_path

        except Exception as e:
            print("PDF Processor failed due to exception")
            logging.error(f"Error occurred in pdfprocessor: {str(e)}")
            raise PropBotException(str(e), sys)

    def process_image(self):
        try:
            logging.info("Starting image processing")
            image_folder_path = os.path.join(Config.OUTPUT_PATH, Config.OUTPUT_DIR_NAME, Config.EXTRACTED_IMAGE_FOLDER_NAME)
            image_text_folder_path = os.path.join(Config.OUTPUT_PATH, Config.OUTPUT_DIR_NAME, Config.EXTRACTED_IMAGE_SUMMARY_DIR_NAME)

            if not os.path.exists(image_folder_path) or not os.path.exists(image_text_folder_path):
                os.makedirs(image_text_folder_path, exist_ok=True)
                os.makedirs(image_folder_path, exist_ok=True)
                logging.info("Image folders created successfully")

                image_extractor = ImageDataExtractor(Config.FULL_DOC_FILE_PATH)
                image_extractor.process_folder_images(
                    Config.FULL_DOC_FILE_PATH, image_folder_path, image_text_folder_path
                )

                convert_jpeg_to_png_in_folder(image_folder_path)
                logging.info("Image processing completed successfully")
                return image_folder_path, image_text_folder_path

            else:
                print("Process image exist")
                logging.info("Image folders already exist.")
                return image_folder_path, image_text_folder_path

        except Exception as e:
            logging.error(f"Error occurred in process_image: {str(e)}")
            raise PropBotException(str(e), sys)

    def combining_images_text_folder(self, image_dir_path, image_text_dir_path):
        try:
            logging.info("Combining image and text folders")
            full_path_dir_name = os.path.join(Config.OUTPUT_PATH, Config.OUTPUT_DIR_NAME, Config.COMBINED_IMAGE_TEXT_DIR_NAME)

            if not os.path.exists(full_path_dir_name):
                os.makedirs(full_path_dir_name, exist_ok=True)
                move_files_to_new_folder(
                    images_folder=image_dir_path, text_folder=image_text_dir_path,
                    target_folder=full_path_dir_name
                )
                logging.info("Combining images and text folders completed successfully")
                return full_path_dir_name

            else:
                print("Combining images and text folders already exist.")
                logging.info("Combined directory already exists.")
                return full_path_dir_name

        except Exception as e:
            logging.error(f"Error occurred in combining_images_text_folder: {str(e)}")
            raise PropBotException(str(e), sys)

    def creating_vector_space(self):
        try:
            logging.info("Creating vector space directory")
            vectorspace_path = os.path.join(Config.MAIN_FILE_PATH, Config.VECTORSTORE_DIR_NAME)

            if not os.path.exists(vectorspace_path):
                os.makedirs(vectorspace_path, exist_ok=True)
                logging.info("Vector space directory created successfully")
                return vectorspace_path

            else:
                print('Vector space directory already exists')
                logging.info("Vector space directory already exists.")
                return vectorspace_path

        except Exception as e:
            logging.error(f"Error occurred in creating_vector_space: {str(e)}")
            raise PropBotException(str(e), sys)

    def QueryRetreiver(self, vectorspace_path, full_path_dir_name):
        try:
            logging.info("Initializing Query Retreiver")
            storage_context = PropBotVectorStore(client_path=vectorspace_path)
            StorageIndex = add_documents_to_store(dir_path=full_path_dir_name, storage_context=storage_context)
            index_retriever = retreiver(text_similarity_top_k=1, image_similarity_top_k=1, index=StorageIndex)
            logging.info("Query Retreiver initialized successfully")
            return index_retriever

        except Exception as e:
            logging.error(f"Error occurred in QueryRetreiver: {str(e)}")
            raise PropBotException(str(e), sys)

    def Agent_query(self, data, llm, max_value_length, max_iterations_limit):
        try:
            logging.info("Initializing Agent Executer")
            agent = AgentExecuter(
                data=data, llm=llm, max_value_length=max_value_length,
                max_iterations_limit=max_iterations_limit
            )
            logging.info("Agent initialized successfully")
            return agent

        except Exception as e:
            logging.error(f"Error occurred in Agent_query: {str(e)}")
            raise PropBotException(str(e), sys)

    def query_executer(self, query, agent, index):
        try:
            logging.info("Executing query")
            result = ProcessQuery(query=query, agent=agent)

            print(result)
            image_result = index.retrieve(query)

            prompt = CustomPrompt(agent_data=result, summary_data=image_result[0].node.text, query=query)
            result = models.chat_model().invoke(prompt)

            final_result = {
                'agent_result': result.content,
                'image_path': image_result[1].node.image_path
            }
            logging.info("Query executed successfully")
            return final_result

        except Exception as e:
            logging.error(f"Error occurred in query_executer: {str(e)}")
            raise PropBotException(str(e), sys)


if 'user_input' not in st.session_state:
    st.session_state.user_input = ''

if 'agent' not in st.session_state:
    st.session_state.agent = None



@st.cache_resource
def initiate():
    try:
        prop = PropBotInitiater()

        json_path = prop.pdfprocessor()
        json_data = open_json(json_path)

        image_folder_path, image_text_folder_path = prop.process_image()
        image_summary_dir_path = prop.combining_images_text_folder(
            image_dir_path=image_folder_path, image_text_dir_path=image_text_folder_path
        )

        vectorspace_path = prop.creating_vector_space()
        index = prop.QueryRetreiver(
            vectorspace_path=vectorspace_path, full_path_dir_name=image_summary_dir_path
        )

        agent = prop.Agent_query(
            data=json_data, llm=models.chat_model(),
            max_value_length=Config.AGENT_MAX_VALUE_LENGTH,
            max_iterations_limit=Config.AGENT_MAX_ITERATION_LIMIT
        )
        return agent, index
    except Exception as e:
        logging.error(f"Error occurred in initiate: {str(e)}")
        raise PropBotException(str(e), sys)


# with st.spinner("Initializing Chatbot..."):
#     st.session_state.agent = initiate()
#     st.write("Initialization Complete")
#
# st.session_state.user_input = st.text_input("Enter your question:", key="user")
#
#
# if st.button("Submit"):
#     if st.session_state.user_input:
#         try:
#             bot = PropBotInitiater()
#             agent, index = initiate()
#             response = bot.query_executer(
#                 query=st.session_state.user_input, agent=agent,
#                 index=index
#             )
#
#             if response.get("agent_result"):
#                 st.write("**Bot Response:**")
#                 st.text(response["agent_result"])
#
#             # if response.get("image_summary_result"):
#             #     st.write("**Image Summary Result:**")
#             #     st.text(response["image_summary_result"])
#
#             if response.get("image_path"):
#                 image_path = response["image_path"]
#                 if os.path.exists(image_path):
#                     image = Image.open(image_path)
#                     st.image(image, caption="Answer Image")
#                 else:
#                     st.warning("Image not found.")
#             st.success("Query processed successfully!")
#
#         except PropBotException as e:
#             st.error(f"An error occurred: {str(e)}")
#         except Exception as e:
#             st.error(f"Unexpected error: {str(e)}")
#             logging.error(f"Unexpected error during query execution: {str(e)}")


# Define Streamlit UI
st.set_page_config(page_title="Proplens Q&A System", page_icon="🏠", layout="wide")
st.title("Proplens Q&A System with Text and Image Responses")

# Add a sidebar for branding and information
st.sidebar.image("/Users/mohitverma/Downloads/house_PNG50.png", use_column_width=True)
st.sidebar.title("Welcome to Proplens")
st.sidebar.info("Proplens helps you navigate property-related queries with ease. Explore our intelligent Q&A system!")

# # Custom CSS for black background and white text
# st.markdown("""
#     <style>
#     .main {
#         background-color: black;
#         color: white;
#         padding: 20px;
#     }
#     .stButton>button {
#         background-color: #4CAF50;
#         color: white;
#         border-radius: 10px;
#         padding: 10px 20px;
#     }
#     .stTextInput>div>input {
#         background-color: black;
#         color: white;
#         border: 2px solid #4CAF50;
#         border-radius: 5px;
#     }
#     .css-1v0mbdj {
#         font-size: 16px;
#         font-weight: bold;
#         color: white;
#     }
#     .footer {
#         text-align: center;
#         font-size: 12px;
#         padding: 10px 0;
#         color: #888;
#     }
#     .sidebar .sidebar-content {
#         background-color: black;
#     }
#     </style>
# """, unsafe_allow_html=True)


# Footer with contact info
st.markdown("<div class='footer'>Developed by Mohit Verma for Proplens. For support, contact me at mohitvvermaa@outlook.com</div>", unsafe_allow_html=True)


with st.spinner("Initializing Proplens Q&A System..."):
    agent, index = initiate()
    st.session_state.agent = (agent, index)
    st.success("Initialization Complete!")


# Input and Action Section
st.session_state.user_input = st.text_input("Ask your property-related question here:", key="user", placeholder="Type your question...")

if st.button("Submit"):
    if st.session_state.user_input:
        try:
            bot = PropBotInitiater()
            agent, index = st.session_state.agent
            response = bot.query_executer(
                query=st.session_state.user_input, agent=agent, index=index
            )

            if response.get("agent_result"):
                st.subheader("**Bot Response:**")
                st.write(response["agent_result"])

            if response.get("image_path"):
                image_path = response["image_path"]
                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    st.image(image, caption="Answer Image", use_column_width=True)
                else:
                    st.warning("Image not found.")
            st.success("Your query was processed successfully!")

        except PropBotException as e:
            st.error(f"An error occurred: {str(e)}")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            logging.error(f"Unexpected error during query execution: {str(e)}")


# Additional Features Section
st.sidebar.subheader("Additional Features:")
st.sidebar.write("- Multi-modal Q&A with text and image responses.")
st.sidebar.write("- High accuracy document and image retrieval.")
st.sidebar.write("- Scalable solutions for property management and assessment.")
