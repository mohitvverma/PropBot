import re
import os
import fitz  # PyMuPDF for handling PDF images
from PIL import Image
from docx import Document
import base64
from PropBots.Model.MultiModel import PropBotModels
from PropBots.logger import logging


models = PropBotModels()


class ImageDataExtractor:
    def __init__(self, file_path):
        self.file_path = file_path

    # Clean filenames by removing special characters
    def clean_filename(self, filename):
        """
        Clean the filename by removing special characters and keeping only alphanumeric and underscores.
        """
        return re.sub(r'[^a-zA-Z0-9_]', '_', filename)

    # Save image bytes to a file
    def save_image_to_folder(self, image_bytes, filename, folder_path, extension):
        """
        Save image bytes to a folder with a clean filename and appropriate extension.
        """
        try:
            cleaned_filename = f"{self.clean_filename(filename)}.{extension}"
            image_path = os.path.join(folder_path, cleaned_filename)
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            logging.info(f"Saved image: {image_path}")
            return image_path
        except Exception as e:
            logging.error(f"Failed to save image {filename}: {str(e)}")
            raise

    # Save text summaries to a file
    @staticmethod
    def save_summary_to_folder(summary, filename, folder_path):
        """
        Save the image summary as a text file in the specified folder.
        """
        try:
            summary_path = os.path.join(folder_path, filename)
            with open(summary_path, "w") as text_file:
                text_file.write(summary)
            logging.info(f"Saved summary: {summary_path}")
            return summary_path
        except Exception as e:
            logging.error(f"Failed to save summary {filename}: {str(e)}")
            raise

    # Extract images from PDFs
    def extract_images_from_pdf(self, file_path, image_folder, text_folder):
        """
        Extract images from PDF files and save both the image and its summary.
        """
        try:
            doc = fitz.open(file_path)
            base_name = self.clean_filename(os.path.splitext(os.path.basename(file_path))[0])
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image.get("image")
                    extension = base_image.get("ext", "png")
                    if image_bytes:
                        filename = f"{base_name}_P{page_num + 1}_I{img_index + 1}"
                        image_path = self.save_image_to_folder(image_bytes, filename, image_folder, extension)
                        image_summary = models.summarize_image(base64.b64encode(image_bytes).decode('utf-8'))
                        summary_filename = f"{filename}.txt"
                        self.save_summary_to_folder(image_summary, summary_filename, text_folder)
            doc.close()
            logging.info(f"Finished extracting images from PDF: {file_path}")
        except Exception as e:
            logging.error(f"Error extracting images from {file_path}: {str(e)}")
            raise

    # Extract images from Word documents
    def extract_images_from_doc(self, file_path, image_folder, text_folder):
        """
        Extract images from Word documents and save both the image and its summary.
        """
        try:
            doc = Document(file_path)
            base_name = self.clean_filename(os.path.splitext(os.path.basename(file_path))[0])
            for i, rel in enumerate(doc.part.rels):
                if "image" in doc.part.rels[rel].target_ref:
                    image_bytes = doc.part.rels[rel].target_part.blob
                    extension = os.path.splitext(doc.part.rels[rel].target_ref)[1][1:]
                    filename = f"{base_name}_I{i + 1}"
                    image_path = self.save_image_to_folder(image_bytes, filename, image_folder, extension)
                    image_summary = models.summarize_image(base64.b64encode(image_bytes).decode('utf-8'))
                    summary_filename = f"{filename}.txt"
                    self.save_summary_to_folder(image_summary, summary_filename, text_folder)
            logging.info(f"Finished extracting images from Word document: {file_path}")
        except Exception as e:
            logging.error(f"Error extracting images from {file_path}: {str(e)}")
            raise

    # Process individual image files (PNG/JPG)
    def process_image_file(self, file_path, image_folder, text_folder):
        """
        Process individual image files, save them, and generate a summary.
        """
        try:
            with open(file_path, "rb") as img_file:
                image_bytes = img_file.read()
                base_name = self.clean_filename(os.path.splitext(os.path.basename(file_path))[0])
                extension = os.path.splitext(file_path)[1][1:]
                image_path = self.save_image_to_folder(image_bytes, base_name, image_folder, extension)
                image_summary = models.summarize_image(base64.b64encode(image_bytes).decode('utf-8'))
                summary_filename = f"{base_name}.txt"
                self.save_summary_to_folder(image_summary, summary_filename, text_folder)
            logging.info(f"Finished processing image file: {file_path}")
        except Exception as e:
            logging.error(f"Error processing image file {file_path}: {str(e)}")
            raise

    # Scan folder and process all supported files
    def process_folder_images(self, folder_path, output_image_folder_path, output_text_folder_path):
        """
        Scan a folder and process all image, PDF, and Word files within it.
        """
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)

                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.process_image_file(file_path, output_image_folder_path, output_text_folder_path)

                elif file.lower().endswith(".pdf"):
                    self.extract_images_from_pdf(file_path, output_image_folder_path, output_text_folder_path)

                elif file.lower().endswith(".docx"):
                    self.extract_images_from_doc(file_path, output_image_folder_path, output_text_folder_path)

