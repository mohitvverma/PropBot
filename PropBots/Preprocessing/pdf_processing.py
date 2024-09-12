import fitz  # PyMuPDF for handling PDF images
from PyPDF2 import PdfReader
import base64
import os
import sys
import camelot
from PropBots.constants import Config
from PropBots.CustomPropException import PropBotException
from PropBots.logger import logging
from PropBots.Model.MultiModel import PropBotModels

# Initialize models for image summarization
models = PropBotModels()

class PDFProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_pdf_text(self):
        """Extracts and returns text from all pages of the PDF."""
        try:
            reader = PdfReader(self.file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text.strip()
        except Exception as e:
            logging.error(f"Error extracting text from {self.file_path}: {e}")
            raise PropBotException(f"Error extracting text: {str(e)}", sys)

    def extract_pdf_images(self):
        """Extracts images from the PDF and summarizes them using PropBotModels."""
        images_data = []
        try:
            doc = fitz.open(self.file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image.get("image")
                    if image_bytes:
                        # Summarize the image using the model
                        text = models.summarize_image(base64.b64encode(image_bytes).decode('utf-8'))
                        images_data.append({
                            "page": page_num + 1,
                            "index": img_index,
                            "text": text,
                        })
            doc.close()
            logging.info(f"Extracted {len(images_data)} images from {self.file_path}")
        except Exception as e:
            logging.error(f"Error extracting images from {self.file_path}: {e}")
            raise PropBotException(f"Error extracting images: {str(e)}", sys)
        return images_data

    def extract_tables_from_pdf(self):
        """Extracts tables from the PDF using Camelot and returns them in JSON format."""
        try:
            tables = camelot.read_pdf(self.file_path, pages='all')
            json_data = {
                f"table_{i + 1}": table.df.to_dict(orient='records') for i, table in enumerate(tables)
            }
            logging.info(f"Extracted {len(tables)} tables from {self.file_path}")
            return json_data
        except Exception as e:
            logging.error(f"Error extracting tables from {self.file_path}: {e}")
            raise PropBotException(f"Error extracting tables: {str(e)}", sys)

    def process_pdf_files(self):
        """Processes the PDF file to extract text, images, and tables."""
        try:
            pdf_images_and_tables = {
                os.path.basename(self.file_path): {
                    "images": self.extract_pdf_images(),
                    "tables": self.extract_tables_from_pdf(),
                    "text": self.extract_pdf_text()
                }
            }
            logging.info(f"Processed PDF file {self.file_path}")
            return pdf_images_and_tables
        except Exception as e:
            logging.error(f"Error processing PDF {self.file_path}: {e}")
            raise PropBotException(f"Error processing PDF: {str(e)}", sys)
