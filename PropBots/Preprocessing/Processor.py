from PropBots.constants import Config
from PropBots.Preprocessing.DocProcessor import load_and_convert_docx_to_json
from PropBots.Preprocessing.ImageProcessor import process_image
from PropBots.Preprocessing.Excel_Processor import extract_excel_data
from PropBots.Preprocessing.pdf_processing import PDFProcessor
import os


# Main function to walk through the directories and extract data
def process_property_docs(base_dir):
    property_data = {}

    try:
        # Traverse all directories and subdirectories
        for root, dirs, files in os.walk(base_dir):
            property_name = os.path.basename(root)
            if property_name not in property_data:
                property_data[property_name] = {}

            for file in files:
                file_path = os.path.join(root, file)
                file_ext = file.split('.')[-1].lower()  # Get file extension

                if file_ext not in Config.SUPPORTED_FILE_TYPES:
                    print(f"Skipping unsupported file type: {file}")
                    continue

                try:
                    if file_ext == 'pdf':
                        # Extract PDF text
                        print(f"Processing PDF: {file_path}")
                        property_data[property_name][file] = PDFProcessor(file_path=file_path).process_pdf_files()

                    elif file_ext == 'docx':
                        # Extract Word document text
                        print(f"Processing DOCX: {file_path}")
                        property_data[property_name][file] = load_and_convert_docx_to_json(file_path)

                    elif file_ext == 'xlsx':
                        # Extract Excel tables
                        print(f"Processing Excel: {file_path}")
                        property_data[property_name][file] = extract_excel_data(file_path)

                    elif file_ext == 'png':
                        # Process image
                        print(f"Processing Image: {file_path}")
                        property_data[property_name][file] = process_image(file_path)

                except Exception as file_error:
                    print(f"Error processing file {file}: {file_error}")

    except Exception as e:
        print(f"An error occurred while processing: {e}")
        raise

    return property_data


#Example usage
if __name__ == "__main__":
    base_directory = Config.FULL_DOC_FILE_PATH
    data = process_property_docs(base_directory)
    print(data)

