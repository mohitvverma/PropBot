from PropBots.logger import logging
from PropBots.CustomPropException import PropBotException
from PIL import Image
import os
import shutil
import matplotlib.pyplot as plt
import json


def convert_jpeg_to_png_in_folder(folder_path):
    try:
        # Loop through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpeg") or filename.endswith(".jpg"):
                jpeg_path = os.path.join(folder_path, filename)
                png_filename = f"{os.path.splitext(filename)[0]}.png"
                png_path = os.path.join(folder_path, png_filename)

                # Open the JPEG image
                img = Image.open(jpeg_path)
                img.save(png_path, 'PNG')
                logging.info(f"Converted {filename} to {png_filename}")
    except FileNotFoundError as e:
        logging.error(f"Folder not found: {folder_path} - {e}")
        raise PropBotException(f"Folder not found: {folder_path}") from e
    except Exception as e:
        logging.error(f"Error converting JPEG to PNG: {e}")
        raise PropBotException("Error converting JPEG to PNG") from e


def ensure_serializable(data):
    try:
        if isinstance(data, dict):
            return {key: ensure_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [ensure_serializable(item) for item in data]
        elif isinstance(data, set):
            return list(data)  # Convert set to list
        elif hasattr(data, 'isoformat'):  # Handle datetime-like objects
            return data.isoformat()
        else:
            return data
    except Exception as e:
        logging.error(f"Serialization error: {e}")
        raise PropBotException("Error ensuring data is serializable") from e


def move_files_to_new_folder(images_folder, text_folder, target_folder):
    try:
        os.makedirs(target_folder, exist_ok=True)
        logging.info(f"Target folder '{target_folder}' created.")

        def move_files(source_folder, target_folder):
            for root, dirs, files in os.walk(source_folder):
                for file in files:
                    source_file = os.path.join(root, file)
                    target_file = os.path.join(target_folder, file)

                    if os.path.exists(target_file):
                        logging.warning(f"File {file} already exists. Renaming to avoid overwriting.")
                        base, extension = os.path.splitext(file)
                        counter = 1
                        while os.path.exists(target_file):
                            target_file = os.path.join(target_folder, f"{base}_{counter}{extension}")
                            counter += 1

                    shutil.move(source_file, target_file)
                    logging.info(f"Moved {source_file} to {target_file}")

        move_files(images_folder, target_folder)
        move_files(text_folder, target_folder)
        logging.info("Files moved successfully.")
    except FileNotFoundError as e:
        logging.error(f"Folder not found: {e}")
        raise PropBotException(f"Folder not found: {e}") from e
    except Exception as e:
        logging.error(f"Error moving files: {e}")
        raise PropBotException("Error moving files") from e


def save_json(json_output_path, property_data_serializable):
    try:
        with open(json_output_path, 'w') as json_file:
            json.dump(property_data_serializable, json_file, indent=4)
            logging.info(f"JSON data saved successfully to {json_output_path}")
    except Exception as e:
        logging.error(f"Error saving JSON: {e}")
        raise PropBotException("Error saving JSON") from e


def plot_images(image_paths):
    try:
        images_shown = 0
        plt.figure(figsize=(16, 9))
        for img_path in image_paths:
            if os.path.isfile(img_path):
                image = Image.open(img_path)
                plt.subplot(4, 4, images_shown + 1)
                plt.imshow(image)
                plt.xticks([])
                plt.yticks([])
                images_shown += 1
                if images_shown >= 9:
                    break
        plt.show()
        logging.info("Images plotted successfully.")
    except FileNotFoundError as e:
        logging.error(f"Image file not found: {e}")
        raise PropBotException(f"Image file not found: {e}") from e
    except Exception as e:
        logging.error(f"Error plotting images: {e}")
        raise PropBotException("Error plotting images") from e


def open_json(json_path):
    try:
        with open(json_path, 'r') as fp:
            data = json.load(fp)
            logging.info(f"JSON file {json_path} loaded successfully.")
            return data
    except FileNotFoundError as e:
        logging.error(f"JSON file not found: {json_path} - {e}")
        raise PropBotException(f"JSON file not found: {json_path}") from e
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {json_path} - {e}")
        raise PropBotException(f"Error decoding JSON: {json_path}") from e
    except Exception as e:
        logging.error(f"Error opening JSON: {json_path} - {e}")
        raise PropBotException(f"Error opening JSON: {json_path}") from e
