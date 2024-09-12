import base64
from PropBot.Model.MultiModel import PropBotModels

models=PropBotModels()

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def process_image(file_path):
    # Extract text via OCR
    image_text = models.summarize_image(encode_image(file_path))

    return {
        'text': image_text
    }