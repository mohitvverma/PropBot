import logging
from datetime import datetime
import os
from PropBots.constants import Config

# Create a directory for logs if it doesn't exist
LOG_DIR = os.path.join(Config.MAIN_FILE_PATH, 'Logs_data')
os.makedirs(LOG_DIR, exist_ok=True)

# Log file configuration
LOG_FILE_NAME = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

# Define the log format as a string for JSON logger
LOG_FORMAT = '%(asctime)s %(filename)s %(lineno)d %(levelname)s %(message)s'

logs_path = os.path.join(os.getcwd(), 'Logs_data', LOG_FILE_NAME)

if not os.path.exists(logs_path):
    os.makedirs(logs_path, exist_ok=True)
else:
    pass

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE_NAME)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format=LOG_FORMAT,
    level=logging.INFO
)

logging.info("hello")