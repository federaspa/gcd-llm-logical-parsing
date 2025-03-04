from datetime import datetime
import logging
import sys
import os
import requests
from dotenv import load_dotenv

def get_logger(script_name, debug: bool = False):
    current_datetime = datetime.now().strftime("%d%m%Y_%H%M%S")
    logger = logging.getLogger(__name__)
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    # log_file_name = f"logs/{script_name}_{current_datetime}.log"
    # file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
    # file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

class PushoverNotifier:
    def __init__(self):
        load_dotenv()
        self.user_key = os.getenv('PUSHOVER_USER_KEY')
        self.app_token = os.getenv('PUSHOVER_APP_TOKEN')
        self.api_url = "https://api.pushover.net/1/messages.json"
        
    def send(self, message: str, title: str = None):
        data = {
            "token": self.app_token,
            "user": self.user_key,
            "message": message
        }
        if title:
            data["title"] = title
            
        requests.post(self.api_url, data=data)