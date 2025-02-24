import os
import requests
from dotenv import load_dotenv

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