from loguru import logger
from pymongo import MongoClient
from loguru import logger
import requests

# User input for MongoDB connection details
mongodb_url = input("MongoDB URL: ")
db_name = input("Database Name: ")
collection_name = input("Collection Name: ")

# Configure Splunk HEC details
splunk_url = "https://your-splunk-hec-url"
splunk_token = "your-splunk-hec-token"

# Create a custom handler for sending logs to Splunk HEC
class SplunkHECHandler:
    def __init__(self, url, token):
        self.url = url
        self.token = token

    def write(self, record):
        data = {
            "event": record["message"],
        }
        headers = {
            "Authorization": f"Splunk {self.token}",
        }

        response = requests.post(self.url, json=data, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to send log to Splunk HEC. Status Code: {response.status_code}")


class MongoDBHandler:
    def __init__(self, connection_url, db_name, collection_name):
        self.connection_url = connection_url
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None

    def __enter__(self):
        self.client = MongoClient(self.connection_url)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.client.close()

def configure_logger(log_handlers):
    loguru_config = {
        "handlers": log_handlers,
        "extra": {"name": "your_logger_name"},
    }
    logger.configure(**loguru_config)

# Initialize the Splunk HEC handler
splunk_handler = SplunkHECHandler(splunk_url, splunk_token)


if __name__ == "__main__":
    # Choose log handlers based on user input
    selected_handlers = []

    console_output = input("Do you want to log to console? (yes/no): ").lower()
    if console_output == "yes":
        selected_handlers.append({"sink": "sys.stdout", "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"})
        
    if splunk_output == 'yes':
        selected_handlers.append({"sink": splunk_handler.write, "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"})
    
    if mongo_output == 'yes':
        # Create a MongoDB handler
        with MongoDBHandler(mongodb_url, db_name, collection_name) as mongo_handler:
            # Add the MongoDB handler
            selected_handlers.append({"sink": mongo_handler.collection.insert_one, "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"})

    # Configure the logger with the selected handlers
    configure_logger(selected_handlers)

    # Log some example messages
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")