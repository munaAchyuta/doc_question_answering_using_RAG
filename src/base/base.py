from loguru import logger

from .globals import get_global_config

config_file_path = "./config/config.yaml"
GLOBAL = get_global_config(config_file_path)

logger.add(GLOBAL.get("log_path", "./log/file_{time}.log"),
           format="{time} {level} {message}",
           backtrace=True,
           diagnose=True,
           enqueue=True,
           level="TRACE",
           retention="30 days",
           rotation="1 MB")


class BaseClass:
    '''
    holds global level attributes.
    '''

    def __init__(self) -> None:
        self.env_vars = GLOBAL
        self.app_name = GLOBAL.get("app_name", "nl_to_sql")
        self.log_path = GLOBAL.get("log_path", None)
        
        self.openai_url = GLOBAL.get("openai_url","https://api.openai.com/v1/completions")
        self.openai_token = GLOBAL.get("openai_token", None)
        self.openai_model = GLOBAL.get("openai_model", "text-davinci-003")
        self.openai_max_token = GLOBAL.get("openai_max_token", 1000)
        self.openai_temperature = GLOBAL.get("openai_temperature", 0.9)

        self.doc_retrieval_config = GLOBAL.get('doc_retrieval_config')
        self.logger = logger

    @classmethod
    def call_api(cls, body={}, headers={}):
        pass
