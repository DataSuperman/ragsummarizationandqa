from pathlib import Path
import json

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

_script_dir = Path(__file__).resolve().parent
_config_file_path = _script_dir / '../config_files/config.json'


class AppResources:

    def __init__(self, config):
        self.config = config
        if self.config['llm_type'] == 'openai':
            # os.environ['OPENAI_API_KEY'] = self.config['openai']['api_key']
            self.llm = ChatOpenAI(temperature=0, model=self.config['openai']['model_name'],
                                  openai_api_key=self.config['openai']['api_key'])
            self.max_tokens = self.config['openai']['max_tokens']
        if self.config['llm_type'] == 'llama3':
            self.llm = ChatOllama(model="llama3", format="json", temperature=0)
        else:
            raise ValueError(f"LLM type {self.config['llm_type']} not supported")

    @classmethod
    def from_config_file(cls, config_path=_config_file_path):
        file_text = config_path.read_text()
        config = json.loads(file_text)
        return cls(config)


if __name__ == '__main__':
    sc = AppResources.from_config_file()
    print(sc.config)
