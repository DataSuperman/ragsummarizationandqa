from abc import ABC, abstractmethod
from typing import List

from langchain.chains import LLMChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.mapreduce import MapReduceDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

from src.config import AppResources


class Summarizer(ABC):

    @staticmethod
    def get_documents_from_text(text: str, chunk_size: int = 16000) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-3.5",
            chunk_size=chunk_size,

            chunk_overlap=100,
        )
        text_list = text_splitter.split_text(text)
        return [Document(doc) for doc in text_list]

    @abstractmethod
    def summarize(self, text: str) -> str:
        pass


class MapReduceSummarizer(Summarizer):

    def __init__(self, res: AppResources):
        self.res = res
        map_template = self.res.config['mapreduce_summarizer']['map_template']
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=self.res.llm, prompt=map_prompt)

        # Todo: Implement this function
        reduce_template = self.res.config['mapreduce_summarizer']['reduce_template']
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = LLMChain(llm=self.res.llm, prompt=reduce_prompt)

        combine_document_chain = StuffDocumentsChain(llm_chain=reduce_chain,
                                                     document_variable_name="docs",
                                                     )
        reduce_document_chain = ReduceDocumentsChain(combine_documents_chain=combine_document_chain,
                                                     collapse_documents_chain=combine_document_chain,
                                                     token_max=1000)
        self.chain = MapReduceDocumentsChain(llm_chain=map_chain,
                                             reduce_documents_chain=reduce_document_chain,
                                             document_variable_name="docs",
                                             return_intermediate_steps=False)

    def summarize(self, text: str) -> str:
        docs = self.get_documents_from_text(text, chunk_size=self.res.config['chunk_size'])
        return self.chain.run(docs)


class RefineSummarizer(Summarizer):

    def __init__(self, res: AppResources):
        self.res = res
        initial_template = self.res.config['refine_summarizer']['prompt_template']
        prompt = PromptTemplate.from_template(initial_template)

        refine_template = self.res.config['refine_summarizer']['refine_template']
        refine_prompt = PromptTemplate.from_template(refine_template)

        refine_chain = load_summarize_chain(llm=self.res.llm,
                                            chain_type="refine",
                                            question_prompt=prompt,
                                            refine_prompt=refine_prompt,
                                            return_intermediate_steps=False,
                                            input_key="input_docs",
                                            output_key="output_text",
                                            document_variable_name='docs'
                                            )
        self.chain = refine_chain

    def summarize(self, text: str) -> str:
        docs = self.get_documents_from_text(text, chunk_size=self.res.config['chunk_size'])
        result = self.chain({'input_docs': docs}, return_only_outputs=True)
        return result['output_text']


class StuffSummarizer(Summarizer):

    def __init__(self, res: AppResources):
        self.res = res
        initial_template = self.res.config['stuff_summarizer']['prompt_template']
        prompt = PromptTemplate.from_template(initial_template)
        llm_chain = LLMChain(llm=self.res.llm, prompt=prompt)
        self.chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="docs")

    def summarize(self, text: str) -> str:
        doc = self.get_documents_from_text(text)
        summary = self.chain.invoke({'input_documents': doc})
        return summary['output_text']


class OpenAISummarizer:

    def __init__(self, res: AppResources, summarizer_type: str):
        self.res = res
        self.summarizer_type = summarizer_type

    def summarize(self, text: str) -> str:
        if self.summarizer_type == 'stuff':
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            num_tokens = len(encoding.encode(text))
            if num_tokens > 16000:
                print("Text is too long, splitting into chunks, falling to MapReduceSummarizer")
                return MapReduceSummarizer(self.res).summarize(text)
            return StuffSummarizer(self.res).summarize(text)
        elif self.summarizer_type == 'mapreduce':
            return MapReduceSummarizer(self.res).summarize(text)

        elif self.summarizer_type == 'refine':
            return RefineSummarizer(self.res).summarize(text)
        else:
            raise ValueError(f"Invalid summarizer type: {self.summarizer_type}, "
                             f"summarizer should be one of the following ['stuff', 'mapreduce', 'refine']")


if __name__ == '__main__':
    from src.config import AppResources
    from pathlib import Path


    # Create Resources object, this class abstracts away everything that the summarizer will need
    app_res = AppResources.from_config_file(config_path=Path('../config_files/config.json'))

    summarizer = OpenAISummarizer(app_res, summarizer_type='stuff')

    summary = summarizer.summarize(dataset.data_as_str)

    print(summary)
