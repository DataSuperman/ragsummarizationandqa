from datetime import datetime
from pathlib import Path
from typing import List, Union, Tuple
import datetime

from langchain import hub
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

from src.config import AppResources

# todo : Add in the future the ability to use different vector databases
_VECTORDATABASES = {
    "faiss": FAISS,
    "chroma": Chroma,
}


def get_documents_from_text(text: str, chunk_size: int = 16000) -> list:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5",
        chunk_size=chunk_size,

        chunk_overlap=100,
    )
    text_list = text_splitter.split_text(text)
    return [Document(doc) for doc in text_list]


class VectorDatabase:

    def __init__(self, config: AppResources):
        self.config = config
        if self.config.config['llm_type'] == 'openai':
            self.embedding_model = OpenAIEmbeddings(openai_api_key=self.config.config['openai']['api_key'], )
        if self.config.config['llm_type'] == 'llama3':
            self.embedding_model = OllamaEmbeddings(model="llama3")
        vector_store = self.config.config['vector_database']['database_name']
        if vector_store not in _VECTORDATABASES:
            raise ValueError(f"Vector store {vector_store} not supported")
        self.database_class = _VECTORDATABASES[vector_store]
        current_file_path = Path('__file__').resolve()
        vector_dir = current_file_path.parent / 'vector_store'
        if not vector_dir.exists():
            vector_dir.mkdir(exist_ok=True)
        self.vector_store = None

    def add_documents(self, documents: Union[List[Document], str]):
        document_list = []
        if isinstance(documents, str):
            document_list = get_documents_from_text(documents, chunk_size=self.config.config['chunk_size'])
        if isinstance(documents, list):
            for doc in documents:
                document_list.extend(get_documents_from_text(doc, chunk_size=self.config.config['chunk_size']))
        else:
            raise ValueError("Documents must be a list of strings or a string")
        self.vector_store = self.database_class.from_documents(documents=document_list, embedding=self.embedding_model)

    def get_retriever(self):
        return self.vector_store.as_retriever()


class DocumentGrader(BaseModel):
    """ Grade the relevance check on retrieved documents as a binary score"""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")


# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")


class QuestionAnsweringSystem:

    def __init__(self, config: AppResources, retriever: VectorStoreRetriever):
        self.config = config
        self.llm = self.config.llm
        self.retriever = retriever
        self.generate_answer_prompt = hub.pull("rlm/rag-prompt")

    def get_document_relevance_as_binary(self, question: str) -> Tuple[DocumentGrader, List[Document]]:
        grader_system_prompt = self.config.config['question_answering']['grader_system_prompt']
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", grader_system_prompt),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ])
        structured_llm_grader = self.llm.with_structured_output(DocumentGrader)
        retrieval_grader = grade_prompt | structured_llm_grader
        docs = self.retriever.invoke(question)
        score_obj: DocumentGrader = DocumentGrader(binary_score="no")
        for i, _ in enumerate(docs):
            doc_txt = docs[i].page_content
            score_obj: DocumentGrader = retrieval_grader.invoke({"question": question, "document": doc_txt})
            if score_obj.binary_score == "yes":
                break
        return score_obj, docs

    def generate_answer(self, question: str, grade_score: DocumentGrader, docs: List[Document]) -> str:

        rag_chain = self.generate_answer_prompt | self.llm | StrOutputParser()
        if grade_score.binary_score == "yes":
            generated_answer = rag_chain.invoke({"question": question, "context": docs})
        else:
            generated_answer = "could not find relevant answer"
        return generated_answer

    def is_not_hallucination(self, generated_answer: str, docs: List[Document]) -> GradeHallucinations:
        structured_llm_hallucination_grader = self.llm.with_structured_output(GradeHallucinations)
        # Prompt
        system = self.config.config['question_answering']['hallucination_system_prompt']
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )
        hallucination_grader = hallucination_prompt | structured_llm_hallucination_grader
        hallucination_score = hallucination_grader.invoke({"documents": docs, "generation": generated_answer})
        return hallucination_score

    def verify_answer_relevance(self, generated_answer: str, question: str) -> GradeAnswer:
        structured_llm_answer_grader = self.llm.with_structured_output(GradeAnswer)
        # Prompt
        system = self.config.config['question_answering']['answer_relevance_system_prompt']
        answer_relevance_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Generated answer: \n\n {answer} \n\n User question: {question}"),
            ]
        )
        answer_relevance_grader = answer_relevance_prompt | structured_llm_answer_grader
        answer_score = answer_relevance_grader.invoke({"answer": generated_answer, "question": question})
        return answer_score

    def run(self, question: str) -> str:
        grade_score, docs = self.get_document_relevance_as_binary(question)
        generated_answer = self.generate_answer(question, grade_score, docs)
        is_not_hallucination_score = self.is_not_hallucination(generated_answer, docs)
        if is_not_hallucination_score.binary_score == "yes":
            print("Answer is  grounded in the facts")
        answer_score = self.verify_answer_relevance(generated_answer, question)
        if answer_score.binary_score == "no":
            print("Answer does not address the question")
        return generated_answer
