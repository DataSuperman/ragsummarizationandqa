from typing import List

from langchain_community.document_loaders import WikipediaLoader
from langchain_core.documents import Document


class WikiHandler:

    @staticmethod
    def get_data(query: str, retrieve_max_docs: int = 5) -> List[Document]:
        pages = WikipediaLoader(query=query, load_max_docs=retrieve_max_docs).load()

        return pages
