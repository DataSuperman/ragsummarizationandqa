import traceback
from pathlib import Path

from src.config import AppResources
from src.data_handling import WikiHandler
from src.question_answering_system import VectorDatabase, QuestionAnsweringSystem
from src.summarizer import OpenAISummarizer


def run_summarizer_flow(app_res: AppResources, query: str, summarizer_type: str = 'stuff'):
    try:
        wiki = WikiHandler()
        pages = wiki.get_data(query)
        summarizer = OpenAISummarizer(app_res, summarizer_type)
        summary = summarizer.summarize(pages)
        print(summary)
    except Exception as e:
        print(f"Following error occured, {e}")
        return


def run_qa_flow(app_res: AppResources, query: str):
    wiki = WikiHandler()
    page = wiki.get_data(query)
    db = VectorDatabase(app_res)
    db.add_documents(page)
    retriever = db.get_retriever()
    qa_system = QuestionAnsweringSystem(app_res, retriever)
    while True:
        print("-" * 50)

        question = input("Question (or nothing to quit): ")

        if not question:
            print("returning to main menu")
            return

        try:
            generated_answer = qa_system.run(question)
        except Exception as e:
            print(f"Following error occured, {traceback.format_exc()}")
            continue

        print(f"Answer: {generated_answer}")


def main():
    _script_dir = Path(__file__).resolve().parent
    default_summarizer_type = 'stuff'
    app_resources = AppResources.from_config_file()
    default_query = "Python programming"
    while True:
        print("*" * 50)

        system_prompt = "Type s for summary, q for question answering system, x if you want to quit \n"
        choice = input(system_prompt).strip()

        if choice not in ['s', 'q', 'x']:
            print("Invalid choice")
            continue
        elif choice == 'x':
            break
        elif choice == 's':
            query = input(f"Provide your query:  ") or default_query
            summarizer_type = input(f"Type of summarizer ({default_summarizer_type}): ") or default_summarizer_type
            run_summarizer_flow(app_res=app_resources,
                                query=query,
                                summarizer_type=summarizer_type)

        elif choice == 'q':
            query = input(f"Provide your query :  ") or default_query
            run_qa_flow(app_res=app_resources, query=query)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting the program")
