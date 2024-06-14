FROM langchain/langchain:latest
WORKDIR /app
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN ollama pull llama3:latest
ADD . .
ENV PYTHONPATH "${PYTHONPATH}:/app"
CMD ["python",  "src/run.py"]
