FROM python:3.12

WORKDIR /PGC

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*


COPY PGC/requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# Numpy packages
RUN python -m nltk.downloader punkt_tab
RUN python -m nltk.downloader wordnet

# Ollama 0.6.3
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN ollama pull deepseek-r1:1.5b
RUN ollama pull llama3.2:1b

COPY PGC/ .

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]