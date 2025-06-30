FROM python:3.12

WORKDIR /PGC

RUN apt-get update && apt-get install -y 
# \
#     build-essential \
#     curl \
#     software-properties-common \
#     git \
#     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

COPY . /PGC

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# # Numpy packages
# RUN python -m nltk.downloader punkt_tab
# RUN python -m nltk.downloader wordnet

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# CMD ["streamlit", "hello", "--server.port=8501", "--server.address=0.0.0.0"]

CMD ["streamlit", "run", "summarticles.py", "--server.port=8501", "--server.address=0.0.0.0"]