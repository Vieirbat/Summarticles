from grobid_client.grobid_client import GrobidClient
import pandas as pd

from article import Article

from collections import Counter

host = 'http://localhost'
port = 8070

client = GrobidClient(host, port)

def process_file(file):
    rsp = client.serve("processFulltextDocument", file)
    tei = rsp.text
    article = Article(rsp.text)
    return(article)

print(process_file(file="data/TesteArtigo"))