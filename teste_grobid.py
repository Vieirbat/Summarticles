import os
import sys
import re

import numpy as np
import pandas as pd

from grobid import grobid_client

host = 'http://localhost'
port = 8070

path = os.path.dirname(os.getcwd())
path_input = os.path.join(path,'artifacts','test_article')
path_output = os.path.join(path,'output','xml')
path_article = os.path.join(path,'artifacts','test_article','b617684b.pdf')

if __name__ == "__main__":
    client = grobid_client.GrobidClient(config_path="./grobid/config.json")
    client.process("processFulltextDocument", # Para pegar todo o texto do documento
                    path_input, # local onde estará os pdfs dos artigos
                    output=path_output, # local onde estará a saída em xml
                    force=True,
                    n=1)