import os
import sys
import re

sys.path.insert(0,os.path.dirname(os.getcwd()))
sys.path.insert(0,os.path.join(os.getcwd(),'grobid'))
sys.path.insert(0,os.getcwd())

import numpy as np
import pandas as pd

from grobid import grobid_client
import grobid_tei_xml
from grobid_to_dataframe import grobid_cli, xmltei_to_dataframe

import streamlit as st

# docker run -t --rm --init -p 8080:8070 -p 8081:8071 --memory="9g" lfoppiano/grobid:0.7.0
# docker run -t --rm --init -p 8080:8070 -p 8081:8071 lfoppiano/grobid:0.6.2

path = os.path.dirname(os.getcwd())

def get_path(path_input_path):
    """"""
    if os.path.exists(path_input_path):
        return path_input_path
    
    return os.getcwd()

def batch_process_path(path_input_path, config_path="./grobid/config.json"):
    """"""
    gcli = grobid_cli(config_path=config_path)
    result_batch = gcli.process_pdfs(input_path=path_input_path,
                                     n_workers=2,
                                     service="processFulltextDocument",
                                     generateIDs=True,
                                     include_raw_citations=True,
                                     include_raw_affiliations=True,
                                     consolidate_header=False,
                                     consolidate_citations=False,
                                     tei_coordinates=False,
                                     segment_sentences=True,
                                     verbose=True)
    return result_batch


def get_dataframes(result_batch):
    """"""
    xml_to_df = xmltei_to_dataframe()
    dict_dfs = xml_to_df.get_dataframe_articles(result_batch)
    
    return dict_dfs
    

def run_batch_process():

    path_input = os.path.join(path,'artifacts','test_article')
    path_input_path = get_path(path_input)
    
    result_batch = batch_process_path(path_input_path)
    dict_dfs = get_dataframes(result_batch)
    
    print('[Process has been finished!!!]')
    
    return dict_dfs

# Para execução do streamlit e dos processos 
if __name__ == '__main__':
    
    
    st.title("**[APP] Sumarticles: Materials Science**")
    st.markdown("Application", unsafe_allow_html=False) # st.write("Application")
    
    dict_dfs = run_batch_process()
    
    #st.progress(0)
    #st.progress(100)
    
    st.dataframe(dict_dfs['df_doc_info'], width=None, height=500)
    
    # st.table(dict_dfs['df_doc_info'])
    
    
    
    
