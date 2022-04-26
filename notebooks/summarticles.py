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

import plotly.express as px
import tkinter as tk
from tkinter import filedialog

import time

import streamlit as st

# docker run -t --rm --init -p 8080:8070 -p 8081:8071 --memory="9g" lfoppiano/grobid:0.7.0
# docker run -t --rm --init -p 8080:8070 -p 8081:8071 lfoppiano/grobid:0.6.2

path = os.path.dirname(os.getcwd())
global input_path

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
    

def run_batch_process(path_input):

    # path_input = os.path.join(path,'artifacts','test_article')
    path_input_path = get_path(path_input)
    
    result_batch = batch_process_path(path_input_path)
    dict_dfs = get_dataframes(result_batch)
    
    print('[Process has been finished!!!]')
    
    return dict_dfs


def files_path(path):
    list_dir = os.listdir(path)
    files = []
    for file in list_dir:
        if os.path.isfile(os.path.join(path,file)):
            files.append(os.path.join(path,file))
    return files

def chars_graph(dict_dfs):
    
    list_chars = []
    for id,row in dict_dfs['df_doc_info'].iterrows():
        for c in row['raw_data']:
            list_chars.append(c)
            
    df_counts = pd.DataFrame({'chars':pd.value_counts(list_chars).index.tolist(),'counts':pd.value_counts(list_chars).tolist()})
    df_counts = df_counts.sort_values(by='counts',ascending=False)
    
    fig = px.bar(df_counts.head(20), x='chars', y='counts')
    
    return fig


# For process execution and streamlit app
if __name__ == '__main__':
    
    
    # Head
    st.set_page_config(
        page_title="[APP] Summarticles: Materials Science ⚛👨‍🔬👩‍🔬",
        page_icon="⚛", # https://www.freecodecamp.org/news/all-emojis-emoji-list-for-copy-and-paste/, https://share.streamlit.io/streamlit/emoji-shortcodes
        layout="wide", # centered
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""<h1 style="text-align:center;">⚛👨‍🔬👩‍🔬 Summarticles: Materials Science</h1>""",
                unsafe_allow_html=True)
    
    st.markdown("""<h3 style="text-align:center;"><b>Summarticles is an application to summarize articles information, 
                    using IA and analytics.</b></h3>""", 
                unsafe_allow_html=True) # st.write("Application")
    
    st.markdown("""<h6 style="text-align:center;">Do you want to use it? So, you only need to specify a folder path clicking 
                on '📁 Get folder path!' and let the magic happen!</h6>""",
                unsafe_allow_html=True)
    
    # st.header("")
    
    # Set up tkinter
    root = tk.Tk()
    root.withdraw()
    
    # Make folder picker dialog appear on top of other windows
    root.wm_attributes('-topmost', 1)
    
    # Sidebar Menu
    with st.sidebar:
        st.markdown("""<p style="text-align:center;font-size:40px;">Menu Sidebar</p>""",unsafe_allow_html=True)
    
    # button containers 
    _, btn_col1, _ = st.columns([2.25,3,1])
    
    input_folder_path = input_path = ""
    
    with btn_col1:
        btn_getfolder = st.button('📁 Get folder path!',key='btn_getfolder')
    st.markdown("""<p style="text-align:center;font-size:12px;">This application works only with PDF files.</p>""", unsafe_allow_html=True)
    
    if btn_getfolder:
        
        st.warning("⚠️ Feature in development!")
        input_folder_path = st.text_input('Selected folder:',filedialog.askdirectory(master=root), key='txt_input_path')

        if not len(input_folder_path) or pd.isna(input_folder_path) or input_folder_path == "":
            st.error(f"❌ You need to specify a valid folder path, click on **'📁 Get folder path!'** Folder path: {str(input_folder_path)}!")
        elif not os.path.exists(input_folder_path):
            st.error(f"❌ The path doesn't exist! You need to specify a valid folder path! Folder path: {str(input_folder_path)}")
        elif not len(files_path(input_folder_path)):
            st.error(f"❌ There are no files in this folder path! You need to specify a valid folder path! Folder path: {str(input_folder_path)}")
        else:
            st.success(f"✔️ **In this folder path we found: {str(len(files_path(input_folder_path)))} files!** Folder path: {str(input_folder_path)}")
            
            st.warning('⚡ Running batch process!')
    
            dict_dfs = run_batch_process(input_folder_path)
            
            st.write("**[Articles Data] df_doc_info:**")
            st.dataframe(dict_dfs['df_doc_info'], width=None, height=None)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Read Articles", str(dict_dfs['df_doc_info'].shape[0]), str(dict_dfs['df_doc_info'].shape[0]))
            col2.metric("Count Chars", str(dict_dfs['df_doc_info'].raw_data.apply(len).sum()), str(-dict_dfs['df_doc_info'].raw_data.apply(len).sum()))
            col3.metric("Mean Chars",str(dict_dfs['df_doc_info'].raw_data.apply(len).mean()), str(dict_dfs['df_doc_info'].raw_data.apply(len).mean()))
        
            st.plotly_chart(chars_graph(dict_dfs),use_container_width=True)
        
            st.write("**[Head Articles Data] df_doc_head:**")
            st.dataframe(dict_dfs['df_doc_head'], width=None, height=None)
            
            st.write("**[Head Articles Data] df_doc_head:**")
            st.dataframe(dict_dfs['df_doc_head'], width=None, height=None)
            
            st.write("**[Authors Articles Data] df_doc_authors:**")
            st.dataframe(dict_dfs['df_doc_authors'], width=None, height=None)
            
            st.write("**[Citations Articles Data] df_doc_citations:**")
            st.dataframe(dict_dfs['df_doc_citations'], width=None, height=None)
            
            st.write("**[Authors Citations Articles Data] df_doc_authors_citations:**")
            st.dataframe(dict_dfs['df_doc_authors_citations'], width=None, height=None)
    
    
