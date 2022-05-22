import os
import sys
import shutil
from distutils.dir_util import copy_tree
import re

sys.path.insert(0,os.path.dirname(os.getcwd()))
sys.path.insert(0,os.path.join(os.getcwd(),'grobid'))
sys.path.insert(0,os.getcwd())

import numpy as np
import pandas as pd

from grobid import grobid_client
import grobid_tei_xml
from grobid_to_dataframe import grobid_cli, xmltei_to_dataframe
from text import text_prep, text_mining, text_viz

import plotly.express as px
import tkinter as tk
from tkinter import filedialog

from customMsg import customMsg

import time

import streamlit as st
import streamlit.components.v1 as components

import networkx as nx
import matplotlib.pyplot as plt
from graph.pyvis.network import Network

# docker run -t --rm --init -p 8080:8070 -p 8081:8071 --memory="9g" lfoppiano/grobid:0.7.0
# docker run -t --rm --init -p 8080:8070 -p 8081:8071 lfoppiano/grobid:0.6.2

path = os.path.dirname(os.getcwd())
global input_path

gcli = grobid_client.GrobidClient(config_path=os.path.join(path,"grobid/config.json"))


def get_path(path_input_path):
    """"""
    if os.path.exists(path_input_path):
        return path_input_path
    
    return os.getcwd()


def batch_process_path(path_input_path, n_workers=10, check_cache=True, cache_folder_name='summarticles_cache', config_path="./grobid/config.json"):
    
    """"""
    
    gcli = grobid_cli(config_path=config_path)
    result_batch = gcli.process_pdfs(input_path=path_input_path,
                                     check_cache=check_cache,
                                     cache_folder_name=cache_folder_name,
                                     n_workers=n_workers,
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
    dict_dfs, dict_errors = xml_to_df.get_dataframe_articles(result_batch)
    
    return dict_dfs, dict_errors


def files_path(path):
    
    """"""
    
    list_dir = os.listdir(path)
    files = []
    for file in list_dir:
        if os.path.isfile(os.path.join(path,file)):
            files.append(os.path.join(path,file))
    return files

def clean_error_results(result_batch):
    
    """"""
    
    new_result = []
    for result in result_batch:
        if "500" not in str(result[1]):
            new_result.append(result)
    return new_result


def run_batch_process(st, path_input, cache_folder_name='summarticles_cache', n_workers=10, display_articles_data=True, save_xmltei=True):

    """"""

    # path_input = os.path.join(path,'artifacts','test_article')
    input_folder_path = get_path(path_input)
    
    print('[Path]:',input_folder_path)
    
    if not len(input_folder_path) or pd.isna(input_folder_path) or input_folder_path == "":
        st.error(f"❌ You need to specify a valid folder path, click on **'📁 Get folder path!'** Folder path: {str(input_folder_path)}!")
    elif not os.path.exists(input_folder_path):
        st.error(f"❌ The path doesn't exist! You need to specify a valid folder path! Folder path: {str(input_folder_path)}")
    elif not len(files_path(input_folder_path)):
        st.error(f"❌ There are no files in this folder path! You need to specify a valid folder path! Folder path: {str(input_folder_path)}")
    elif not gcli.check_typefile_inpath(files_path(input_folder_path)):
        st.error(f"❌ There are no files in this folder with APP required file type! Please make sure if that path is the correct path!")
    else:
        
        st.success(f"✔️ **In this folder path we found: {str(len(files_path(input_folder_path)))} files!** Folder path: {str(input_folder_path)}")
        # msg = customMsg('⚡ Running batch process!','warning')
        # st.warning('⚡ Running batch process!')
        
        if len(files_path(input_folder_path)) < 2:
            st.error(f"❌ You need to specify a path with at least two pdf files!")
            return None
        
        with st.spinner('⚡ Running batch process...'):
            
            result_batch = batch_process_path(input_folder_path, n_workers= n_workers)
            result_batch = clean_error_results(result_batch)
            
            if not len(result_batch):
                st.error(f"⚠️ Something is wrong, I can't get any result! 😕 Please, look if you selected the correct file path or if files in the selected path have information to extract!")
                return None
            
            dict_dfs, dict_errors = get_dataframes(result_batch)

            if save_xmltei:
                gcli.save_xmltei_files(result_batch, input_folder_path, cache_folder_name=cache_folder_name)

            if display_articles_data and len(result_batch):
                with st.spinner('🧾 Showing articles information...'):
                    show_articles_data(st, dict_dfs)
    
        print('[Process has been finished!!!]')
        #msg.empty()
        #msg = customMsg('⚡ Finish process!','warning')
        
        return dict_dfs
    
    return None


def chars_graph(dict_dfs):
    
    """"""
    
    list_chars = []
    for id,row in dict_dfs['df_doc_info'].iterrows():
        for c in row['raw_data']:
            list_chars.append(c)
            
    df_counts = pd.DataFrame({'chars':pd.value_counts(list_chars).index.tolist(),'counts':pd.value_counts(list_chars).tolist()})
    df_counts = df_counts.sort_values(by='counts',ascending=False)
    
    fig = px.bar(df_counts.head(20), x='chars', y='counts')
    
    return fig


def tk_configs():
    
    """"""
    
    # Set up tkinter
    tk_root = tk.Tk()
    tk_root.withdraw()
    
    # Make folder picker dialog appear on top of other windows
    tk_root.wm_attributes('-topmost', 1)
    
    return tk_root


def _max_width_(st):
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


def make_sidebar(st):
    
    """"""

    with st.sidebar:
        st.markdown("""<p style="text-align:center;font-size:40px;">Menu Sidebar</p>""", unsafe_allow_html=False)


def make_head(st):

    """"""

    # Head
    st.set_page_config(
        page_title="[APP] Summarticles: Materials Science ⚛👨‍🔬👩‍🔬",
        page_icon="⚛", # https://www.freecodecamp.org/news/all-emojis-emoji-list-for-copy-and-paste/, https://share.streamlit.io/streamlit/emoji-shortcodes
        layout="wide", # centered
        initial_sidebar_state="collapsed", #collapsed #auto #expanded
        menu_items={"About":"https://github.com/Vieirbat/PGC",
                    "Get help":"https://github.com/Vieirbat/PGC",
                    "Report a bug":"https://github.com/Vieirbat/PGC"}) 
    
    st.markdown("""<h1 style="text-align:center;">⚛👨‍🔬👩‍🔬 Summarticles: Materials Science</h1>""",
                unsafe_allow_html=True)
    
    st.markdown("""<h3 style="text-align:center;"><b>Summarticles is an application to summarize articles information, 
                    using IA and analytics.</b></h3>""", 
                unsafe_allow_html=True) # st.write("Application")
    
    st.markdown("""<h6 style="text-align:center;">Do you want to use it? So, you only need to specify a folder path clicking 
                on '📁 Select a folder path!' and let the magic happen!</h6>""",
                unsafe_allow_html=True)


def make_getpath_button(st):
    
    """"""
    
    _, btn_col1, _ = st.columns([3,3,1])
    
    with btn_col1:
        btn_getfolder = st.button('📁 Select a folder path!',key='btn_getfolder')
        st.markdown("""<p style="text-align:left;font-size:11px;">This application only works with PDF files.</p>""", unsafe_allow_html=True)
    
    return btn_getfolder


def show_articles_data(st, dict_dfs):
    
    """"""
    
    st.write("**[Articles Data] df_doc_info (with 5 rows sample):**")
    st.dataframe(dict_dfs['df_doc_info'].head(5).astype(str), width=None, height=None)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Read Articles", str(dict_dfs['df_doc_info'].shape[0]), str(dict_dfs['df_doc_info'].shape[0]))
    col2.metric("Count Chars", str(dict_dfs['df_doc_info'].raw_data.apply(len).sum()), str(-dict_dfs['df_doc_info'].raw_data.apply(len).sum()))
    col3.metric("Mean Chars",str(dict_dfs['df_doc_info'].raw_data.apply(len).mean()), str(dict_dfs['df_doc_info'].raw_data.apply(len).mean()))

    st.plotly_chart(chars_graph(dict_dfs),use_container_width=True)
    
    st.write("**[Head Articles Data] df_doc_head (with 5 rows sample):**")
    st.dataframe(dict_dfs['df_doc_head'].head(5).astype(str), width=None, height=None)
    
    st.write("**[Authors Articles Data] df_doc_authors (with 5 rows sample):**")
    st.dataframe(dict_dfs['df_doc_authors'].head(5).astype(str), width=None, height=None)
    
    st.write("**[Citations Articles Data] df_doc_citations (with 5 rows sample):**")
    st.dataframe(dict_dfs['df_doc_citations'].head(5).astype(str), width=None, height=None)
    
    st.write("**[Authors Citations Articles Data] df_doc_authors_citations (with 5 rows sample):**")
    st.dataframe(dict_dfs['df_doc_authors_citations'].head(5).astype(str), width=None, height=None)


def show_word_cloud(st, dict_dfs, input_path, folder_images='app_images', wc_image_name='wc_image.png',
                    cache_folder_name='summarticles_cache'):
    
    """"""
    
    tviz = text_viz()
    tprep = text_prep()

    # st.warning("🛠🧾 Text in preparation for WordCloud!")
    dict_dfs['df_doc_info']['abstract_prep'] = tprep.text_preparation_column(dict_dfs['df_doc_info']['abstract'])
    documents = dict_dfs['df_doc_info']['abstract_prep'].fillna(' ').tolist()
    
    path_images = os.path.join(input_path, cache_folder_name, folder_images)
    if not os.path.exists(path_images):
        os.mkdir(path_images)

    wc, ax, fig = tviz.word_cloud(documents, 
                                  path_image=os.path.join(path_images, wc_image_name), 
                                  show_wc=False, 
                                  width=1000, 
                                  height=500, 
                                  collocations=True, 
                                  background_color='white')
    st.markdown("""<h3 style="text-align:left;"><b>WordCloud:</b></h3>""",unsafe_allow_html=True)
    # st.markdown("""<h5 style="text-align:left;"><b>WordCloud:</b></h5>""",unsafe_allow_html=True)
    st.pyplot(fig)


def cossine_similarity_data(st, dict_dfs, column='abstract',n_sim=200,
                            percentil="99%", sim_value_min=0, sim_value_max=0.99):
    
    """"""

    tmining = text_mining()
    
    with st.spinner('📄✢📄 Making similarity relations...'):
        
        documents = dict_dfs['df_doc_info'][column + '_prep'].fillna(' ').tolist()
        df_tfidf_abstract = tmining.get_df_tfidf(documents)
        
        df_cos_tfidf_sim = tmining.get_cossine_similarity_matrix(df_tfidf_abstract,
                                                                dict_dfs['df_doc_info'].index.tolist())
        
    with st.spinner('📑🔍 Filtering best similarities relations...'):
        
        df_cos_tfidf_sim_filter = tmining.filter_sim_matrix(df_cos_tfidf_sim,
                                                            df_cos_tfidf_sim.index.tolist(),
                                                            percentil=percentil,
                                                            value_min=sim_value_min,
                                                            value_max=sim_value_max)
    
    return  df_cos_tfidf_sim_filter


def nodes_data(st, dict_dfs, df_cos_sim_filter):
    
    """"""
    
    with st.spinner('📄➞📄 Similarity Graph: extract nodes information...'):
        
        # Selecting head article data
        cols_head = ['title_head', 'doi_head', 'date_head',]
        head_data = dict_dfs['df_doc_head'].loc[:,cols_head].reset_index().copy()
        head_data['title_head'] = head_data['title_head'].apply(lambda e: str(e)[0:50] + "..." if len(str(e)) > 50 else str(e))

        # Selecting head article data
        cols_info = ['abstract','file']
        doc_info_data = dict_dfs['df_doc_info'].loc[:,cols_info].reset_index().copy()
        doc_info_data['file_name'] = doc_info_data['file'].apply(lambda e: os.path.split(e)[-1])
        doc_info_data['abstract_short'] = doc_info_data['abstract'].apply(lambda e: str(e)[0:20] + "..." if len(str(e)) > 20 else str(e))
        doc_info_data.drop(labels=['abstract'], axis=1, inplace=True)

        # Selecting authors information
        authors_data = dict_dfs['df_doc_authors'].reset_index()
        authors_data = authors_data.groupby(by=['pdf_md5'], as_index=False)['full_name_author'].count()
        authors_data.rename(columns={'full_name_author':'author_count'}, inplace=True)

        # Selecting citations information
        citations_data = dict_dfs['df_doc_citations'].reset_index()
        citations_data = citations_data.groupby(by=['pdf_md5'], as_index=False)['index_citation'].count()
        citations_data.rename(columns={'index_citation':'citation_count'}, inplace=True)

        nodes = list(set(df_cos_sim_filter.doc_a.tolist() + df_cos_sim_filter.doc_b.tolist()))
        df_nodes = pd.DataFrame(nodes, columns=['pdf_md5'])

        df_nodes = df_nodes.merge(head_data, how='left', on='pdf_md5')
        df_nodes = df_nodes.merge(doc_info_data, how='left', on='pdf_md5')
        df_nodes = df_nodes.merge(authors_data, how='left', on='pdf_md5')
        df_nodes = df_nodes.merge(citations_data, how='left', on='pdf_md5')

    return df_nodes


def similarity_graph(st, dict_dfs, input_folder_path, folder_graph='graphs', name_file="graph.html", cache_folder_name='summarticles_cache', 
                     column='abstract', n_sim=200, percentil="99%", sim_value_min=0, sim_value_max=0.99, buttons=False):
    
    """"""
    
    tmining = text_mining()
    
    df_cos_tfidf_sim_filter = cossine_similarity_data(st, dict_dfs, column, n_sim, percentil,
                                                      sim_value_min, sim_value_max)
    
    df_nodes = nodes_data(st, dict_dfs, df_cos_tfidf_sim_filter)
    
    path_write_graph = os.path.join(input_folder_path, cache_folder_name)

    sim_graph, path_graph, path_folder_graph = tmining.make_sim_graph(matrix=df_cos_tfidf_sim_filter, node_data=df_nodes,
                                                   source_column="doc_a", to_column="doc_b", value_column="value",
                                                   height="500px", width="100%", directed=False, notebook=False,
                                                   bgcolor="#ffffff", font_color=False, layout=None, 
                                                   heading="Similarity Graph: this graph shows you similarity across articles.",
                                                   path_graph=path_write_graph, folder_graph=folder_graph, buttons=buttons,
                                                   name_file=name_file)
    with st.container():
        show_graph_graph(sim_graph, path_graph, path_folder_graph)


def show_graph_graph(sim_graph, path_graph, path_folder_graph):
    
    """"""
    
    path = os.path.dirname(os.getcwd())
    path_graph_depend = os.path.join(path,'notebooks','graph','pyvis','templates','dependencies')
    path_depend_dst = os.path.join(path_folder_graph,'dependencies')
    
    if not os.path.exists(path_depend_dst):
        os.mkdir(path_depend_dst)
    
    list_files = os.listdir(path_graph_depend)
    for file in list_files:
        f_source = os.path.join(path_graph_depend, file)
        f_destiny = os.path.join(path_depend_dst, file)
        shutil.copy(f_source, f_destiny)
        
    with st.spinner('👁‍🗨 Similarity Graph: drawing...'):
        
        GraphHtmlFile = open(path_graph, 'r', encoding='utf-8')
        GraphHtml = GraphHtmlFile.read()
        
        css_inject = open(os.path.join(path_graph_depend,'vis-network.css'), 'r', encoding='utf-8')
        css_inject_str = css_inject.read()
        css_inject.close()
        css_inject_str = f"""<style type="text/css">{css_inject_str}</style>"""
        
        script_inject = open(os.path.join(path_graph_depend,'vis-network.min.js'), 'r', encoding='utf-8')
        script_inject_str = script_inject.read()
        script_inject.close()
        script_inject_str = f"""<script type="text/javascript">{script_inject_str}</script>"""
        
        str_replace_css = """<link rel="stylesheet" href="./dependencies/vis.min.css" type="text/css" />"""
        str_replace_script = """<script type="text/javascript" src="./dependencies/vis-network.min.js"> </script>"""
        
        GraphHtml = GraphHtml.replace(str_replace_css, css_inject_str)
        GraphHtml = GraphHtml.replace(str_replace_script, script_inject_str)
        components.html(GraphHtml, height=600, width=None, scrolling=True)
        GraphHtmlFile.close()


def text_preparation(st, dict_dfs, input_folder_path):
    
    """"""
    
    tprep = text_prep()
    
    # dict_dfs['df_doc_info']['acknowledgement_prep'] = tprep.text_prep_column(dict_dfs['df_doc_info']['acknowledgement'])
    dict_dfs['df_doc_info']['abstract_prep'] = tprep.text_preparation_column(dict_dfs['df_doc_info']['abstract'])
    dict_dfs['df_doc_info']['body_prep'] = tprep.text_preparation_column(dict_dfs['df_doc_info']['body'])
    
    return dict_dfs
    
    

def btn_clicked_folder(st, input_folder_path, n_workers, show_wordcloud=True, show_similaritygraph=True,
                       cache_folder_name='summarticles_cache'):

    """"""
    
    # cache path files
    
    
    # Run and display batch process, return a dictionary of dataframes with all data extract from articles
    dict_dfs = run_batch_process(st, input_folder_path, n_workers=n_workers, cache_folder_name=cache_folder_name,
                                 display_articles_data=False, save_xmltei=True)
    
    if not dict_dfs:
        return None
    
    if not len(dict_dfs.keys()) or not dict_dfs['df_doc_info'].shape[0]:
        st.error("❓ There is no information to extract from articles in the specified path! Please, choose another fila path.")
    
    with st.spinner('🛠️📄 Text prepatation...'):
        dict_dfs = text_preparation(st, dict_dfs, input_folder_path)
    
    # Process continues with another things...
    if show_wordcloud:
        with st.spinner('📄➞☁️ Making WordCloud...'):
            show_word_cloud(st, dict_dfs, input_folder_path, cache_folder_name=cache_folder_name,
                            folder_images='images', wc_image_name='wc_image.png')

    if show_similaritygraph:
        with st.spinner('📄➞📄  Making Similarity Graph...'):
            similarity_graph(st, dict_dfs, input_folder_path, percentil="75%", n_sim=100, 
                             cache_folder_name=cache_folder_name,)


def choose_filepath(st):
    
    """"""
    
    with st.spinner(' 📁 Choosing a file path...'):
        # Get articles files path
        st.warning("⚠️ Feature in development!")
        input_folder_path = input_path = ""
        input_folder_path = st.text_input('Selected folder:',filedialog.askdirectory(master=tk_root), key='txt_input_path')
    return input_folder_path
    

###############################################################################################
# ---------------------------------------------------------------------------------------------
# For process execution and streamlit app

if __name__ == '__main__':
    
    # ----------------------------------------------------------------------------
    # ATTENTION THIS CODE NEED SOME MODULARIZATION AND ORGANIZATION
    # BUT AT THIS TIME WE ONLY START APP DEVELOP
    
    # ----------------------------------------------------------------------------
    # Entrance of app
    make_head(st) # st.header("")
    
    # ----------------------------------------------------------------------------
    # TKINTER configs for get file path with filebox dialog
    tk_root = tk_configs()
    
    # ----------------------------------------------------------------------------
    # Sidebar Menu    
    make_sidebar(st)
    
    # ----------------------------------------------------------------------------
    # button getpath containers
    btn_getfolder = make_getpath_button(st)
    
    # ----------------------------------------------------------------------------
    # If button clicked, then start APP process with some verifications
    if btn_getfolder:
        input_folder_path = choose_filepath(st)
        with st.spinner('💻⚙️ Process running...'):
            btn_clicked_folder(st, input_folder_path, n_workers=10)
            