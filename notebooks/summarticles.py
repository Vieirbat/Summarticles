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
import nltk

from keybert import KeyBERT

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

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


def show_macro_numbers(st, dict_dfs):
    
    st.markdown("""<h3 style="text-align:left;"><b>Articles Macro Numbers</b></h3>""", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🧾 Read Articles", str(dict_dfs['df_doc_info'].shape[0]))
    col2.metric("👥 Total Authors", str(dict_dfs['df_doc_authors'].shape[0]))
    col3.metric("📄➞📄 Total Citations",str(dict_dfs['df_doc_citations'].shape[0]))
    col4.metric("👥➞👥 Total Authors from Citations",str(dict_dfs['df_doc_authors_citations'].shape[0]))

    # st.plotly_chart(chars_graph(dict_dfs),use_container_width=True)


def text_statistics(text):
    
    """"""
    
    tprep = text_prep()
    
    dictStats = {}
    dictStats['num_chars'] = len(text)
    list_tokens = tprep.text_tokenize(str(text))
    dictStats['num_words'] = len(list_tokens)
    dictStats['num_words_unique'] = len(set(list_tokens))
    len_words = pd.Series([len(w) for w in list_tokens]).describe().to_dict()
    dictStats['mean_lenght_word'] = len_words['mean']
    dictStats['min_lenght_word'] = len_words['min']
    dictStats['max_lenght_word'] = len_words['max']
    dictStats['std_lenght_word'] = len_words['std']
    dictStats['first_quartile_lenght_word'] = len_words['25%']
    dictStats['second_quartile_median_lenght_word'] = len_words['50%']
    dictStats['third_quartile_lenght_word'] = len_words['75%']
    
    return dictStats

    
def show_text_numbers(st, dict_dfs):
    
    """"""
    
    tprep = text_prep()
    
    documents_abs = dict_dfs['df_doc_info']['abstract_prep'].fillna(' ').tolist()
    documents_body = dict_dfs['df_doc_info']['body_prep'].fillna(' ').tolist()
    documents_all_text = [' '.join([abst, body]) for abst, body in zip(documents_abs, documents_body)]
    
    # ------------------------------------------------------------------------
    # Making stats
    
    df_articles_stats = pd.DataFrame(list(map(lambda e: text_statistics(e), documents_all_text)))
    dict_agg_stats = {}

    # Chars
    dict_agg_stats['num_total_chars'] = df_articles_stats['num_chars'].sum()
    dict_agg_stats['num_mean_chars'] = df_articles_stats['num_chars'].mean()
    dict_agg_stats['num_min_chars'] = df_articles_stats['num_chars'].min()
    dict_agg_stats['num_max_chars'] = df_articles_stats['num_chars'].max()

    # num_words
    dict_agg_stats['num_total_words'] = df_articles_stats['num_words'].sum()
    dict_agg_stats['num_mean_words'] = df_articles_stats['num_words'].mean()
    dict_agg_stats['num_min_words'] = df_articles_stats['num_words'].min()
    dict_agg_stats['num_max_words'] = df_articles_stats['num_words'].max()

    # num_words_unique
    dict_agg_stats['num_total_words_unique'] = df_articles_stats['num_words'].sum()
    dict_agg_stats['num_mean_words_unique'] = df_articles_stats['num_words'].mean()
    dict_agg_stats['num_min_words_unique'] = df_articles_stats['num_words'].min()
    dict_agg_stats['num_max_chars_unique'] = df_articles_stats['num_words'].max()

    # mean_lenght_word
    dict_agg_stats['mean_length_words'] = df_articles_stats['mean_lenght_word'].mean()

    # mean_lenght_word
    dict_agg_stats['lexical_density'] = dict_agg_stats['num_mean_words']/dict_agg_stats['num_mean_words_unique']
    
    # Number articles at least 280 characters
    filtro = (df_articles_stats.num_chars <= 280)
    dict_agg_stats['twitter_articles'] = df_articles_stats.loc[(filtro)].shape[0]
    
    # Number articles at least 280 characters
    filtro = (df_articles_stats.num_words < 100)
    dict_agg_stats['articles_lower'] = df_articles_stats.loc[(filtro)].shape[0]
    
    # token all text
    token_text = tprep.text_tokenize(' '.join(documents_all_text))
    
    # Unigram
    words_freq = pd.value_counts(token_text)
    words_freq = pd.DataFrame(words_freq,columns=['frequency'])
    words_freq.index.name = 'unigram'
    words_freq = words_freq.reset_index()
    dict_agg_stats['num_total_words_unique'] = len(list(set(words_freq.unigram.tolist())))
    
    # Bigram - this code can be in the prep/mining class on the text.py
    list_bigrams = list(nltk.bigrams(token_text))
    bigram_freq = pd.value_counts(list_bigrams)
    df_bigram = pd.DataFrame(bigram_freq, columns=['frequency'])
    df_bigram.index.name = 'bigram'
    df_bigram = df_bigram.reset_index()
    
    # Trigram - this code can be in the prep/mining class on the text.py
    list_trigam = list(nltk.trigrams(token_text))
    trigam_freq = pd.value_counts(list_trigam)
    df_trigram = pd.DataFrame(trigam_freq, columns=['frequency'])
    df_trigram.index.name = 'trigram'
    df_trigram = df_trigram.reset_index()
    
    with st.container():
        _ , col1, col2, col3, col4, _ = st.columns([0.5,2,2,2,3,0.5])
        with col1:
            col1.metric("🔠 Total Words", str(dict_agg_stats['num_total_words']))
            col1.metric("🔢 Mean Words per Article", str(round(dict_agg_stats['num_mean_words'],1)))
            col1.metric("🔣 Total Characters", str(dict_agg_stats['num_total_chars']))
        with col2:
            col2.metric("🆕 Total Unique Words", str(dict_agg_stats['num_total_words_unique']))
            col2.metric("🔢 Mean Unique Words per Article", str(round(dict_agg_stats['num_mean_words_unique'],1)))
            col2.metric("🔤 Mean Lenght Words", str(round(dict_agg_stats['mean_length_words'],1)))
        with col3:
            col3.metric("*️⃣ Mean Lexical Density (words/unique words)",str(round(dict_agg_stats['lexical_density'],1)))
            col3.metric("#️⃣ Twitter Articles", str(dict_agg_stats['twitter_articles']))
            col3.metric("💬 Articles words lower 1000", str(dict_agg_stats['articles_lower']))
        with col4:
            AgGrid(words_freq.head(100),
                   data_return_mode='AS_INPUT', 
                   # update_mode='MODEL_CHANGED', 
                   fit_columns_on_grid_load=False,
                   # theme='fresh',
                   enable_enterprise_modules=False,
                   height=250, 
                   width='100%',
                   reload_data=True)
            
    with st.container():
        _ , col1, col2, _ = st.columns([0.75,2.75,2.75,1])
        with col1:
            AgGrid(df_bigram.head(50),
                   data_return_mode='AS_INPUT', 
                   # update_mode='MODEL_CHANGED', 
                   fit_columns_on_grid_load=False,
                   # theme='fresh',
                   enable_enterprise_modules=False,
                   height=250, 
                   width='100%',
                   reload_data=True)
        with col2:
            AgGrid(df_trigram.head(50),
                   data_return_mode='AS_INPUT', 
                   # update_mode='MODEL_CHANGED', 
                   fit_columns_on_grid_load=False,
                   # theme='fresh',
                   enable_enterprise_modules=False,
                   height=250, 
                   width='100%',
                   reload_data=True)


def show_articles_data(st, dict_dfs):
    
    """"""
    
    st.write("**[Articles Data] df_doc_info (with 5 rows sample):**")
    st.dataframe(dict_dfs['df_doc_info'].head(5).astype(str), width=None, height=None)
    
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
        st.markdown("""<h3 style="text-align:left;"><b>Similarity Graph: this graph shows you similarity across articles.</b></h3>""", unsafe_allow_html=True)
        
        # path_comp_sim = os.path.join(path,'components','collapse_similarity.html')
        # print(path_comp_sim)
        # components.html(get_component_from_file(path_comp_sim),height=None, width=None, scrolling=False)
        
        with st.expander("How it works?"):
            st.write("This is MAGIC!")
            
        col1, col2 = st.columns([1,2])
        with col1:
            df_cos_tfidf_sim_filter['value'] = df_cos_tfidf_sim_filter['value'].apply(lambda e: round(100*e,2))
            df_cos_tfidf_sim_filter['doc_a'] = df_cos_tfidf_sim_filter['doc_a'].apply(lambda e: e[0:4])
            df_cos_tfidf_sim_filter['doc_b'] = df_cos_tfidf_sim_filter['doc_b'].apply(lambda e: e[0:4])
            AgGrid(df_cos_tfidf_sim_filter.head(50),
                   data_return_mode='AS_INPUT', 
                   # update_mode='MODEL_CHANGED', 
                   fit_columns_on_grid_load=False,
                   # theme='fresh',
                   enable_enterprise_modules=False,
                   height=510, 
                   width='100%',
                   reload_data=True)
        with col2:
            show_graph_graph(sim_graph, path_graph, path_folder_graph)

def get_component_from_file(path_html):
    
    """"""

    GraphHtmlFile = open(path_html, 'r', encoding='utf-8')
    GraphHtml = GraphHtmlFile.read()
    GraphHtmlFile.close()
    
    return GraphHtml
    

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
        GraphHtmlFile.close()
        
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


def text_preparation(st, dict_dfs, input_folder_path):
    
    """"""
    
    tprep = text_prep()
    
    # dict_dfs['df_doc_info']['acknowledgement_prep'] = tprep.text_prep_column(dict_dfs['df_doc_info']['acknowledgement'])
    dict_dfs['df_doc_info']['abstract_prep'] = tprep.text_preparation_column(dict_dfs['df_doc_info']['abstract'])
    dict_dfs['df_doc_info']['body_prep'] = tprep.text_preparation_column(dict_dfs['df_doc_info']['body'])
    
    return dict_dfs


def show_keywords(st, dict_dfs):
    
    """"""
    
    kw_model = KeyBERT()
    
    dict_keywords = {}
    col_select = ['pdf_md5','abstract']
    docs = dict_dfs['df_doc_info'].reset_index().loc[:, col_select]

    list_keywordsdf = []
    for i, row in docs.iterrows():
        
        doc = str(row['abstract'])
        id = row['pdf_md5']
        
        keywords_unigram = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words='english', highlight=False, top_n=10)
        if len(keywords_unigram):
            df_unigram = pd.DataFrame([{'keyword':v[0],'value':v[1]} for v in keywords_unigram])
        else:
            df_unigram = pd.DataFrame([], columns=['keyword','value'])

        keywords_bigram = kw_model.extract_keywords(doc, keyphrase_ngram_range=(2, 2), stop_words='english', highlight=False, top_n=10)
        if len(keywords_bigram):
            df_bigram = pd.DataFrame([{'keyword':v[0],'value':v[1]} for v in keywords_bigram])
        else:
            df_bigram = pd.DataFrame([], columns=['keyword','value'])

        keywords_trigam = kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 3), stop_words='english', highlight=False, top_n=10)
        if len(keywords_bigram):
            df_trigram = pd.DataFrame([{'keyword':v[0],'value':v[1]} for v in keywords_trigam])
        else:
            df_trigram = pd.DataFrame([], columns=['keyword','value'])
        
        dict_keywords[id] = {'unigram':df_unigram, 'bigram':df_bigram, 'trigram':df_trigram}
        
        df_unigram.rename(columns={'keyword':'keyword_unigram','value':'value_unigram'}, inplace=True)
        df_bigram.rename(columns={'keyword':'keyword_bigram','value':'value_bigram'}, inplace=True)
        df_trigram.rename(columns={'keyword':'keyword_trigram','value':'value_trigram'}, inplace=True)
        
        df_keywords_article = pd.concat([df_unigram, df_bigram, df_trigram], axis=1)
        dict_keywords[id]['df_keywords'] = df_keywords_article
        
        list_keywordsdf.append(df_keywords_article)
        
    df_keywords_all = pd.concat(list_keywordsdf)
    df_keywords_all.dropna(inplace=True)

    df_keywords_unigram = df_keywords_all.groupby(by=['keyword_unigram'], as_index=False)['value_unigram'].sum()
    df_keywords_unigram.sort_values(by='value_unigram', ascending=False, inplace=True)

    df_keywords_bigram = df_keywords_all.groupby(by=['keyword_bigram'], as_index=False)['value_bigram'].sum()
    df_keywords_bigram.sort_values(by='value_bigram', ascending=False, inplace=True)

    df_keywords_trigram = df_keywords_all.groupby(by=['keyword_trigram'], as_index=False)['value_trigram'].sum()
    df_keywords_trigram.sort_values(by='value_trigram', ascending=False, inplace=True)

    df_keywords_all = pd.concat([df_keywords_unigram, df_keywords_bigram, df_keywords_trigram], axis=1)
    df_keywords_all = df_keywords_all.head(200)
    
    f = lambda e: round(e,2) if not pd.isna(e) else e
    df_keywords_all['value_unigram'] = df_keywords_all['value_unigram'].apply(f)
    df_keywords_all['value_bigram'] = df_keywords_all['value_bigram'].apply(f)
    df_keywords_all['value_trigram'] = df_keywords_all['value_trigram'].apply(f)
    
    with st.container():
        _, col, _ = st.columns([0.1,8,0.1])
        with col:
            AgGrid(df_keywords_all.head(50),
                data_return_mode='AS_INPUT', 
                # update_mode='MODEL_CHANGED', 
                fit_columns_on_grid_load=False,
                # theme='fresh',
                enable_enterprise_modules=False,
                height=510, 
                width='100%',
                reload_data=True)
   

def btn_clicked_folder(st, input_folder_path, n_workers, show_wordcloud=True, show_similaritygraph=True,
                       cache_folder_name='summarticles_cache', show_text_macro=True, show_keywords_table=True):

    """"""
    
    # cache path files
    
    
    # Run and display batch process, return a dictionary of dataframes with all data extract from articles
    dict_dfs = run_batch_process(st, input_folder_path, n_workers=n_workers, cache_folder_name=cache_folder_name,
                                 display_articles_data=False, save_xmltei=True)
    
    if not dict_dfs:
        return None
    
    if not len(dict_dfs.keys()) or not dict_dfs['df_doc_info'].shape[0]:
        st.error("❓ There is no information to extract from articles in the specified path! Please, choose another fila path.")
    
    with st.container():
        with st.spinner('🔢📊 Generating articles macro numbers...'):
            st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
            show_macro_numbers(st, dict_dfs)
    
    with st.spinner('🛠️📄 Text prepatation...'):
        dict_dfs = text_preparation(st, dict_dfs, input_folder_path)
    
    # Container for wordcloud and text macro numbers
    with st.container():
        st.markdown("""<hr style="height:1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
        st.markdown("""<h3 style="text-align:left;"><b>Text Macro Numbers</b></h3>""",unsafe_allow_html=True)
    
        if show_wordcloud:
            with st.spinner('📄➞☁️ Making WordCloud...'):
                show_word_cloud(st, dict_dfs, input_folder_path, cache_folder_name=cache_folder_name,
                                folder_images='images', wc_image_name='wc_image.png')
        if show_text_macro:
            with st.spinner('🔢📊 Generating articles text numbers/stats...'):
                st.markdown("""<hr style="height:0.1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
                show_text_numbers(st, dict_dfs)

    if show_keywords_table:
        st.markdown("""<hr style="height:1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
        st.markdown("""<h3 style="text-align:left;"><b>KeyWords</b></h3>""",unsafe_allow_html=True)
        with st.spinner('📄➞🔤  Extracting KeyWords...'):
            show_keywords(st, dict_dfs)
    
    if show_similaritygraph:
        with st.spinner('📄➞📄  Making Similarity Graph...'):
            st.markdown("""<hr style="height:1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
            similarity_graph(st, dict_dfs, input_folder_path, percentil="75%", 
                             n_sim=100, cache_folder_name=cache_folder_name)


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
        with st.spinner('💻⚙️ Process running... Leave the rest to us! In the meantime maybe you can have some coffee. ☕'):
            btn_clicked_folder(st, input_folder_path, n_workers=10)
            