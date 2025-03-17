from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import numpy as np
import pandas as pd
import os
import re
import nltk

nltk.download('stopwords')

import plotly.express as px
from tkinter import filedialog

from datetime import datetime as dt

from summautils import *

from joblib import dump, load

import yake


def checkey(dic, key):
    """"""
    return True if key in dic else False


def get_component_from_file(path_html):
    
    """"""

    GraphHtmlFile = open(path_html, 'r', encoding='utf-8')
    GraphHtml = GraphHtmlFile.read()
    GraphHtmlFile.close()
    
    return GraphHtml


def text_statistics(text, tprep):
    
    """"""
    
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


def make_head(st):

    """"""

    # Head
    st.set_page_config(
        page_title="Summarticles",
        page_icon="📑", # https://www.freecodecamp.org/news/all-emojis-emoji-list-for-copy-and-paste/, https://share.streamlit.io/streamlit/emoji-shortcodes
        layout="wide", # centered wide
        initial_sidebar_state="collapsed", #collapsed #auto #expanded
        menu_items={"About":"https://github.com/Vieirbat/PGC",
                    "Get help":"https://github.com/Vieirbat/PGC",
                    "Report a bug":"https://github.com/Vieirbat/PGC"}) 
    
    st.markdown("""<h1 style="text-align:center;">🧾➜📊 Summarticles</h1>""",
                unsafe_allow_html=True)
    
    st.markdown("""<h3 style="text-align:center;"><b>Summarticles is an application to summarize articles information, 
                    using IA and analytics.</b></h3>""", 
                unsafe_allow_html=True) # st.write("Application")
    
    st.markdown("""<h6 style="text-align:center;">Do you want to use it? So, you only need to specify a folder path (with PDF article files) clicking 
                on '📁 Select a folder path!' and let the magic happen!</h6>""",
                unsafe_allow_html=True)


def show_macro_numbers(st, dict_dfs):
    
    st.markdown("""<h3 style="text-align:left;"><b>Articles Macro Numbers</b></h3>""", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🧾 Read Articles", str(dict_dfs['df_doc_info'].shape[0]))
    col2.metric("👥 Total Authors", str(dict_dfs['df_doc_authors'].shape[0]))
    col3.metric("📄➞📄 Total Citations",str(dict_dfs['df_doc_citations'].shape[0]))
    col4.metric("👥➞👥 Total Authors from Citations",str(dict_dfs['df_doc_authors_citations'].shape[0]))

    # st.plotly_chart(chars_graph(dict_dfs),use_container_width=True)


def show_text_numbers(st, dict_dfs, tprep):
    
    """"""
    print(dict_dfs.keys())
    if not checkey(dict_dfs,'show_text_numbers'):
    
        dict_dfs['show_text_numbers'] = {}
        
        if not checkey(dict_dfs['show_text_numbers'],'documents_abs'):
            documents_abs = dict_dfs['df_doc_info']['abstract_prep'].fillna(' ').tolist()
            dict_dfs['show_text_numbers']['documents_abs'] = documents_abs
        else:
            documents_abs = dict_dfs['show_text_numbers']['documents_abs']
        
        if not checkey(dict_dfs['show_text_numbers'],'documents_body'):
            documents_body = dict_dfs['df_doc_info']['body_prep'].fillna(' ').tolist()
            dict_dfs['show_text_numbers']['documents_body'] = documents_body
        else:
            documents_body = dict_dfs['show_text_numbers']['documents_body']
        
        if not checkey(dict_dfs['show_text_numbers'],'documents_all_text'):
            documents_all_text = [' '.join([abst, body]) for abst, body in zip(documents_abs, documents_body)]
            dict_dfs['show_text_numbers']['documents_all_text'] = documents_all_text
        else:
            documents_body = dict_dfs['show_text_numbers']['documents_body']
    
    print(dict_dfs.keys())
    print(dict_dfs['show_text_numbers']['documents_all_text'][0][0:50])
    # ------------------------------------------------------------------------
    # Making stats
    
    if not checkey(dict_dfs['show_text_numbers'],'dict_agg_stats'):
    
        df_articles_stats = pd.DataFrame(list(map(lambda e: text_statistics(e, tprep), documents_all_text)))
        df_articles_stats['file_name'] = [os.path.split(e)[-1] for e in dict_dfs['df_doc_info']['file'].tolist()]
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
        print('PALAVRAS ÚNICAS',df_articles_stats['num_words_unique'])
        print('PALAVRAS ÚNICAS',df_articles_stats['num_words_unique'].sum())
        dict_agg_stats['num_total_words_unique'] = df_articles_stats['num_words_unique'].sum()
        dict_agg_stats['num_mean_words_unique'] = df_articles_stats['num_words_unique'].mean()
        dict_agg_stats['num_min_words_unique'] = df_articles_stats['num_words_unique'].min()
        dict_agg_stats['num_max_chars_unique'] = df_articles_stats['num_words_unique'].max()

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
        
        dict_dfs['show_text_numbers']['dict_agg_stats'] = dict_agg_stats
        dict_dfs['show_text_numbers']['df_articles_stats'] = df_articles_stats
        
    else:
        dict_agg_stats = dict_dfs['show_text_numbers']['dict_agg_stats']
        df_articles_stats = dict_dfs['show_text_numbers']['df_articles_stats']
    
    
    # token all text
    if not checkey(dict_dfs['show_text_numbers'],'token_text'):
        token_text = tprep.text_tokenize(' '.join(documents_all_text))
        dict_dfs['show_text_numbers']['token_text'] = token_text
    else:
        token_text = dict_dfs['show_text_numbers']['token_text']
    
    # regex
    def f_reg(t):
        texto = re.sub(r'\W+',' ',str(t))
        texto = re.sub(r'\s+', ' ', texto)
        texto = texto.strip()
        return texto
    
    # Unigram
    if not checkey(dict_dfs['show_text_numbers'],'words_freq'):
        words_freq = pd.Series(token_text)
        words_freq = words_freq.value_counts()
        words_freq = pd.DataFrame(words_freq).rename(columns={'count':'frequency'})
        words_freq.index.name = 'unigram'
        words_freq = words_freq.reset_index()
        dict_agg_stats['num_total_words_unique'] = len(list(set(words_freq.unigram.tolist())))
        words_freq.unigram = words_freq.unigram.apply(f_reg)
        dict_dfs['show_text_numbers']['words_freq'] = words_freq
    else:
        words_freq = dict_dfs['show_text_numbers']['words_freq']
    
    # Bigram - this code can be in the prep/mining class on the text.py
    if not checkey(dict_dfs['show_text_numbers'],'df_bigram'):
        list_bigrams = list(nltk.bigrams(token_text))
        bigram_freq = pd.Series(list_bigrams).value_counts()
        df_bigram = pd.DataFrame(bigram_freq).rename(columns={'count':'frequency'})
        df_bigram.index.name = 'bigram'
        df_bigram = df_bigram.reset_index()
        df_bigram.bigram = df_bigram.bigram.apply(f_reg)
        dict_dfs['show_text_numbers']['df_bigram'] = df_bigram
    else:
        df_bigram = dict_dfs['show_text_numbers']['df_bigram']
    
    # Trigram - this code can be in the prep/mining class on the text.py
    if not checkey(dict_dfs['show_text_numbers'],'df_trigram'):
        list_trigam = list(nltk.trigrams(token_text))
        trigam_freq = pd.Series(list_trigam).value_counts()
        df_trigram = pd.DataFrame(trigam_freq).rename(columns={'count':'frequency'})
        df_trigram.index.name = 'trigram'
        df_trigram = df_trigram.reset_index()
        df_trigram.trigram = df_trigram.trigram.apply(f_reg)
        dict_dfs['show_text_numbers']['df_trigram'] = df_trigram
    else:
        df_trigram = dict_dfs['show_text_numbers']['df_trigram']
    
    with st.container():
        _ , col1, col2, col3, _ = st.columns([1.5, 3, 3, 3, 0.25])
        with col1:
            col1.metric("🔠 Total Words", str(dict_agg_stats['num_total_words']))
            col1.metric("🔢 Mean Words per Article", str(round(dict_agg_stats['num_mean_words'],1)))
        with col2:
            print('PRINT AQUI: ',str(dict_agg_stats['num_total_words_unique']))
            col2.metric("🆕 Total Unique Words", str(dict_agg_stats['num_total_words_unique']))
            col2.metric("🔢 Mean Unique Words per Article", str(round(dict_agg_stats['num_mean_words_unique'],1)))
        with col3:
            col3.metric("🔣 Total Characters", str(dict_agg_stats['num_total_chars']))
            col3.metric("🔤 Mean Lenght Words", str(round(dict_agg_stats['mean_length_words'],1)))
            
        with st.expander(" ❕ Information!"):
            body = """Above:<br>These numbers represent informations from all article text together.<br><br>
                      Below:<br>This chart represent the number of unique words by article cross number of words by article.<br>
                      Each point represent an article, you can pass cursor over these points and get more information."""
            st.markdown(body, unsafe_allow_html=True)
        
        with st.container():
            fig = px.scatter(df_articles_stats, 
                            x="num_words_unique", 
                            y="num_words", 
                            size="mean_lenght_word", 
                            color="num_chars",
                            custom_data=['file_name','num_chars','mean_lenght_word'],
                            labels={"num_words_unique":"Number of Unique Words",
                                    "num_words":"Number of Words",
                                    "num_chars":"Number of<br>Characters"})
            
            labels = ["Article File Name: %{customdata[0]}",
                      "Number Words: %{y}",
                      "Number Unique Words: %{x}",
                      "Total Number of Characters: %{customdata[1]}",
                      "Mean Leangth of Words: %{customdata[2]}"]
                        
            fig.update_traces(hovertemplate="<br>".join(labels))
            
            st.plotly_chart(fig, use_container_width=True)
            
            
    with st.container():
        _ , col1, col2, col3, _ = st.columns([0.15,2.25,2.5,2.75,0.15])
        with col1:
            AgGrid(words_freq.head(100),
                   data_return_mode='AS_INPUT', 
                   # update_mode='MODEL_CHANGED', 
                   fit_columns_on_grid_load=False,
                   # theme='fresh',
                   enable_enterprise_modules=False,
                   height=250, 
                   width='100%',
                   reload_data=True)
        with col2:
            AgGrid(df_bigram.head(50),
                   data_return_mode='AS_INPUT', 
                   # update_mode='MODEL_CHANGED', 
                   fit_columns_on_grid_load=False,
                   # theme='fresh',
                   enable_enterprise_modules=False,
                   height=250, 
                   width='100%',
                   reload_data=True)
        with col3:
            AgGrid(df_trigram.head(50),
                   data_return_mode='AS_INPUT', 
                   # update_mode='MODEL_CHANGED', 
                   fit_columns_on_grid_load=False,
                   # theme='fresh',
                   enable_enterprise_modules=False,
                   height=250, 
                   width='100%',
                   reload_data=True)
            
        with st.expander(" ❕ Information!"):
            body = """Unigram: only one frequent word that happen with high frequency over the all articles text<br>
                      Bigram: two words that happen with high frequency together over the all articles text<br>
                      Trigram: three words that happen with high frequency together over the all articles text<br>"""
            st.markdown(body, unsafe_allow_html=True)
            
    return dict_dfs


def make_sidebar(st):
    
    """"""

    with st.sidebar:
        st.markdown("""Menu Sidebar""", unsafe_allow_html=False)
        
        clear_all = st.button("🛑 Clear all memory and execution!", key="clear_all")
        if clear_all:
            st.session_state = {}
            st.cache_data.clear()
            st.rerun()


def make_run_button(st):
    
    """"""
    
    _, btn_col1, _ = st.columns([3,3,1])
    
    with btn_col1:
        btn_getfolder = st.button('⚡ Run process!', key='btn_run_app')
        st.markdown("""<p style="text-align:left;font-size:11px;">Only enabled if input path is correctly!</p>""", unsafe_allow_html=True)
    
    return btn_getfolder


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
    
    return dict_dfs


def show_keywords(st, dict_dfs):
    
    """"""
    
    # with st.container():
    #     _, col, _ = st.columns([0.1,8,0.1])
    #     with col:
    #         AgGrid(dict_dfs['keywords']['df_keywords_all'].head(50),
    #             data_return_mode='AS_INPUT', 
    #             # update_mode='MODEL_CHANGED', 
    #             fit_columns_on_grid_load=False,
    #             # theme='fresh',
    #             enable_enterprise_modules=False,
    #             height=510, 
    #             width='100%',
    #             reload_data=True)
            
    with st.container():
        _ , col1, col2, col3, _ = st.columns([0.01,3,3,3,0.01])
        with col1:
            df_show = dict_dfs['keywords']['df_unigram']
            df_show = df_show.rename(columns={"keyword_unigram":"Keyword Unigram",
                                              "value_unigram":"Relevance"})
            df_show['Relevance'] = df_show['Relevance'].apply(lambda x: np.round(float(x),3) if not pd.isna(x) else x)
            df_show = df_show.sort_values(by=['Relevance'], ascending=True)
            
            AgGrid(df_show.head(100),
                   data_return_mode='AS_INPUT', 
                   # update_mode='MODEL_CHANGED', 
                   fit_columns_on_grid_load=False,
                   # theme='fresh',
                   enable_enterprise_modules=False,
                   height=250, 
                   width='100%',
                   reload_data=True)
        with col2:
            df_show = dict_dfs['keywords']['df_bigram']
            df_show = df_show.rename(columns={"keyword_bigram":"Keyword Bigram",
                                              "value_bigram":"Relevance"})
            df_show['Relevance'] = df_show['Relevance'].apply(lambda x: np.round(float(x),3) if not pd.isna(x) else x)
            df_show = df_show.sort_values(by=['Relevance'], ascending=True)
            
            AgGrid(df_show.head(100),
                   data_return_mode='AS_INPUT', 
                   # update_mode='MODEL_CHANGED', 
                   fit_columns_on_grid_load=False,
                   # theme='fresh',
                   enable_enterprise_modules=False,
                   height=250, 
                   width='100%',
                   reload_data=True)
        with col3:
            df_show = dict_dfs['keywords']['df_trigram']
            df_show = df_show.rename(columns={"keyword_trigram":"Keyword Trigram",
                                              "value_trigram":"Relevance"})
            df_show['Relevance'] = df_show['Relevance'].apply(lambda x: np.round(float(x),3) if not pd.isna(x) else x)
            df_show = df_show.sort_values(by=['Relevance'], ascending=True)
            
            AgGrid(df_show.head(100),
                   data_return_mode='AS_INPUT', 
                   # update_mode='MODEL_CHANGED', 
                   fit_columns_on_grid_load=False,
                   # theme='fresh',
                   enable_enterprise_modules=False,
                   height=250, 
                   width='100%',
                   reload_data=True)
            
            
def choose_filepath(st, tk_root):
    
    """"""
    
    with st.spinner(' 📁 Choosing a file path...'):
        # Get articles files path
        st.warning("⚠️ Feature in development!")
        input_folder_path = input_path = ""
        input_folder_path = st.text_input('Selected folder:',filedialog.askdirectory(master=tk_root), key='txt_input_path')
        
    return input_folder_path


def table_author_contrib(st, dict_dfs):
    """"""

    from itertools import permutations, combinations

    if "authors_contrib" not in dict_dfs:
        
        dict_dfs["authors_contrib"] = {}
    
        df_articles = dict_dfs['df_doc_authors'].reset_index()
        list_colab = []
        for article_id in df_articles.article_id.unique():
            df_aux = df_articles.loc[df_articles.article_id==article_id].copy()
            list_authors = df_aux.full_name_author.to_list()
            
            list_permut = []
            if len(list_authors) >= 2:
                for i in list(range(2,3,1)):
                    list_permut += list(permutations(list_authors,i))
            
            list_colab += list_permut

        colab = pd.value_counts(list_colab)

        list_colab = []
        for i, tup in enumerate(colab.index):
            dictColab = {}
            source, target = tup
            dictColab['source'] = source
            dictColab['target'] = target
            dictColab['value'] = colab[i]
            list_colab.append(dictColab)

        df_colab = pd.DataFrame(list_colab)
        df_colab = df_colab.loc[(df_colab.source!=df_colab.target)].copy()
        df_colab = df_colab.sort_values(by=['value'], ascending=False)

        filter_comb = ~df_colab[['source', 'target']].apply(frozenset, axis=1).duplicated()
        df_colab = df_colab.loc[filter_comb]
        
        dict_dfs["authors_contrib"]["df_colab"] = df_colab

        # matrix_colab = df_colab.pivot_table(columns=['target'], index=['source'], values=["value"])
        # matrix_colab = matrix_colab.fillna(0)
    
    AgGrid(dict_dfs["authors_contrib"]["df_colab"].head(30),
            data_return_mode='AS_INPUT', 
            fit_columns_on_grid_load=False,
            enable_enterprise_modules=False,
            height=400, 
            width='100%',
            reload_data=True)

    return dict_dfs


def input_path_success_message(st, input_folder_path):
    files_count = len(files_path(input_folder_path))
    st.success(f"✔️ **In this folder path we found: {str(files_count)} files!** Folder path: {str(input_folder_path)}")
    return files_count
    
def generate_keywords(st, dict_dfs, num_keywords=20):
    
    """Keywords tables."""
    
    if 'keywords' not in dict_dfs:
        
        dict_dfs['keywords'] = {}
        
        # kw_model = KeyBERT()
        
        dict_keywords = {}
        id_column = 'article_id'
        text_column = 'abstract'
        col_select = [id_column,text_column]
        docs = dict_dfs['df_doc_info'].reset_index().loc[:, col_select]

        list_keywordsdf = []
        list_keywordsdf_article = []
        for i, row in docs.iterrows():
            
            doc = str(row[text_column])
            id = row[id_column]
            
            # keywords_unigram = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words='english', highlight=False, top_n=10)
            kw_extractor  = yake.KeywordExtractor(lan='en', n=1, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=num_keywords, features=None)
            keywords_unigram = kw_extractor.extract_keywords(doc)
            if len(keywords_unigram):
                df_unigram = pd.DataFrame([{'keyword':v[0],'value':v[1]} for v in keywords_unigram])
            else:
                df_unigram = pd.DataFrame([], columns=['keyword','value'])

            # keywords_bigram = kw_model.extract_keywords(doc, keyphrase_ngram_range=(2, 2), stop_words='english', highlight=False, top_n=10)
            kw_extractor  = yake.KeywordExtractor(lan='en', n=2, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=num_keywords, features=None)
            keywords_bigram = kw_extractor.extract_keywords(doc)
            if len(keywords_bigram):
                df_bigram = pd.DataFrame([{'keyword':v[0],'value':v[1]} for v in keywords_bigram])
            else:
                df_bigram = pd.DataFrame([], columns=['keyword','value'])

            # keywords_trigam = kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 3), stop_words='english', highlight=False, top_n=10)
            kw_extractor  = yake.KeywordExtractor(lan='en', n=3, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=num_keywords, features=None)
            keywords_trigam = kw_extractor.extract_keywords(doc)
            if len(keywords_trigam):
                df_trigram = pd.DataFrame([{'keyword':v[0],'value':v[1]} for v in keywords_trigam])
            else:
                df_trigram = pd.DataFrame([], columns=['keyword','value'])
            
            dict_keywords[id] = {'unigram':df_unigram, 'bigram':df_bigram, 'trigram':df_trigram}
            
            df_article_keywords = pd.concat([df_unigram, df_bigram, df_trigram])
            df_article_keywords[id_column] = id
            df_article_keywords = df_article_keywords.loc[:,[id_column,'keyword', 'value']].copy()
            list_keywordsdf_article.append(df_article_keywords)
            
            df_unigram.rename(columns={'keyword':'keyword_unigram','value':'value_unigram'}, inplace=True)
            df_bigram.rename(columns={'keyword':'keyword_bigram','value':'value_bigram'}, inplace=True)
            df_trigram.rename(columns={'keyword':'keyword_trigram','value':'value_trigram'}, inplace=True)
            
            df_keywords_article = pd.concat([df_unigram, df_bigram, df_trigram], axis=1)
            dict_keywords[id]['df_keywords'] = df_keywords_article
            
            list_keywordsdf.append(df_keywords_article)
        
        dict_dfs['keywords']['list_keywordsdf'] = list_keywordsdf
        dict_dfs['keywords']['list_keywordsdf_article'] = list_keywordsdf_article
        dict_dfs['keywords']['df_unigram'] = df_unigram
        dict_dfs['keywords']['df_bigram'] = df_bigram
        dict_dfs['keywords']['df_trigram'] = df_trigram
        
        
    df_keywords_all = pd.concat(dict_dfs['keywords']['list_keywordsdf'])
    df_keywords_all.dropna(inplace=True)

    df_article_keywords_all = pd.concat(dict_dfs['keywords']['list_keywordsdf_article'])
    df_article_keywords_all.dropna(inplace=True)
    dict_dfs['keywords']['df_article_keywords_all'] = df_article_keywords_all

    df_keywords_unigram = df_keywords_all.groupby(by=['keyword_unigram'], as_index=False)['value_unigram'].sum()
    df_keywords_unigram.sort_values(by='value_unigram', ascending=True, inplace=True)

    df_keywords_bigram = df_keywords_all.groupby(by=['keyword_bigram'], as_index=False)['value_bigram'].sum()
    df_keywords_bigram.sort_values(by='value_bigram', ascending=True, inplace=True)

    df_keywords_trigram = df_keywords_all.groupby(by=['keyword_trigram'], as_index=False)['value_trigram'].sum()
    df_keywords_trigram.sort_values(by='value_trigram', ascending=True, inplace=True)

    df_keywords_all = pd.concat([df_keywords_unigram, df_keywords_bigram, df_keywords_trigram], axis=1)
    df_keywords_all = df_keywords_all.head(200)
    
    f = lambda e: np.round(e,3) if not pd.isna(e) else e
    df_keywords_all['value_unigram'] = df_keywords_all['value_unigram'].apply(f)
    df_keywords_all['value_bigram'] = df_keywords_all['value_bigram'].apply(f)
    df_keywords_all['value_trigram'] = df_keywords_all['value_trigram'].apply(f)
    dict_dfs['keywords']['df_keywords_all'] = df_keywords_all
    
    return dict_dfs


def load_previous_execution(file, inputpath, cache_folder_name='summarticles_cache', folder_execs='summa_files', ext_file='summa'):
    """"""
    dict_dfs = None
    file_path = os.path.join(inputpath,cache_folder_name,folder_execs,file)
    if os.path.exists(file_path):
        dict_dfs = load(file_path)
    return dict_dfs


def show_last_executions(st, list_files_execs, inputpath, cache_folder_name='summarticles_cache', 
                         folder_execs='summa_files', ext_file='summa'):
    """"""
    
    choice_exec = None
    dict_dfs = None
    st.session_state['previous_exec_check'] = False
    
    opt_default = "Select an option"
    opt_new_run = "No I don't want to use previously execution, continue with new execution"
    default_opt = [opt_default, opt_new_run]
    list_files_execs_final = default_opt + list_files_execs
    
    if len(list_files_execs_final) > len(default_opt):
        
        st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
        st.info("❕ We found previously executions, do you want to recovery it?")
        
        choice_exec = st.selectbox("Please, choose one: ", list_files_execs_final, key="select_box_execs")
        # st.write('You selected:', choice_exec)
        if (choice_exec != opt_new_run) and (choice_exec != opt_default):
            try:
                dict_dfs = load_previous_execution(choice_exec, inputpath)
                st.session_state['previous_exec_check'] = True
                st.session_state['save_execution'] = False
                st.success("✅ Reload complete!")
            except Exception as error:
                st.error(f"❌ Error while reload previously execution: {str(error)}")
        elif choice_exec == opt_new_run:
            st.session_state['previous_exec_check'] = True
            st.session_state['save_execution'] = True
        else:
            st.session_state['previous_exec_check'] = False
            st.session_state['save_execution'] = True
            st.warning("⚠️ You need to choose an option!")
    else:
        st.session_state['previous_exec_check'] = True
        st.session_state['save_execution'] = True
        st.info("❕ There is no previously executions to recovery.")
        
    return dict_dfs


def check_input_path(st, path_input, gcli):

    # path_input = os.path.join(path,'artifacts','test_article')
    input_folder_path = get_path(path_input)
    
    print('[Path]:',input_folder_path)
    
    if not len(input_folder_path) or pd.isna(input_folder_path) or input_folder_path == "":
        st.error(f"❌ You need to specify a valid folder path, click on **'📁 Get folder path!'** Folder path: {str(input_folder_path)}!")
        return False
    elif not os.path.exists(input_folder_path):
        st.error(f"❌ The path doesn't exist! You need to specify a valid folder path! Folder path: {str(input_folder_path)}")
        return False
    elif not len(files_path(input_folder_path)):
        st.error(f"❌ There are no files in this folder path! You need to specify a valid folder path! Folder path: {str(input_folder_path)}")
        return False
    elif not gcli.check_typefile_inpath(files_path(input_folder_path)):
        st.error(f"❌ There are no files in this folder with APP required file type! Please make sure if that path is the correct path!")
        return False
    return True


def make_getpath_button(st):
    
    """"""
    
    _, btn_col1, _ = st.columns([3,3,1])
    
    with btn_col1:
        btn_getfolder = st.button('📁 Select a folder path!',key='btn_getfolder')
        st.markdown("""<p style="text-align:left;font-size:11px;">This application only works with PDF files.</p>""", unsafe_allow_html=True)
    
    return btn_getfolder


def make_reset_button(st):
    """"""
    _, btn_reset, _ = st.columns([3,3,1])
    with btn_reset:
        clear_all = st.button("🔄 Reset execution!", key="reset_exec")
        if clear_all:
            st.session_state = {}
            st.cache_data.clear()
            st.rerun()


def write_previous_execution(st, object, input_path, cache_folder_name='summarticles_cache', folder_execs='summa_files', file_name="report_summarticles", ext_file='summa'):
    """"""
    path = os.path.join(input_path, cache_folder_name, folder_execs)
    if not os.path.exists(path):
        rpath = os.mkdir(path)
    
    dtnow = dt.now().strftime("%Y%m%d%H%M%S")
    file_name_full = f"{file_name}_{dtnow}.{ext_file}" 
    file_path = os.path.join(path, file_name_full)
    rdump = dump(object, file_path, compress=0)
    
    st.success("✅ Save execution complete!")
    
    
def make_select_box(list_files_execs, label, id):
    
    values = """"""
    for valor in list_files_execs:
        values += f"""<option value="{valor}">{valor}</option>"""
    
    html = f"""<label for="cars">{label}</label>
               <select name="{id}" id="{id}">
                   {values}
               </select>"""
               
               
def get_last_executions(input_folder_path, cache_folder_name='summarticles_cache', folder_execs='summa_files', ext_file='summa'):
    
    path_execs = os.path.join(input_folder_path, cache_folder_name, folder_execs)
    list_files_execs = []
    if os.path.exists(path_execs):
        for f in os.listdir(path_execs):
            if str(f).endswith(ext_file):
                list_files_execs.append(f)
    return list_files_execs



