import os
import sys

sys.path.insert(0,os.path.dirname(os.getcwd()))
sys.path.insert(0,os.path.join(os.getcwd(),'grobid'))
sys.path.insert(0,os.getcwd())

from grobid import grobid_client
from grobid_to_dataframe import grobid_cli, xmltei_to_dataframe
from text import text_prep, text_mining, text_viz

import tkinter as tk

import streamlit as st

import networkx as nx
import matplotlib.pyplot as plt
from graph.pyvis.network import Network

from summautils import * 
from summaetl import *
from summacomponents import *
from summaviz import *
# from summachat import *

from st_aggrid import AgGrid


# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------

# docker run -t --rm --init -p 8080:8070 -p 8081:8071 --memory="9g" lfoppiano/grobid:0.7.0
# docker run -t --rm --init -p 8080:8070 -p 8081:8071 lfoppiano/grobid:0.6.2

path = os.path.dirname(os.getcwd())
global input_path

gcli = grobid_client.GrobidClient(config_path=os.path.join(path,"grobid/config.json"))



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



def run_batch_process(st, path_input, cache_folder_name='summarticles_cache', n_workers=10, display_articles_data=True, save_xmltei=True):

    """"""

    # path_input = os.path.join(path,'artifacts','test_article')
    input_folder_path = get_path(path_input)
        
    if len(files_path(input_folder_path)) < 2:
        st.error(f"❌ You need to specify a path with at least two pdf files!")
        return None
    
    with st.spinner('⚡ [Extract Text from Articles] Running batch process...'):
        
        result_batch = batch_process_path(input_folder_path, n_workers= n_workers)
        result_batch = clean_error_results(result_batch)
        
        if not len(result_batch):
            st.error(f"⚠️ Something is wrong, I can't get any result! 😕 Please, look if you selected the correct file path or if files in the selected path have information to extract!")
            return None
        
        dict_dfs, dict_errors = get_dataframes(result_batch, xmltei_to_dataframe())

        if save_xmltei:
            gcli.save_xmltei_files(result_batch, input_folder_path, cache_folder_name=cache_folder_name)

        if display_articles_data and len(result_batch):
            with st.spinner('🧾 Showing articles information...'):
                show_articles_data(st, dict_dfs)

    print('[Process has been finished!!!]')
    #msg.empty()
    #msg = customMsg('⚡ Finish process!','warning')
    
    return dict_dfs



def tk_configs():
    
    """"""
    
    # Set up tkinter
    tk_root = tk.Tk()
    tk_root.withdraw()
    
    # Make folder picker dialog appear on top of other windows
    tk_root.wm_attributes('-topmost', 1)
    
    return tk_root
    


def checkey(dic, key):
    """"""
    return True if key in dic else False

            

###############################################################################################
# ---------------------------------------------------------------------------------------------
# For process execution and streamlit app

if __name__ == '__main__':
    
    # ----------------------------------------------------------------------------
    # ATTENTION THIS CODE NEED SOME MODULARIZATION AND ORGANIZATION
    # BUT AT THIS TIME WE ONLY START APP DEVELOP
    
    # ----------------------------------------------------------------------------
    # State variables initialize
    
    if 'input_path' not in st.session_state:
        st.session_state['input_path'] = ""
    if 'path_check' not in st.session_state:
        st.session_state['path_check'] = False
    if 'previous_exec_check' not in st.session_state:
        st.session_state['previous_exec_check'] = False
    if 'choice_exec' not in st.session_state:
        st.session_state['choice_exec'] = "Select one option"
    if 'dict_dfs' not in st.session_state:
        st.session_state['dict_dfs'] = None
    if 'save_execution' not in st.session_state:
        st.session_state['save_execution'] = True
    if 'param_n_sim' not in st.session_state:
        st.session_state['param_n_sim'] = 100
    if 'param_sim' not in st.session_state:
        st.session_state['param_sim'] = 0.0
    if 'rb_reddim' not in st.session_state:
        st.session_state['rb_reddim'] = 'PCA'
    
    # For SummaChat
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Welcome to SummaChat, how can I help you? 🤖"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hi!"]
        
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
    # Reset execution
    if st.session_state['path_check']:
        make_reset_button(st)
        
    
    # ----------------------------------------------------------------------------
    # button getpath containers
    
    if not st.session_state['path_check']:
    
        btn_getfolder = make_getpath_button(st)
        
        # ----------------------------------------------------------------------------
        # Settings
        # This variables shall convert to checkbox 
        
        with st.expander("Settings and parameters"):
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                st.session_state['show_text_macro'] = show_text_macro = st.checkbox("Show Macro Text Information", True, key="chk_show_text_macro")
                st.session_state['show_wordcloud'] = show_wordcloud = st.checkbox("Show WordCloud Chart", True, key="chk_show_wordcloud")
            with c2:
                st.session_state['show_keywords_table'] = show_keywords_table = st.checkbox("Show KeyWords Information", True, key="chk_show_keywords_table")
                st.session_state['show_keywords_graph_cond'] = show_keywords_graph_cond = st.checkbox("Show KeyWords Chart", True, key="chk_show_keywords_graph_cond")
            with c3:
                st.session_state['show_similaritygraph'] = show_similaritygraph = st.checkbox("Show Similarity Graph", True, key="chk_show_similaritygraph")
                st.session_state['show_clustering'] = show_clustering = st.checkbox("Show Clustering Graph", True, key="chk_show_clustering")

        if btn_getfolder:
            
            input_folder_path = choose_filepath(st, tk_root)
            
            if check_input_path(st, input_folder_path, gcli):
                st.session_state['input_path'] = input_folder_path
                st.session_state['path_check'] = True
            else:
                st.session_state['path_check'] = False
                
        if st.session_state['path_check']:
            make_reset_button(st)
    
    # ----------------------------------------------------------------------------
    # Check if there are another executions in the cache
    
    if st.session_state['path_check'] and not st.session_state['previous_exec_check']:
        
        input_path_success_message(st, st.session_state['input_path'])
    
        with st.spinner('⚙️ Check if there are another executions in the cache!'):
            
            list_last_executions = get_last_executions(st.session_state['input_path'],
                                                       cache_folder_name='summarticles_cache',
                                                       folder_execs='summa_files',
                                                       ext_file='summa')
            dict_dfs = show_last_executions(st, list_last_executions, st.session_state['input_path'],
                                            cache_folder_name='summarticles_cache',
                                            folder_execs='summa_files', ext_file='summa')
                   
            st.session_state['dict_dfs'] = dict_dfs
        
    # ----------------------------------------------------------------------------
    # button getpath containers
    
    if st.session_state['previous_exec_check']:
        
        input_folder_path = st.session_state['input_path']
        with st.spinner('💻⚙️ Process running... Leave the rest to us! Meanwhile maybe you can have some coffee. ☕'):

            # Run and display batch process, return a dictionary of dataframes with all data extract from articles
            # if not option and not checkey(dict_dfs, 'df_doc_info'):
            
            if not st.session_state['dict_dfs']:
                st.session_state['dict_dfs'] = run_batch_process(st, input_folder_path, n_workers=10, cache_folder_name='summarticles_cache',
                                                display_articles_data=False, save_xmltei=True)
            
            if not st.session_state['dict_dfs']:
                st.error("❓ There is no information to extract from articles in the specified path! Please, choose another file path.")
            else:
                if not len(st.session_state['dict_dfs'].keys()) or not st.session_state['dict_dfs']['df_doc_info'].shape[0]:
                    st.error("❓ There is no information to extract from articles in the specified path! Please, choose another file path.")
                else:
                    with st.container():
                        with st.spinner('🔢📊 Generating articles macro numbers...'):
                            st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
                            show_macro_numbers(st, st.session_state['dict_dfs'])
                        
                    with st.spinner('🛠️📄 Text prepatation...'):
                        st.session_state['dict_dfs'] = text_preparation(st, st.session_state['dict_dfs'], input_folder_path, text_prep())
                    
                    # Container for wordcloud and text macro numbers
                    with st.container():
                        
                        st.markdown("""<hr style="height:1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
                        st.markdown("""<h3 style="text-align:left;"><b>Text Macro Numbers</b></h3>""",unsafe_allow_html=True)
                    
                        if st.session_state['show_wordcloud']:
                            with st.spinner('📄➞☁️ Making WordCloud...'):
                                st.session_state['dict_dfs'] = show_word_cloud(st, st.session_state['dict_dfs'], text_viz(), input_folder_path, 
                                                                               cache_folder_name='summarticles_cache',
                                                                               folder_images='images', wc_image_name='wc_image.png')
                                
                        if st.session_state['show_text_macro']:
                            with st.spinner('🔢📊 Generating articles text numbers/stats...'):
                                st.markdown("""<hr style="height:0.1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
                                st.session_state['dict_dfs'] = show_text_numbers(st, st.session_state['dict_dfs'], text_prep())
                        
                        # with st.container():
                        #     with st.spinner('📄➞📄  Overview information...'):
                        #         st.markdown("""<hr style="height:1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
                        #         st.markdown("""<h3 style="text-align:left;"><b>Overview Information</b></h3>""", unsafe_allow_html=True)

                        #         st.session_state['dict_dfs'] = article_overview_information(st, st.session_state['dict_dfs'])
                                
                        with st.container():
                            with st.spinner('📄➞📄  Authors Information...'):
                                st.markdown("""<hr style="height:1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
                                st.markdown("""<h3 style="text-align:left;"><b>Authors Information</b></h3>""", unsafe_allow_html=True)
                                st.session_state['dict_dfs'] = article_authors_information(st, st.session_state['dict_dfs'])
                                st.session_state['dict_dfs'] = plot_maps(st, st.session_state['dict_dfs'], path)                                
                                c1, _, c2 = st.columns([0.5,0.08,0.42])
                                with c1:
                                    years_plot_article(st, st.session_state['dict_dfs'])
                                with c2:
                                    st.markdown("""<br><b>Authors Work Together Table</b>""", unsafe_allow_html=True)
                                    st.session_state['dict_dfs'] = table_author_contrib(st, st.session_state['dict_dfs'])
                                _, c1, _ = st.columns([0.1,0.6,0.2])
                                with c1:
                                    st.session_state['dict_dfs'] = plot_top_contrib(st, st.session_state['dict_dfs'])
                                
                        with st.container():
                            with st.spinner('📄➞📄  Citation Information...'):
                                st.markdown("""<hr style="height:1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
                                st.markdown("""<h3 style="text-align:left;"><b>Citation Information</b></h3>""", unsafe_allow_html=True)
                                st.session_state['dict_dfs'] = article_citation_information(st, st.session_state['dict_dfs'])

                    with st.container():
                        if  st.session_state['show_keywords_table']:
                            
                            st.markdown("""<hr style="height:1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
                            st.markdown("""<h3 style="text-align:left;"><b>KeyWords</b></h3>""",unsafe_allow_html=True)
                            
                            with st.expander("How it works?"):
                                st.write("""Below you can see all keywords extract from abstract text over all articles.
                                            The lower the score, the more relevant the keyword is.""")
                            
                            with st.spinner('📄➞🔤  Extracting KeyWords...'):
                                st.session_state['dict_dfs'] = generate_keywords(st, st.session_state['dict_dfs'])
                                
                            with st.spinner('📄➞🔤  Showing KeyWords...'):
                                show_keywords(st, st.session_state['dict_dfs'])
                                
                            with st.spinner('📄➞🔤  Showing KeyWords WordCloud...'):
                                st.session_state['dict_dfs'] = show_keyword_word_cloud(st, st.session_state['dict_dfs'], 
                                                                                       input_folder_path, text_viz(),
                                                                                       cache_folder_name='summarticles_cache',
                                                                                       folder_images='images', wc_image_name='wc_image_keyword.png')
                                
                            if st.session_state['show_keywords_graph_cond']:
                                with st.spinner('📄➞📄  Making KeyWord Graph...'):
                                    with st.expander(" ❕ How can I read this graph?"):
                                        body = """In this graph Summarticles plot Keywords (from Abstract) and Articles.<br>
                                                  The blue dots are articles, and you can pass cursor over this points and get more information.<br>
                                                  The colors boxes are keywords, and you can pass cursor over this boxes and get more information.<br><br>
                                                  The edges conecting blue dots with keyword boxes represent relevance of the keyword for the abstract article
                                                  and the edge thickness represent the level/value of keyword relevance.<br><br>
                                                  For now, you only can see the top 10 keywords.<br><br>
                                                  You can interact with graph, moving blue dots, boxes and edges.<br>
                                                  You also can zoom-in and zoom-out the graph."""
                                        st.markdown(body, unsafe_allow_html=True)
                                    st.session_state['dict_dfs'] = show_keywords_graph(st, st.session_state['dict_dfs'], 
                                                                                       st.session_state['dict_dfs']['keywords']['df_article_keywords_all'],
                                                                                       input_folder_path, 
                                                                                       text_mining())
                                    
                    with st.container():
                        if st.session_state['show_similaritygraph']:
                            with st.spinner('📄➞📄  Making Similarity Graph...'):
                                
                                st.markdown("""<hr style="height:1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
                                st.markdown("""<h3 style="text-align:left;"><b>Similarity Graph: this graph shows you similarity across articles.</b></h3>""", unsafe_allow_html=True)

                                with st.expander("How it works?"):
                                    st.write("""Using abstract text from all articles, Summarticles create a TF-IDF matrix and 
                                                compute cossine similarity cross all articles. 
                                                In Summarticles cossine similarity range from 0 to 1, closer than 1 means high similarity
                                                0 is the opposite.""")
                                    
                                def del_similarity_graph():
                                    del st.session_state['dict_dfs']['similarity_graph']
                                    
                                st.session_state['dict_dfs'], rel_size, sim_size = similarity_graph(st, st.session_state['dict_dfs'],
                                                                                                    input_folder_path, text_mining(),
                                                                                                    percentil="75%",
                                                                                                    n_sim=st.session_state['param_n_sim'],
                                                                                                    cache_folder_name='summarticles_cache')
                                    
                    with st.container():
                        if st.session_state['show_clustering']:
                            with st.spinner('📄➞📄  Making Clustering...'):
                                
                                st.markdown("""<hr style="height:1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
                                st.markdown("""<h3 style="text-align:left;"><b>Clustering Articles</b></h3>""", unsafe_allow_html=True)

                                with st.expander("How it works?"):
                                    st.write("""Using abstract text from all articles, Summarticles create a TF-IDF matrix and clustering the articles.
                                                For group the data, Summarticles use K-Means and the number of the groups is defined
                                                by committee vote using Silhouette-Score, Dabies-Bouldin Index and Calinsk-Harabaz Index.""")
                                
                                c1, c2 = st.columns([0.5,1])
                                
                                with st.container():
                                    with c2:
                                        with st.container():
                                            st.session_state['rb_reddim'] = st.radio("Select Data Projection Algorithm:",
                                                                                    ('PCA', 'UMAP', 'MDS', 'TSNE'),
                                                                                    horizontal=True,
                                                                                    help="Choose one of these algorithms for groups data projection!")
                                        st.session_state['dict_dfs'] = clustering_2d(st, st.session_state['dict_dfs'], 
                                                                                     text_mining(),
                                                                                     title_text="Group Articles 2D",
                                                                                     n_components=2,
                                                                                     algorithm=st.session_state['rb_reddim']) # UMAP, TSNE, PCA, MDS
                                    with c1:
                                        df_show = st.session_state['dict_dfs']['clustering_data']['cluster_data_table']
                                        df_show = df_show.rename(columns={"file_name":"File Name",
                                                                          "title_head":"Title",
                                                                          "label":"Group"})
                                        
                                        AgGrid(df_show,
                                               data_return_mode='AS_INPUT', 
                                               # update_mode='MODEL_CHANGED', 
                                               fit_columns_on_grid_load=False,
                                               # theme='fresh',
                                               enable_enterprise_modules=False,
                                               height=550, 
                                               width='100%',
                                               reload_data=True)
                                
                                with st.container():
                                    st.session_state['dict_dfs'] = clustering_3d(st, 
                                                                                 st.session_state['dict_dfs'],
                                                                                 text_mining(),
                                                                                 title_text="Group Articles 3D",
                                                                                 n_components=3,
                                                                                 algorithm=st.session_state['rb_reddim']) # UMAP, TSNE, PCA, MDS
                     
                    # with st.container():
                    #     with st.spinner('📄➞📄  Part-of-speech and Named Entities...'):
                            
                    #         st.markdown("""<hr style="height:1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
                    #         st.markdown("""<h3 style="text-align:left;"><b>Part-of-speech and Named Entities</b></h3>""", unsafe_allow_html=True)

                    #         part_of_speech(st, st.session_state['dict_dfs'])
                            
                    # SummaChat
                    # with st.container():
                        
                    #     llm = LlamaCpp(streaming=True,
                    #                    model_path="models/llama-2-7b-chat.Q2_K.gguf", # "llama-2-7b-chat.Q5_K_S.gguf", #"mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                    #                    temperature=0.75,
                    #                    top_p=1, 
                    #                    verbose=True,
                    #                    n_ctx=4096)
                        
                    #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
                    #     text_chunks = text_splitter.split_documents(text)

                    #     # Create embeddings
                    #     # model_id = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"  
                    #     # model_id = "sentence-transformers/all-MiniLM-L6-v2"
                    #     model_id = "sentence-transformers/paraphrase-MiniLM-L3-v2"
                    #     embeddings = HuggingFaceEmbeddings(model_name=model_id, 
                    #                                     model_kwargs={'device': 'cpu'})

                    #     # Create vector store
                    #     vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

                    #     # Create the chain object
                    #     chain = create_conversational_chain(vector_store, llm)
                        
                    #     display_chat_history(chain)
                    
                                
    if st.session_state['dict_dfs'] and st.session_state['save_execution']:
                             
        write_previous_execution(st, st.session_state['dict_dfs'], 
                                 st.session_state['input_path'],
                                 file_name="report_summarticles",
                                 ext_file='summa',
                                 cache_folder_name='summarticles_cache', 
                                 folder_execs='summa_files')
        
        st.session_state['save_execution'] = False
        