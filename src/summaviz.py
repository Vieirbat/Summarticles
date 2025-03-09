import numpy as np
import pandas as pd
import os
import plotly.express as px



import shutil

# import spacy
# from spacy_streamlit import visualize_ner

from st_aggrid import AgGrid

import holoviews as hv # '1.15.0'
from holoviews import opts, dim 

import streamlit.components.v1 as components

# hv.extension('bokeh') # '2.4.3'


from streamlit_folium import st_folium, folium_static
import folium
from folium.plugins import HeatMap

from summaetl import *


def checkey(dic, key):
    """"""
    return True if key in dic else False


def getColumnsWithData(df, return_percent=False, n_round=2):
    
    """"""
    
    list_col_with_data = []
    for col in df.columns.tolist():
        rows = df[col].shape
        n_null = df[col].isnull().sum()
        not_null_data_perc = (1-n_null/rows)
        if not_null_data_perc:
            if return_percent:
                list_col_with_data.append((col,np.round(not_null_data_perc, n_round)))
            list_col_with_data.append(col)
            
    return list_col_with_data   


def clustering_3d(st, dict_dfs, instance_tmining, title_text="Group Articles", n_components=3, algorithm='UMAP'):
    
    """"""
    
    # tmining = text_mining()
    tmining = instance_tmining
    
    if "clustering_data" not in dict_dfs:
        dict_dfs["clustering_data"] = {}
    
    if "clustering_data_3d" not in dict_dfs["clustering_data"]:
        dict_dfs["clustering_data"]["clustering_data_3d"] = {}
    
    if "documents_abs" not in dict_dfs["clustering_data"]["clustering_data_3d"]:
        dict_dfs["clustering_data"]["clustering_data_3d"]['documents_abs'] = dict_dfs['df_doc_info']['abstract_prep'].fillna(' ').tolist()
    # documents_body = dict_dfs['df_doc_info']['body_prep'].fillna(' ').tolist()
    
    if "df_tfidf_abstract_abs" not in dict_dfs["clustering_data"]["clustering_data_3d"]:
        dict_dfs["clustering_data"]["clustering_data_3d"]['df_tfidf_abstract_abs'] = tmining.get_df_tfidf(dict_dfs["clustering_data"]["clustering_data_3d"]['documents_abs'])
    # df_tfidf_abstract_body = tmining.get_df_tfidf(documents_body)
    
    # df_bow_abstract_abs = tmining.get_df_bow(documents_abs)
    # df_bow_abstract_body = tmining.get_df_bow(documents_body)
    if "cluster_label" not in dict_dfs["clustering_data"]:
        cluster_labels = tmining.make_clustering(dict_dfs["clustering_data_3d"]['df_tfidf_abstract_abs'].values,
                                                lim_sup=None, 
                                                init='k-means++', 
                                                n_init=10, 
                                                max_iter=30, 
                                                tol=1e-4, 
                                                random_state=0)
        dict_dfs["clustering_data"]["cluster_label"] = cluster_labels
    
    if "cluster_reduce_dim3" not in dict_dfs['clustering_data']:
        dictRedDim = tmining.reduce_dimensionality(dict_dfs['clustering_data']["clustering_data_3d"]['df_tfidf_abstract_abs'], 
                                                      y=dict_dfs['clustering_data']['cluster_label'],
                                                      n_components=n_components)
        dict_dfs['clustering_data']['cluster_reduce_dim3'] = dictRedDim

    dict_dfs['df_doc_info']['file_name'] = dict_dfs['df_doc_info']['file'].apply(lambda e: os.path.split(e)[-1])
        
    # Concatenate X and y arrays
    article_title = dict_dfs['df_doc_head']['title_head'].apply(lambda e: ''.join([str(e)[0:30],'...']) if len(str(e)) >= 30 else str(e)).values.reshape(dict_dfs['df_doc_info']['file'].shape[0],1)
    file_name = dict_dfs['df_doc_info']['file_name'].values.reshape(dict_dfs['df_doc_info']['file_name'].shape[0],1)

    arr_concat=np.concatenate((dict_dfs['clustering_data']['cluster_reduce_dim3'][algorithm],
                               dict_dfs["clustering_data"]["cluster_label"].reshape(dict_dfs["clustering_data"]["cluster_label"].shape[0],1),
                               file_name,
                               article_title), axis=1)

    # Create a Pandas dataframe using the above array
    df=pd.DataFrame(arr_concat, columns=['x', 'y', 'z', 'label', 'file_name', 'title_head'])
    
    dict_dfs['cluster_data_table'] = df.loc[:,['file_name', 'title_head','label']].copy()
    
    # Convert label data type from float to integer
    df['label'] = df['label'].astype(int)
    # Finally, sort the dataframe by label
    df.sort_values(by='label', axis=0, ascending=True, inplace=True)
    #--------------------------------------------------------------------------#
    # Create a 3D graph
    fig = px.scatter_3d(df, 
                        x='x',
                        y='y',
                        z='z',
                        color='label',
                        height=600,
                        width=750,
                        custom_data=['file_name','title_head','label','x','y','z'])

    # Update chart looks
    fig.update_layout(title_text=title_text,
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5))

    labels = ["Article File Name: %{customdata[0]}",
                "Article Title: %{customdata[1]}",
                "Grupo: %{customdata[2]}",
                "X: %{x}",
                "Y: %{y}",
                "Z: %{z}"]
                
    fig.update_traces(hovertemplate="<br>".join(labels))
    fig.update_coloraxes(showscale=False)

    # Update marker size
    # fig.update_traces(marker=dict(size=3, line=dict(color='black', width=0.1)))
    dict_dfs['clustering_data']["clustering_data_3d"]["fig"] = fig
        
    st.plotly_chart(dict_dfs['clustering_data']["clustering_data_3d"]["fig"], use_container_width=True)
    
    return dict_dfs

def show_word_cloud(st, dict_dfs, tviz, input_path, folder_images='app_images', wc_image_name='wc_image.png',
                    cache_folder_name='summarticles_cache'):
    
    """"""
    
    # tviz = text_viz()
    # tprep = text_prep()
    
    with st.expander(" ❕ Information!"):
        st.write("""This WordCloud contain information from all article text together! 
                    But, only uses the most frequent unigrams and bigrams.""")
    
    # st.warning("🛠🧾 Text in preparation for WordCloud!")
    if not checkey(dict_dfs,'show_word_cloud'):
        
        dict_dfs['show_word_cloud'] = {}
        
        # if "abstract_prep" not in dict_dfs['df_doc_info']:
        #     dict_dfs['df_doc_info']['abstract_prep'] = tprep.text_preparation_column(dict_dfs['df_doc_info']['abstract'])
        
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
        dict_dfs['show_word_cloud']['word_cloud_fig'] = fig
    
    st.pyplot(dict_dfs['show_word_cloud']['word_cloud_fig'])
    
    return dict_dfs


def show_keyword_word_cloud(st, 
                            dict_dfs,
                            input_path,
                            tviz,
                            folder_images='app_images',
                            wc_image_name='wc_image_keyword.png',
                            cache_folder_name='summarticles_cache'):
    
    """"""
    
    # tviz = text_viz()
    # tprep = text_prep()
    
    # st.warning("🛠🧾 Text in preparation for WordCloud!")
    if not 'show_keyword_word_cloud' in dict_dfs:
        
        dict_dfs['show_keyword_word_cloud'] = {}
        
        df_unigram = dict_dfs['keywords']['df_unigram']
        df_unigram = df_unigram.rename(columns={'keyword_unigram':'keyword',
                                                'value_unigram':'value'})
        df_bigram = dict_dfs['keywords']['df_bigram']
        df_bigram = df_bigram.rename(columns={'keyword_bigram':'keyword',
                                              'value_bigram':'value'})
        df_trigram = dict_dfs['keywords']['df_trigram'].rename(columns={'keyword_trigram':'keyword',
                                                                        'value_trigram':'value'})
        df_trigram = df_trigram.rename(columns={'keyword_trigram':'keyword',
                                                'value_trigram':'value'})
        
        df_keywords = pd.concat([df_unigram, df_bigram, df_trigram])
        
        dict_freq = {}
        for i, row in df_keywords.iterrows():
            dict_freq[row['keyword']] = 1/row['value']
        
        path_images = os.path.join(input_path, cache_folder_name, folder_images)
        if not os.path.exists(path_images):
            os.mkdir(path_images)

        wc, ax, fig = tviz.keyword_word_cloud(dict_freq, 
                                              path_image=os.path.join(path_images, wc_image_name), 
                                              show_wc=False, 
                                              width=1000, 
                                              height=300, 
                                              collocations=True, 
                                              background_color='white')
        
        # st.markdown("""<h5 style="text-align:left;"><b>WordCloud:</b></h5>""",unsafe_allow_html=True)
        dict_dfs['show_keyword_word_cloud']['word_cloud_fig'] = fig
    
    st.pyplot(dict_dfs['show_keyword_word_cloud']['word_cloud_fig'])
    
    return dict_dfs


def plot_top_contrib(st, dict_dfs):
    """"""
    
    if "authors_contrib_plot" not in dict_dfs['authors_contrib']:

        df_source = dict_dfs["authors_contrib"]["df_colab"].loc[:,['source','value']].rename(columns={'source':'name'})
        df_target = dict_dfs["authors_contrib"]["df_colab"].loc[:,['target','value']].rename(columns={'target':'name'})
        df_nodes = pd.concat([df_source, df_target])
        df_nodes = df_nodes.drop_duplicates()
        df_nodes = df_nodes.reset_index()
        
        sample = dict_dfs["authors_contrib"]["df_colab"].head(10)

        names = list(set(sample["source"].tolist() + sample["target"].tolist()))
        df_names = hv.Dataset(pd.DataFrame(names, columns=["name"]))

        nodes = hv.Dataset(df_nodes, 'index')

        chord = hv.Chord((sample, df_names))
        
        dict_dfs['authors_contrib']["authors_contrib_plot"] = chord
    
    
    dict_dfs['authors_contrib']["authors_contrib_plot"].opts(
        opts.Chord(height=700,
                    width=700,
                    title="Top Authors Work Together",
                    cmap='Category20',
                    edge_cmap='Category20',
                    edge_color="source", 
                    labels='name',
                    node_color="name",
                    node_size=20, 
                    edge_alpha=0.8))
    
    st.bokeh_chart(hv.render(dict_dfs['authors_contrib']["authors_contrib_plot"], backend='bokeh'))
        
    return dict_dfs


def article_authors_information(st, dict_dfs):
    
    """"""
    
    with st.expander(" ❕ Information!"):
        body = """This section contains authors articles information."""
        st.markdown(body, unsafe_allow_html=True)
    
    # DataSets and Preparation
    df_doc_info = dict_dfs['df_doc_info'].loc[:,getColumnsWithData(dict_dfs['df_doc_info'])]
    df_doc_head = dict_dfs['df_doc_head'].loc[:,getColumnsWithData(dict_dfs['df_doc_head'])]
    df_doc_authors = dict_dfs['df_doc_authors'].loc[:,getColumnsWithData(dict_dfs['df_doc_authors'])]
    df_doc_info_head = df_doc_info.join(df_doc_head, how='left')
    
    list_delete_authors = ['A R T I C L E I N F O', np.nan, 'Null', 'NaN','nan', 'null', '', ' ']
    filter_delete_authors = ~(df_doc_authors.full_name_author.isin(list_delete_authors))
    df_doc_authors = df_doc_authors.loc[filter_delete_authors].copy()
    
    # ---------------------------------------------------------------------------
    # Top Authors
    top_authors = df_doc_authors.full_name_author.value_counts()
    df_top_authors = pd.DataFrame({'Full Name':top_authors.index,
                                'Number of Articles':top_authors.values.tolist()})
    
    top_authors = df_top_authors.nlargest(20,'Number of Articles')
    top_authors = top_authors.sort_values('Number of Articles',ascending=True)
    fig_authors = px.bar(top_authors,
                         title='Top 20 Number of Articles by Authors',
                         y='Full Name',
                         x='Number of Articles',
                         color='Number of Articles',
                         width=400, 
                         height=600,
                         text='Number of Articles')
    fig_authors.update(layout_coloraxis_showscale=False)
    # fig_authors.update_traces(showlegend=False)
    # fig_authors.update_traces(marker_showscale=False)
    fig_authors.update_xaxes(visible=False)
    fig_authors.update_layout(yaxis_title=None, xaxis_title=True)

    # ---------------------------------------------------------------------------
    # Authors by Location
    
    columns_select = ['country_author','settlement_author', 'institution_author']
    df_sun_agg = df_doc_authors.groupby(by=columns_select, as_index=False, dropna=False)['full_name_author'].count()
    df_sun_agg = df_sun_agg.fillna("Null Value")
    df_sun_agg.rename(columns={'country_author':'Author Country',
                               'settlement_author':'Author Settlement',
                               'institution_author':'Author Institution',
                               'full_name_author':'Number of Authors'},
                      inplace=True)
    
    df_sun_agg['Percentage'] = (df_sun_agg['Number of Authors']/df_sun_agg['Number of Authors'].sum())
    df_sun_agg['Percentage'] = df_sun_agg['Percentage'].apply(lambda e: int(100*np.round(float(e),2)))
    
    fig_authors_loc = px.sunburst(df_sun_agg,
                                  title='Number of Authors by Location',
                                  width=550, 
                                  height=600,
                                  path=['Author Country',
                                        'Author Settlement',
                                        'Author Institution'],
                                  hover_data=['Percentage'],
                                  values='Number of Authors')
                                  # values='Percentage')

    col0 , _,col1 = st.columns([0.25,2,3])
    with col0:
        st.plotly_chart(fig_authors)
    with col1:
        st.plotly_chart(fig_authors_loc)
        
    return dict_dfs


def clustering_2d(st, dict_dfs, tmining, title_text="Group Articles", n_components=2, algorithm='UMAP'):
    
    """"""
    
    if "clustering_data" not in dict_dfs:
        dict_dfs["clustering_data"] = {}
    
    if "clustering_data_2d" not in dict_dfs["clustering_data"]:
        dict_dfs["clustering_data"]["clustering_data_2d"] = {}
    
    if "documents_abs" not in dict_dfs['clustering_data']["clustering_data_2d"]:
        dict_dfs['clustering_data']["clustering_data_2d"]['documents_abs'] = dict_dfs['df_doc_info']['abstract_prep'].fillna(' ').tolist()
    # documents_body = dict_dfs['df_doc_info']['body_prep'].fillna(' ').tolist()
    
    if "df_tfidf_abstract_abs" not in dict_dfs['clustering_data']["clustering_data_2d"]:
        dict_dfs['clustering_data']["clustering_data_2d"]['df_tfidf_abstract_abs'] = tmining.get_df_tfidf(dict_dfs['clustering_data']["clustering_data_2d"]['documents_abs'])
    # df_tfidf_abstract_body = tmining.get_df_tfidf(documents_body)
    
    # df_bow_abstract_abs = tmining.get_df_bow(documents_abs)
    # df_bow_abstract_body = tmining.get_df_bow(documents_body)
    
    if "cluster_label" not in dict_dfs["clustering_data"]:
        cluster_labels = tmining.make_clustering(dict_dfs['clustering_data']["clustering_data_2d"]['df_tfidf_abstract_abs'],
                                                lim_sup=None, 
                                                init='k-means++', 
                                                n_init=10, 
                                                max_iter=30, 
                                                tol=1e-4, 
                                                random_state=0)
        dict_dfs['clustering_data']['cluster_label'] = cluster_labels
    
    if "cluster_reduce_dim" not in dict_dfs['clustering_data']:
        dictRedDim, y = tmining.reduce_dimensionality(dict_dfs['clustering_data']["clustering_data_2d"]['df_tfidf_abstract_abs'], 
                                                      y=dict_dfs['clustering_data']['cluster_label'],
                                                      n_components=n_components
                                                      ), dict_dfs['clustering_data']['cluster_label']
        dict_dfs['clustering_data']['cluster_reduce_dim'] = dictRedDim

    dict_dfs['df_doc_info']['file_name'] = dict_dfs['df_doc_info']['file'].apply(lambda e: os.path.split(e)[-1])
        
    # Concatenate X and y arrays
    article_title = dict_dfs['df_doc_head']['title_head'].apply(lambda e: ''.join([str(e)[0:30],'...']) if len(str(e)) >= 30 else str(e)).values.reshape(dict_dfs['df_doc_info']['file'].shape[0],1)
    file_name = dict_dfs['df_doc_info']['file_name'].values.reshape(dict_dfs['df_doc_info']['file_name'].shape[0],1)

    arr_concat=np.concatenate((dict_dfs['clustering_data']['cluster_reduce_dim'][algorithm],
                               dict_dfs['clustering_data']['cluster_label'].reshape(dict_dfs['clustering_data']['cluster_label'].shape[0],1),
                               file_name,
                               article_title), axis=1)

    # Create a Pandas dataframe using the above array
    df=pd.DataFrame(arr_concat, columns=['x', 'y', 'label', 'file_name', 'title_head'])
    
    dict_dfs['clustering_data']['cluster_data_table'] = df.loc[:,['file_name', 'title_head','label']].copy()
    
    # Convert label data type from float to integer
    df['label'] = df['label'].astype(int)
    # Finally, sort the dataframe by label
    df.sort_values(by='label', axis=0, ascending=True, inplace=True)
    #--------------------------------------------------------------------------#

    # Create a 3D graph
    fig = px.scatter(df, 
                        x='x',
                        y='y',
                        color='label',
                        height=550,
                        width=750,
                        custom_data=['file_name','title_head','label','x','y'])

    # Update chart looks
    fig.update_layout(title_text=title_text,
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5))

    labels = ["Article File Name: %{customdata[0]}",
                "Article Title: %{customdata[1]}",
                "Group: %{customdata[2]}",
                "X: %{x}",
                "Y: %{y}"]
                
    fig.update_traces(hovertemplate="<br>".join(labels))
    fig.update_coloraxes(showscale=False)

    # Update marker size
    # fig.update_traces(marker=dict(size=3, line=dict(color='black', width=0.1)))
    dict_dfs['clustering_data']["clustering_data_2d"]["fig"] = fig
        
    st.plotly_chart(dict_dfs['clustering_data']["clustering_data_2d"]["fig"], use_container_width=True)
    
    return dict_dfs


def show_graph(st, graph, path_graph, path_folder_graph, text_spinner='👁‍🗨 Keyword Graph: drawing...'):
    
    """"""
    
    path = os.getcwd()
    path_graph_depend = os.path.join(path,'graph','pyvis','templates','dependencies')
    path_depend_dst = os.path.join(path_folder_graph,'dependencies')
    
    if not os.path.exists(path_depend_dst):
        os.mkdir(path_depend_dst)
    
    list_files = os.listdir(path_graph_depend)
    for file in list_files:
        f_source = os.path.join(path_graph_depend, file)
        f_destiny = os.path.join(path_depend_dst, file)
        shutil.copy(f_source, f_destiny)
        
    with st.spinner(text_spinner):
        
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



def similarity_graph(st, dict_dfs, input_folder_path, tmining, folder_graph='graphs', name_file="graph.html", cache_folder_name='summarticles_cache', 
                     column='abstract', n_sim=200, percentil="99%", sim_value_min=0, sim_value_max=0.99, buttons=False):
    
    """"""

    if  not ('similarity_graph' in dict_dfs):
        
        dict_dfs['similarity_graph'] = {}
    
        # tmining = text_mining()
        
        df_cos_tfidf_sim_filter = cossine_similarity_data(st, dict_dfs, tmining, column, n_sim, percentil,
                                                          sim_value_min, sim_value_max)
        
        print("Saindo da cossine_similarity_data: ", df_cos_tfidf_sim_filter.shape)
        
        dict_dfs['similarity_graph']['df_cos_tfidf_sim_filter'] = df_cos_tfidf_sim_filter
    
        df_nodes = nodes_data(st, dict_dfs, df_cos_tfidf_sim_filter)
        dict_dfs['similarity_graph']['df_nodes'] = df_nodes
        
        path_write_graph = os.path.join(input_folder_path, cache_folder_name)
        dict_dfs['similarity_graph']['path_write_graph'] = path_write_graph

        print('SIM SHAPE: ', df_cos_tfidf_sim_filter.shape)

        sim_graph, path_graph, path_folder_graph = tmining.make_sim_graph(matrix=df_cos_tfidf_sim_filter, node_data=df_nodes,
                                                    source_column="doc_a", to_column="doc_b", value_column="value",
                                                    height="500px", width="100%", directed=False, notebook=False,
                                                    bgcolor="#ffffff", font_color=False, layout=None, 
                                                    heading="Similarity Graph: this graph shows you similarity across articles.",
                                                    path_graph=path_write_graph, folder_graph=folder_graph, buttons=buttons,
                                                    name_file=name_file)
        
        dict_dfs['similarity_graph']['sim_graph'] = sim_graph
        dict_dfs['similarity_graph']['path_graph'] = path_graph
        dict_dfs['similarity_graph']['path_folder_graph'] = path_folder_graph
        
    with st.container():
        
        df_sim = None
        df_sim = dict_dfs['similarity_graph']['df_cos_tfidf_sim_filter']
        df_sim['value'] = df_sim['value'].apply(lambda e: round(e,2))
        df_sim['doc_a'] = df_sim['doc_a'].apply(lambda e: e[0:4])
        df_sim['doc_b'] = df_sim['doc_b'].apply(lambda e: e[0:4])

        df_nodes_info = dict_dfs['similarity_graph']['df_nodes']
        cols = ['article_id','title_head','file_name']
        df_nodes_info = df_nodes_info.loc[:,cols].copy()
        df_sim_all = df_sim.merge(df_nodes_info,
                                  how='left', 
                                  left_on='doc_a', 
                                  right_on='article_id')
        df_sim_all.drop(labels=['article_id'], axis=1, inplace=True)
        df_sim_all.rename(columns={"doc_a":"ID Article Source",
                                   "value":"Similarity",
                                   "doc_b":"ID Article Target",
                                   "title_head":"Article Title Source", 
                                   "file_name":"Article File Name Source"}, inplace=True)
        df_sim_all = df_sim_all.merge(df_nodes_info,
                                      how='left', 
                                      left_on='ID Article Target', 
                                      right_on='article_id')
        df_sim_all.drop(labels=['article_id'], axis=1, inplace=True)
        df_sim_all.rename(columns={"title_head":"Article Title Target", 
                                   "file_name":"Article File Name Target"}, inplace=True)
                
        df_sim = df_sim.rename(columns={"value":"Similarity",
                                        "doc_a":"Article Source",
                                        "doc_b":"Article Target"})
        
        AgGrid(df_sim_all,
               data_return_mode='AS_INPUT', 
               fit_columns_on_grid_load=False,
               enable_enterprise_modules=False,
               height=350, 
               width='100%',
               reload_data=True)
        
        show_graph(st, dict_dfs['similarity_graph']['sim_graph'],
                   dict_dfs['similarity_graph']['path_graph'],
                   dict_dfs['similarity_graph']['path_folder_graph'],
                   text_spinner='👁‍🗨 Similarity Graph: drawing...')
            
        relations_size = dict_dfs['similarity_graph']['df_cos_tfidf_sim_filter'].shape[0]
        sim_max_val = dict_dfs['similarity_graph']['df_cos_tfidf_sim_filter'].value.max()
        
        return dict_dfs, relations_size, sim_max_val
    
    
def article_citation_information(st, dict_dfs):
    
    df_doc_authors_citations = dict_dfs['df_doc_authors_citations'].loc[:,getColumnsWithData(dict_dfs['df_doc_authors_citations'])]
    df_doc_authors = dict_dfs['df_doc_authors'].loc[:,getColumnsWithData(dict_dfs['df_doc_authors'])]
    
    with st.expander(" ❕ Information!"):
        body = """This section contains information about all of citations from load articles."""
        st.markdown(body, unsafe_allow_html=True)
    
    def agg_citations(grupo):
        dictR = {}
        dictR['citation_name'] = grupo['full_name_citation'].iat[0]
        dictR['citation_count'] = grupo.shape[0]
        dictR['number_authors'] = grupo['full_name_author'].nunique()
        dictR['number_countries'] = grupo['country_author'].nunique()
        return pd.Series(dictR)

    df_authors_citations = df_doc_authors_citations.loc[:,['full_name_citation']].reset_index()
    df_authors = df_doc_authors.loc[:,['full_name_author','country_author']].reset_index()

    df_authors_citations = df_authors_citations.drop_duplicates(subset=['article_id','full_name_citation'])
    df_authors = df_authors.drop_duplicates(subset=['article_id','full_name_author'])

    df_authors_and_citations = df_authors.merge(df_authors_citations, on='article_id')

    df_citation_plot = df_authors_and_citations.groupby(by=['full_name_citation'], as_index=False).apply(lambda g: agg_citations(g))

    df_citation_plot = df_citation_plot.sort_values(by=['citation_count'], ascending=False)
    df_citation_plot = df_citation_plot.head(100)

    fig = px.treemap(df_citation_plot, 
                     path=['full_name_citation'],
                     values='citation_count',
                     color='number_authors',
                     hover_data=['number_countries'],
                     color_continuous_scale='RdBu',
                     width=925,
                     height=400,
                     maxdepth=2,
                     labels={"number_authors":"Number of<br>Authors"})
    
    fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
    
    labels = ["Author Cited: %{id}",
               "Number of Citations: %{value}",
               "Number of Unique Authors Using This Citation: %{color}",
               "Number of Distinct Countries from Citation: %{customdata[0]}"]
                
    fig.update_traces(hovertemplate="<br>".join(labels))
    fig.update_traces(root_color="white")
    fig.update_traces(marker_line_width=4)
    
    st.plotly_chart(fig)
    
    with st.expander(" ❕ How can I read this chart?"):
        body = """Each rectangle represent an mentioned author, the size of space area mean the number of citations.<br>
                  The color represent the number of distinct authors that quote this author (cited).<br><br>
                  If you need more information, then you can pass cursor over each rectangle."""
        st.markdown(body, unsafe_allow_html=True)
    
    return dict_dfs


def plot_maps(st, dict_dfs, path):
    """Plot folium maps."""
    
    df_doc_authors = dict_dfs['df_doc_authors'].loc[:,getColumnsWithData(dict_dfs['df_doc_authors'])]
    
    path_geo = os.path.join(path,'data','0_external')
    country_latlong = pd.read_csv(os.path.join(path_geo,'countries_lat_long.csv'), sep=';', decimal='.')

    df_country_agg = df_doc_authors.groupby(by=['country_author'], as_index=False, dropna=True)['full_name_author'].count()

    shapes_correct = pd.read_csv(os.path.join(path_geo,'countries_correct.csv'), sep=';', decimal='.')
    dictCorrectShapes = {e[0]:e[1] for e in zip(shapes_correct.country_name, shapes_correct.country_target)}
    df_country_agg.country_author = df_country_agg.country_author.apply(lambda e: dictCorrectShapes.get(e,e))
    
    df_country_agg = df_country_agg.rename(columns={'country_author':'country',
                                                    'full_name_author':'count'})
    
    df_country_agg = df_country_agg.groupby(by=['country'], as_index=False)['count'].sum()
    df_country_agg = df_country_agg.merge(country_latlong, how='left', on='country')
    df_country_agg = df_country_agg.dropna()
    
    _, col1, _ = st.columns([0.75,3,0.1])
    with st.container():
    #     # with col1:
    #         # map = folium.Map(location=[25.552354,14.814465], zoom_start=1.5)
    #         # heat_points = []
    #         # for i,row in df_country_agg.iterrows():
    #         #     heat_points.append([row['Latitude'], row['Longitude'], row['count']])
    #         #     folium.Marker([row['Latitude'], row['Longitude']],
    #         #                 popup="<i>Number of Authors {0}<i>".format(row['count']),
    #         #                 tooltip=f"Number of Authors {row['count']}").add_to(map)
    #         #     st_point_map = st_folium(map)
        with col1:
            map = folium.Map(location=[25.552354,14.814465], zoom_start=1)
            heat_points = []
            for i,row in df_country_agg.iterrows():
                heat_points.append([row['latitude'], row['longitude'], row['count']])
            HeatMap(heat_points, radius=40, blur=20).add_to(map)
            st_heat_map = folium_static(map, width=600, height=400)
    
    return dict_dfs


# def part_of_speech(st, dict_dfs):

#     article_titles = dict_dfs['df_doc_head']['title_head'].tolist()
#     file_names = dict_dfs['df_doc_info']['file'].apply(lambda e: os.path.split(e)[-1])

#     list_articles = list(zip(article_titles,file_names))
#     list_articles_select = [' - '.join([str(e[0]),str(e[1])]) for e in list_articles]

#     abstracts = dict_dfs['df_doc_info']['abstract'].tolist()
#     list_articles_abstracts = list(zip(list_articles_select,abstracts))
#     dict_articles_abstracts = {e[0]:e[1] for e in list_articles_abstracts}
    
#     choice_exec = st.selectbox("Please, choose one article: ", list_articles_select, key="select_box_pos")
    
#     # Há mais modelos para testar aqui https://github.com/allenai/scispacy
#     # python -m spacy download en_core_web_sm
#     nlp = spacy.load("en_core_sci_sm")
#     doc = nlp(dict_articles_abstracts[choice_exec])
#     list_pos_tag = [(ent.text,ent.label_) for ent in doc.ents]
    
#     # col1, col2, col3 = st.columns([0.1,2,0.1])
#     # with col2:
    
#     doc.user_data["title"] = f"File: {choice_exec}"
    
#     # https://github.com/explosion/spacy-streamlit
#     visualize_ner(doc, show_table=False, title=False)
#     #visualize_parser(doc)
    
#     # annotated_text(*list_pos_tag)
    
    
def show_keywords_graph(st, dict_dfs, df_article_keywords_all, input_folder_path, tmining, folder_graph='graphs', 
                        name_file="graph_keywords.html", cache_folder_name='summarticles_cache', top_keywords=10, buttons=False):
    
    """"""
    
    if 'graph' not in dict_dfs['keywords']:
    
        dict_dfs['keywords']['graph'] = {}

        df_keyword_data = df_article_keywords_all.groupby(by=['keyword'], as_index=False).apply(agg_keys_node_data)
        df_keyword_data = df_keyword_data.sort_values(by=['article_count'], ascending=False).head(top_keywords)
        
        df_keyword_data['value_sum'] = df_keyword_data['value_sum'].apply(lambda e: np.round(e,2))
        df_keyword_data['value_mean'] = df_keyword_data['value_mean'].apply(lambda e: np.round(e,2))
        dict_dfs['keywords']['graph']['df_keyword_data'] = df_keyword_data
        
        # Selecting edges that contains top keywords
        filtro = (df_article_keywords_all.keyword.isin(df_keyword_data.keyword.tolist()))
        df_art_key_all = df_article_keywords_all.loc[(filtro)].copy()

        # Selecting nodes in the list of selected edges
        df_nodes = get_node_data(dict_dfs)

        filtro = (df_nodes['article_id'].isin(df_art_key_all['article_id'].tolist()))
        df_nodes = df_nodes.loc[(filtro)].copy()
        
        path_write_graph = os.path.join(input_folder_path, cache_folder_name)
            
        keywords_graph, path_graph, path_folder_graph = tmining.make_keywords_graph(edges_key_articles=df_art_key_all, node_data=df_nodes,
                                                        node_keywords_data=df_keyword_data, source_column="keyword", to_column="article_id", 
                                                        value_column="value", height="500px", width="100%", directed=False, notebook=False,
                                                        bgcolor="#ffffff", font_color=False, layout=None, heading="", path_graph=path_write_graph,
                                                        folder_graph=folder_graph, buttons=buttons, name_file=name_file)
        dict_dfs['keywords']['graph']['keywords_graph'] = keywords_graph
        dict_dfs['keywords']['graph']['path_graph'] = path_graph
        dict_dfs['keywords']['graph']['path_folder_graph'] = path_folder_graph
        
    col1, col2 = st.columns([2,1])
    with col1:
        show_graph(st, dict_dfs['keywords']['graph']['keywords_graph'],
                   dict_dfs['keywords']['graph']['path_graph'],
                   dict_dfs['keywords']['graph']['path_folder_graph'],
                   text_spinner='👁‍🗨 Similarity Graph: drawing...')
    with col2:
        df_show = dict_dfs['keywords']['graph']['df_keyword_data'].head(100)
        df_show = df_show.rename(columns={"keyword":"Keyword",
                                          "article_count":"Total Articles",
                                          "value_sum":"Total Relevance",
                                          "value_mean":"Mean Relevance"})
        AgGrid(df_show,
               data_return_mode='AS_INPUT', 
               # update_mode='MODEL_CHANGED', 
               fit_columns_on_grid_load=False,
               # theme='fresh',
               enable_enterprise_modules=False,
               height=510, 
               width='100%',
               reload_data=True)
        
    return dict_dfs


def years_plot_article(st, dict_dfs):
    
    import plotly.express as px

    df_doc_head = dict_dfs['df_doc_head']
    df_doc_info = dict_dfs['df_doc_info'].loc[:,getColumnsWithData(dict_dfs['df_doc_info'])]
    df_doc_info_head = df_doc_info.join(df_doc_head, how='left')
    df_doc_info_head.date_head = df_doc_info_head.date_head.apply(lambda e: pd.to_datetime(e))
    df_doc_info_head['year'] = df_doc_info_head.date_head.apply(lambda e: e if pd.isna(e) else int(e.year))
    df_doc_info_head.year.fillna('Null Value', inplace=True)
    data_year = df_doc_info_head.year.value_counts()
    df_year = pd.DataFrame({'year':data_year.index, 
                            'article_count':data_year.values})

    fig = px.pie(df_year,
                height=500,
                width=500,
                values='article_count', 
                names='year',
                title='Number of Articles by Year',
                hover_data=['year'], 
                labels={'values':'Percentage',
                        'year':'Year of Article',
                        'article_count':'Number of Articles'},
                hole=.5)

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)
    # fig.show()
    
    
def article_overview_information(st, dict_dfs):
    
    """"""
    
    import plotly.express as px
    
    df_doc_info = dict_dfs['df_doc_info'].loc[:,getColumnsWithData(dict_dfs['df_doc_info'])]
    df_doc_head = dict_dfs['df_doc_head'].loc[:,getColumnsWithData(dict_dfs['df_doc_head'])]
    df_doc_info_head = df_doc_info.join(df_doc_head, how='left')
    
    if 'date_head' in df_doc_info_head.columns.tolist():
        df_doc_info_head.date_head = df_doc_info_head.date_head.apply(lambda e: pd.to_datetime(e))
        df_doc_info_head['year'] = df_doc_info_head.date_head.apply(lambda e: e if pd.isna(e) else int(e.year))
        df_doc_info_head.year = df_doc_info_head.year.fillna('Null Value')
        fig = px.pie(df_doc_info_head, 
                    values='year', 
                    names='year',
                    title='Number of Articles by Year',
                    hover_data=['year'], 
                    labels={'values':'Percentage','year':'Year of Article'}, hole=.5)

        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)
        
    # _, col, _ = st.columns([0.1,3,0.1])
    # with col:
        # fig.show()
        
    return dict_dfs