import numpy as np
import pandas as pd
import os

from summautils import checkey

def agg_keys_node_data(grupo):
    """"""
    dictAgg = {}
    dictAgg['keyword'] = grupo['keyword'].iat[0]
    dictAgg['article_count'] = grupo['article_id'].shape[0]
    dictAgg['value_sum'] = grupo['value'].sum()
    dictAgg['value_mean'] = grupo['value'].mean()
    
    return pd.Series(dictAgg)


def get_node_data(dict_dfs):
    
    """"""
    
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
    authors_data = authors_data.groupby(by=['article_id'], as_index=False)['full_name_author'].count()
    authors_data.rename(columns={'full_name_author':'author_count'}, inplace=True)

    # Selecting citations information
    citations_data = dict_dfs['df_doc_citations'].reset_index()
    citations_data = citations_data.groupby(by=['article_id'], as_index=False)['index_citation'].count()
    citations_data.rename(columns={'index_citation':'citation_count'}, inplace=True)

    nodes = dict_dfs['df_doc_info'].reset_index()['article_id'].tolist()
    df_nodes = pd.DataFrame(nodes, columns=['article_id'])

    df_nodes = df_nodes.merge(head_data, how='left', on='article_id')
    df_nodes = df_nodes.merge(doc_info_data, how='left', on='article_id')
    df_nodes = df_nodes.merge(authors_data, how='left', on='article_id')
    df_nodes = df_nodes.merge(citations_data, how='left', on='article_id')
    
    return df_nodes


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
        authors_data = authors_data.groupby(by=['article_id'], as_index=False)['full_name_author'].count()
        authors_data.rename(columns={'full_name_author':'author_count'}, inplace=True)

        # Selecting citations information
        citations_data = dict_dfs['df_doc_citations'].reset_index()
        citations_data = citations_data.groupby(by=['article_id'], as_index=False)['index_citation'].count()
        citations_data.rename(columns={'index_citation':'citation_count'}, inplace=True)

        nodes = list(set(df_cos_sim_filter.doc_a.tolist() + df_cos_sim_filter.doc_b.tolist()))
        df_nodes = pd.DataFrame(nodes, columns=['article_id'])

        df_nodes = df_nodes.merge(head_data, how='left', on='article_id')
        df_nodes = df_nodes.merge(doc_info_data, how='left', on='article_id')
        df_nodes = df_nodes.merge(authors_data, how='left', on='article_id')
        df_nodes = df_nodes.merge(citations_data, how='left', on='article_id')

    return df_nodes


def text_preparation(st, dict_dfs, input_folder_path, tprep):
    
    """"""
    
    if not checkey(dict_dfs,'text_preparation'):
        
        dict_dfs['text_preparation'] = {}
        
        dict_dfs['df_doc_info']['abstract_prep'] = tprep.text_preparation_column(dict_dfs['df_doc_info']['abstract'])
        dict_dfs['text_preparation']['abstract_prep'] = dict_dfs['df_doc_info']['abstract_prep']
        
        # dict_dfs['df_doc_info']['acknowledgement_prep'] = tprep.text_prep_column(dict_dfs['df_doc_info']['acknowledgement'])
        
        dict_dfs['df_doc_info']['body_prep'] = tprep.text_preparation_column(dict_dfs['df_doc_info']['body'])
        dict_dfs['text_preparation']['body_prep'] = dict_dfs['df_doc_info']['body_prep']
    else:
        dict_dfs['df_doc_info']['abstract_prep'] = dict_dfs['text_preparation']['abstract_prep']
        dict_dfs['df_doc_info']['body_prep'] = dict_dfs['text_preparation']['body_prep']
        
    return dict_dfs


def cossine_similarity_data(st, dict_dfs, tmining, column='abstract', n_sim=200,
                            percentil="99%", sim_value_min=0, sim_value_max=0.99):
    
    """"""
    
    with st.spinner('📄✢📄 Making similarity relations...'):
        
        documents = dict_dfs['df_doc_info'][column + '_prep'].fillna(' ').tolist()
        df_tfidf_abstract = tmining.get_df_tfidf(documents)
        
        df_cos_tfidf_sim = tmining.get_cossine_similarity_matrix(df_tfidf_abstract,
                                                                 dict_dfs['df_doc_info'].index.tolist())
        
    with st.spinner('📑🔍 Filtering best similarities relations...'):
        
        df_cos_tfidf_sim_filter = tmining.filter_sim_matrix(df_cos_tfidf_sim,
                                                            df_cos_tfidf_sim.index.tolist(),
                                                            n_sim=n_sim,
                                                            percentil=percentil,
                                                            value_min=sim_value_min,
                                                            value_max=sim_value_max)
    
    return  df_cos_tfidf_sim_filter