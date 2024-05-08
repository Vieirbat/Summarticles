import os
import pandas as pd
import plotly.express as px

def checkey(dic, key):
    """"""
    return True if key in dic else False


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


def get_path(path_input_path):
    """"""
    if os.path.exists(path_input_path):
        return path_input_path
    
    return os.getcwd()


def clean_error_results(result_batch):
    
    """"""
    
    new_result = []
    for result in result_batch:
        if "500" not in str(result[1]):
            new_result.append(result)
    return new_result


def files_path(path):
    
    """"""
    
    list_dir = os.listdir(path)
    files = []
    for file in list_dir:
        if os.path.isfile(os.path.join(path,file)):
            files.append(os.path.join(path,file))
    return files


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

def get_dataframes(result_batch, xml_to_df):
    
    """"""
    
    # xml_to_df = xmltei_to_dataframe()
    dict_dfs, dict_errors = xml_to_df.get_dataframe_articles(result_batch)
    
    return dict_dfs, dict_errors