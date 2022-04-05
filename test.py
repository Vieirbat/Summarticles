import streamlit as st
##from st_annotated_text import annotated_text

import scispacy
import spacy
from spacy import displacy

from spacy_streamlit import visualize_ner

from grobid.client import GrobidClient
import pandas as pd


from pyecharts import options as opts
from pyecharts.charts import Bar
from streamlit_echarts import st_pyecharts

from article import Article
import dateparser

from collections import Counter

## import chemlistem as cl
## model = cl.get_ensemble_model() 

nlp = spacy.load("en_core_sci_scibert")

host = 'http://localhost'
port = 8070

client = GrobidClient(host, port)

# class DB:

#    def __init__(self):
#        self.grobidport = 8070
#        self.grobidhost = 'http://localhost'
#        self.db = Factory.create('./')
#        self.client = GrobidClient(self.grobidhost,
#                                   self.grobidport)

#    def add_file(self, file):
#        rsp, _ = self.client.serve("processFulltextDocument", file)
#        tei = rsp.text
#        article = TEI.parse(tei, None, "../models/")
#        self.db.save(article)


def process_file(file):
    rsp, _ = client.serve("processFulltextDocument", file)
    tei = rsp.text
    article = Article(rsp.text)
    return(article)


option = st.sidebar.radio(
    "Select an option",
    ('Explore Abstracts', 'Designs', 'Display Graphs', 'Search Web'))


@st.cache(allow_output_mutation=True)
def get_db():
    return pd.DataFrame(columns=['published',
                                 'publication',
                                 'authors',
                                 'title',
                                 'reference',
                                 'abstract',
                                 'text',
                                 'design'
                                ])


@st.cache(allow_output_mutation=True)
def get_stored_keys():
    """This dictionary is initialized once and can be used to
       store the files uploaded"""
    return set()


files = st.sidebar.file_uploader("Choose Files",
                                 type='pdf',
                                 accept_multiple_files=True,
                                 )


keys = get_stored_keys()
db = get_db()

if files is not None:
    current = set([f.id for f in files])
    additions = (current - keys)
    deletions = (keys - current)
    for file in files:
        if file.id in additions:
            keys.add(file.id)
            article = process_file(file)
            #article.annotate_abstract(model)
            db.loc[file.id] = article.get_data()
    for fid in deletions:
        db.drop(fid, inplace=True)
        keys.discard(fid)


if option == 'Explore Abstracts':
    for _, row in db.iterrows():
        with st.beta_expander(row['title']):
            st.write(",".join(row['authors']))
            #if 'abstract' in row['sections'].keys():
            doc = nlp(row['abstract'])
        
            html = displacy.render(
                doc, style="ent"
            )
            style = "<style>mark.entity { display: inline-block }</style>"
            st.write(html, unsafe_allow_html=True)

            #visualize_ner(doc, labels=nlp.get_pipe("ner").labels)


if option == 'Designs':
    designs = []
    for _, row in db.iterrows():
        designs.append(row['design'])

    designs = Counter(designs)


    b = (
        Bar()
        .add_xaxis(list(designs.keys()))
        .add_yaxis(
            "Counts", list(designs.values())
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
            title="Design types"
        ),
        toolbox_opts=opts.ToolboxOpts(),
        )
    )
    st_pyecharts(b)
