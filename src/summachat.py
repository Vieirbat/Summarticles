# --------------------------------------------------------------------------------------------------------
# O streamlit chat é um plugin do Streamlit para criar chats
# Já vem com uma coleção de utilitários para chat no streamlit
# https://github.com/AI-Yash/st-chat

from streamlit_chat import message

# ------------------------------------------------------------------------------------------------------
# Essa cadeia coleta o histórico do bate-papo (uma lista de mensagens) e novas perguntas e, em seguida, 
# retorna uma resposta a essas perguntas. O algoritmo para esta cadeia consiste em três partes:

# 1. Use o histórico de bate-papo e a nova pergunta para criar uma “pergunta independente”. 
#    Isso é feito para que esta questão possa ser passada para a etapa de recuperação para buscar documentos relevantes. 
#    Se apenas a nova pergunta foi transmitida, pode faltar contexto relevante. 
#    Se toda a conversa for recuperada, pode haver informações desnecessárias que poderiam desviar a atenção da recuperação.

# 2. Esta nova pergunta é passada ao recuperador e os documentos relevantes são devolvidos.

# 3. Os documentos recuperados são passados ​​para um LLM juntamente com a nova pergunta (comportamento padrão)
# ou a pergunta original e o histórico de bate-papo para gerar uma resposta final.
# https://python.langchain.com/docs/modules/chains/

from langchain.chains import ConversationalRetrievalChain

# --------------------------------------------------------------------------------------------------------
# Aqui estamos olhando os modelos de embbedding do langchain, especificamente os embeddings do huggingface
# O embedding serve para transformar nossos textos em números, esses embeddings já estão pré-treinados
# https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub/
# https://huggingface.co/blog/getting-started-with-embeddings model names
# https://huggingface.co/sentence-transformers and https://www.sbert.net/
# Aqui está a lista de modelos pré-treinados: https://www.sbert.net/docs/pretrained_models.html

# As incorporações criam uma representação vetorial de um trecho de texto. 
# Isso é útil porque significa que podemos pensar no texto no espaço vetorial e fazer coisas como pesquisa semântica,
# onde procuramos trechos de texto mais semelhantes no espaço vetorial.
 
from langchain.embeddings import HuggingFaceEmbeddings

# --------------------------------------------------------------------------------------------------------
# Large Language Models (LLMs) são um componente central do LangChain. L
# angChain não oferece seus próprios LLMs, mas fornece uma interface padrão para interagir com muitos LLMs diferentes. 
# Para ser mais específico, essa interface recebe como entrada uma string e retorna uma string.

# O principal objetivo llama.cpp é permitir a inferência LLM com configuração mínima e desempenho de última geração
# em uma ampla variedade de hardware - localmente e na nuvem.

# llama-cpp-python é uma ligação Python para llama.cpp .
# Ele suporta inferência para muitos modelos de LLMs, que podem ser acessados ​​em Hugging Face .

from langchain_community.llms import LlamaCpp # pip install langchain-community --upgrade

# --------------------------------------------------------------------------------------------------------
# Este divisor de texto é o recomendado para texto genérico. É parametrizado por uma lista de caracteres. 
# Ele tenta dividi-los em ordem até que os pedaços sejam pequenos o suficiente. A lista padrão é ["\n\n", "\n", " ", ""]. 
# Isso tem o efeito de tentar manter todos os parágrafos (e depois as sentenças e depois as palavras) juntos pelo maior tempo possível, 
# já que esses geralmente pareceriam ser os trechos de texto semanticamente mais fortes.
# https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter/
# Entenda melhor na documentação: https://python.langchain.com/docs/modules/data_connection/document_transformers/

from langchain.text_splitter import RecursiveCharacterTextSplitter

# --------------------------------------------------------------------------------------------------------
# https://python.langchain.com/docs/modules/data_connection/vectorstores/

# Uma das maneiras mais comuns de armazenar e pesquisar dados não estruturados é incorporá-los 
# e armazenar os vetores de incorporação resultantes e, em seguida, no momento da consulta, 
# incorporar a consulta não estruturada e recuperar os vetores de incorporação que são 'mais semelhantes' à consulta incorporada.
# Um armazenamento de vetores se encarrega de armazenar dados incorporados e realizar pesquisas de vetores para você.

# Facebook AI Similarity Search (Faiss) é uma biblioteca para pesquisa eficiente de similaridade e agrupamento de vetores densos. 
# Contém algoritmos que pesquisam em conjuntos de vetores de qualquer tamanho, até aqueles que possivelmente não cabem na RAM. 
# Ele também contém código de suporte para avaliação e ajuste de parâmetros.
# https://faiss.ai/
# chromadb é outra forma de fazer isso

# As incorporações criam uma representação vetorial de um trecho de texto. 
# Isso é útil porque significa que podemos pensar no texto no espaço vetorial e fazer coisas como pesquisa semântica,
# onde procuramos trechos de texto mais semelhantes no espaço vetorial.

from langchain_community.vectorstores import FAISS

# --------------------------------------------------------------------------------------------------------
# A maioria dos aplicativos LLM possui uma interface conversacional. 
# Um componente essencial de uma conversa é ser capaz de fazer referência a informações introduzidas anteriormente na conversa.
# No mínimo, um sistema de conversação deve ser capaz de acessar diretamente alguma janela de mensagens anteriores.

# Chamamos essa capacidade de armazenar informações sobre interações passadas de “memória”.
# LangChain fornece muitos utilitários para adicionar memória a um sistema. 
# Esses utilitários podem ser usados ​​sozinhos ou incorporados perfeitamente em uma cadeia.

# https://python.langchain.com/docs/modules/memory/ 

from langchain.memory import ConversationBufferMemory

from langchain_core.documents.base import Document

# --------------------------------------------------------------------------------------------------------
# Deepseek 

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

import openai
from openai import OpenAI

# --------------------------------------------------------------------------------------------------------
# A biblioteca abaixo é para ler dados de pdfs
# from langchain.document_loaders import PyPDFLoader

# OS é uma biblioteca para utilização de recursos do SO
import os

# Para gerar arquivos e diretórios temporários
import tempfile

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]


def display_chat_history(st, chain):

    reply_container = st.container()
    container = st.container()

    with container:

        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Article PDF", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="avataaars-neutral",)
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts-neutral")


def display_chat_history_openai(st, client):

    with st.chat_message("assistant"):

        messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state['messages']]

        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=messages,
            stream=True
        )

        response = st.write_stream(stream)
        
    st.session_state['messages'].append({"role": "assistant", "content": response})


def create_conversational_chain(vector_store, llm, model="local"):

    # Create llm
    # Carregando LLM
    # llm = LlamaCpp(streaming = True,
    #                model_path="llama-2-7b-chat.Q2_K.gguf", # "llama-2-7b-chat.Q5_K_S.gguf", #"mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    #                temperature=0.75,
    #                top_p=1, 
    #                verbose=True,
    #                n_ctx=4096)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    if model == "local":
        chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                      retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                      memory=memory)

    return chain


def load_llm_model(model_paph="llama-2-7b-chat.Q2_K.gguf"):
    
    llm = LlamaCpp(streaming=True,
                   model_path=model_paph, # "llama-2-7b-chat.Q5_K_S.gguf", #"mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                   temperature=0.7,
                   tfs=0.95,
                   top_p=1, 
                   verbose=True,
                   n_ctx=10240,
                   top_k=0)
    
    return llm


def make_vector_store(docs):
    
    article_data = tuple(zip(docs['df_doc_info']['abstract'], docs['df_doc_info']['body']))
    article_text = []
    
    for i, texts in article_data:
        # make document for langchain
        doc_str = ' '.join(texts)
        doc = Document(page_content=doc_str)
        article_text.append(doc)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(article_text)

    # Create embeddings
    # model_id = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"  
    # model_id = "sentence-transformers/all-MiniLM-L6-v2"
    
    model_id = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_id, 
                                       model_kwargs={'device': 'cpu'})

    # Create vector store
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    
    return vector_store

# ------------------------------------------------------------------------------------------------------------------------
# Using Llama CCP

    # with st.spinner('📄➞📄  Creating Vector Store...'):
    #     vector_store = make_vector_store(st.session_state['dict_dfs'])
    
    # with st.spinner('📄➞📄  Loading LLM Model...'):
    #     model_file_name = "llama-2-7b-chat.Q2_K.gguf"
    #     path_llm = os.path.join(path,"models",model_file_name)
    #     llm_model = load_llm_model(model_paph=path_llm)
    
    # chain = create_conversational_chain(vector_store, llm_model)
    # display_chat_history(st, chain)

# ------------------------------------------------------------------------------------------------------------------------
# Ollama use below

def get_template():

    template = """
        You are an assistant for question-answering tasks. 
        Use the following articles text data of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:
    """

    return template


def get_articles_information(list_documents):
    article_text = []  
    for text in list_documents:
        doc = Document(page_content=text)
        article_text.append(doc)

    return article_text


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    return text_splitter.split_documents(documents)


def create_vector_store(documents, model_name="llama3.2:1b"):
    embeddings = OllamaEmbeddings(model=model_name)
    vector_store = InMemoryVectorStore(embeddings)
    chunked_documents = split_text(documents)
    vector_store.add_documents(chunked_documents)

    return vector_store


def rag(st, query, template, vector_store, model):

    related_documents = vector_store.similarity_search(query)

    context = "\n\n".join([doc.page_content for doc in related_documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    with st.spinner('📄➞📄  Generating a response...'):
        answer = chain.invoke({"question": query, "context": context})

    return answer


def get_model(model_name='llama3.2:1b'):
    model = OllamaLLM(model=model_name)
    return model


def summachat_ollama(st, model_type='llamma', model_name='llama3.2:1b'):

    if not st.session_state['summachat'].get('template', ''):
        st.session_state['summachat']['template'] = get_template()

    if not st.session_state['summachat'].get('article_text', []):
        article_data = st.session_state['dict_dfs']['df_doc_info']['abstract']
        st.session_state['summachat']['article_text'] = get_articles_information(article_data)

    if not st.session_state['summachat'][f'local_{model_type}'].get('vector_store', False):
        with st.spinner('📄➞📄  Creating Vector Store...'):
            vector_store = create_vector_store(st.session_state['summachat']['article_text'], model_name=model_name)
            st.session_state['summachat'][f'local_{model_type}']['vector_store'] = vector_store

    if not st.session_state['summachat'][f'local_{model_type}'].get('model', False):
        with st.spinner('📄➞📄  Loading LLM Model...'):
            model = get_model(model_name)
            st.session_state['summachat'][f'local_{model_type}']['model'] = model

    question = st.chat_input()
    if question:

        st.chat_message('user').write(question)

        answer = rag(st, question, st.session_state['summachat']['template'],
                     st.session_state['summachat'][f'local_{model_type}']['vector_store'],
                     st.session_state['summachat'][f'local_{model_type}']['model'])

        answer = answer.split('</think>')[-1] if model_type=='deepseek' else answer
        st.chat_message("assistant").write(answer)


def summachat_api(st, model_type="openai"):

    # try:

    if model_type=="openai":
        client = OpenAI(api_key=st.session_state['summachat'][f'api_key_{model_type}'])
    elif model_type=="deepseek":
        client = OpenAI(base_url="https://api.deepseek.com",
                        api_key=st.session_state['summachat'][f'api_key_{model_type}'])

    with st.container():

        if not st.session_state['summachat'].get('article_text', []):
            article_data = st.session_state['dict_dfs']['df_doc_info']['abstract']
            st.session_state['summachat']['article_text_list'] = article_data
        
        if not st.session_state['summachat'].get('context', False):
            context = 'Article Context: ' + '\n\n'.join(st.session_state['summachat']['article_text_list'])
            st.session_state['summachat']['context'] = context

        if not st.session_state['summachat'].get('api_prompt', False):

            api_prompt = """You are an assistant for question-answering tasks. 
                            Use the following articles text data of retrieved context to answer the question. 
                            If you don't know the answer, just say that you don't know. 
                            Use three sentences maximum and keep the answer concise.
                            Context: {context}"""

            st.session_state['summachat']['api_prompt'] = api_prompt

        display_messages(st, st.session_state['summachat']['messages'])

        if prompt := st.chat_input("Ask to your article files, what do you want to know?"):

            # Add user message to chat history
            st.session_state['summachat']['messages'].append({"role": "user", "content": prompt})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):

                with st.spinner('📄➞📄 Generating a response...'):

                    m0 = [{"role": "system",
                           "content": st.session_state['summachat']['api_prompt'].format(context=st.session_state['summachat']['context'])}]
                    messages = m0 + [{"role": m["role"], "content": m["content"]} for m in st.session_state['summachat']['messages']]

                    stream = client.chat.completions.create(
                        model=st.session_state['summachat'][f"{model_type}_model"],
                        messages=messages,
                        stream=True
                    )
                    
                    response = st.write_stream(stream)
                    
                    st.session_state['summachat']['messages'].append({"role": "assistant", "content": response})

    # except Exception as error:
    #     st.session_state[f'api_key_{model_type}'] = ''
    #     st.session_state['messages'] = []
    #     st.error(error)


def display_messages(st, messages):
    if len(messages):
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def summachat_variables(st):

    # For SummaChat
    if 'summachat' not in st.session_state:
        st.session_state['summachat'] = {}

    if 'local_deepseek' not in st.session_state['summachat']:
        st.session_state['summachat']['local_deepseek'] = {}

    if 'local_llamma' not in st.session_state['summachat']:
        st.session_state['summachat']['local_llamma'] = {}

    if 'template' not in st.session_state['summachat']:
        st.session_state['summachat']['template'] = ''

    if 'article_text' not in st.session_state['summachat']:
        st.session_state['summachat']['article_text'] = []

    if 'article_text_list' not in st.session_state['summachat']:
        st.session_state['summachat']['article_text_list'] = []

    if 'vector_store' not in st.session_state['summachat']['local_deepseek']:
        st.session_state['summachat']['local_deepseek']['vector_store'] = []

    if 'model' not in st.session_state['summachat']['local_deepseek']:
        st.session_state['summachat']['local_deepseek']['model'] = []

    if 'vector_store' not in st.session_state['summachat']['local_llamma']:
        st.session_state['summachat']['local_llamma']['vector_store'] = []

    if 'model' not in st.session_state['summachat']['local_llamma']:
        st.session_state['summachat']['local_llamma']['model'] = []

    if 'api_prompt' not in st.session_state['summachat']:
        st.session_state['summachat']['api_prompt'] = ''

    if 'context' not in st.session_state['summachat']:
        st.session_state['summachat']['context'] = ''

    if 'history' not in st.session_state['summachat']:
        st.session_state['summachat']['history'] = []

    if 'generated' not in st.session_state['summachat']:
        st.session_state['summachat']['generated'] = ["Welcome to SummaChat, how can I help you? 🤖"]

    if 'past' not in st.session_state['summachat']:
        st.session_state['summachat']['past'] = ["Hi!"]

    if 'api_key_openai' not in st.session_state['summachat']:
        st.session_state['summachat']['api_key_openai'] = ''

    if 'api_key_deepseek' not in st.session_state['summachat']:
        st.session_state['summachat']['api_key_deepseek'] = ''

    if 'rb_modelchat' not in st.session_state['summachat']:
        st.session_state['summachat']['rb_modelchat'] = 'Disable SummaChat'

    if "openai_model" not in st.session_state['summachat']:
        st.session_state['summachat']["openai_model"] = "gpt-4o-mini"

    if "deepseek_model" not in st.session_state['summachat']:
        st.session_state['summachat']["deepseek_model"] = "deepseek-chat"

    if "messages" not in st.session_state['summachat']:
        st.session_state['summachat']['messages'] = []