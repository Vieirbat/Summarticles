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

from langchain.llms import LlamaCpp

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

from langchain.vectorstores import FAISS

# --------------------------------------------------------------------------------------------------------
# A maioria dos aplicativos LLM possui uma interface conversacional. 
# Um componente essencial de uma conversa é ser capaz de fazer referência a informações introduzidas anteriormente na conversa.
# No mínimo, um sistema de conversação deve ser capaz de acessar diretamente alguma janela de mensagens anteriores.

# Chamamos essa capacidade de armazenar informações sobre interações passadas de “memória”.
# LangChain fornece muitos utilitários para adicionar memória a um sistema. 
# Esses utilitários podem ser usados ​​sozinhos ou incorporados perfeitamente em uma cadeia.

# https://python.langchain.com/docs/modules/memory/ 

from langchain.memory import ConversationBufferMemory

# --------------------------------------------------------------------------------------------------------
# A biblioteca abaixo é para ler dados de pdfs
from langchain.document_loaders import PyPDFLoader

# OS é uma biblioteca para utilização de recursos do SO
import os

# Para gerar arquivos e diretórios temporários
import tempfile


def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]


def display_chat_history(chain, st):
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


def create_conversational_chain(vector_store, llm):
    # Create llm
    # Carregando LLM
    # llm = LlamaCpp(streaming = True,
    #                model_path="llama-2-7b-chat.Q2_K.gguf", # "llama-2-7b-chat.Q5_K_S.gguf", #"mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    #                temperature=0.75,
    #                top_p=1, 
    #                verbose=True,
    #                n_ctx=4096)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain