"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI



from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate


from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ChatVectorDBChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from typing import List
from langchain.llms import OpenAIChat
import json, os
from langchain.chains import VectorDBQAWithSourcesChain
from langchain import OpenAI
import pinecone 



basic_llm = OpenAI(temperature=0)
chat_llm = OpenAIChat(model_name="gpt-3.5-turbo")

base_embeddings = OpenAIEmbeddings()
hypothetical_embeddings = HypotheticalDocumentEmbedder.from_llm(chat_llm, base_embeddings, "web_search")


pinecone.init(
    api_key="8206321a-4b8f-46db-b782-ab8205385041",  # find at app.pinecone.io
    environment="us-west1-gcp"  # next to api key in console
)

index_name = "askhuberman"


# refine_template = (
#     "The original question is as follows: {question}\n"
#     "We have provided an existing answer, including sources (just the ones given in the metadata of the documents, don't make up your own sources): {existing_answer}\n"
#     "We have the opportunity to refine the existing answer"
#     "(only if needed) with some more context below.\n"
#     "------------\n"
#     "{context_str}\n"
#     "------------\n"
#     "Given the new context, add to the original answer to better "
#     "answer the question"
#     "If you do update it, please update the sources as well. "
#     "If the context isn't useful, print the original answer."
#     "Use only the context passed to you and original answer to give answers. Don't use any other information. Don't make up stuff on your own. If you don't have data, it's ok to just answer based on the given context and answer"
#     "The final answer should incorporate information from the original answer and the new context, but "
#     "don't make use of phrases like 'additional context', 'original answer','new context', 'old answer' \n "
#     "don't make use of phrases like 'additional context', 'original answer','new context', 'old answer'\n "
#     "Do not use external sources at any cost, use only the context provided.\n"
#     "Include the sources in the answer\n"
#     "Please update the answer only if the new information is relevant to the question.\n"
#     "Try to be concise and brief"
#     "Format the answer so that it's easy to skim"
# )
# refine_prompt = PromptTemplate(
#     input_variables=["question", "existing_answer", "context_str"],
#     template=refine_template,
# )


# question_template = (
#     "Context information is below. \n"
#     "---------------------\n"
#     "{context_str}"
#     "\n---------------------\n"
#     "Given the context information and not prior knowledge, "
#     "answer the question: {question}\n"
# )

# question_prompt = PromptTemplate(
#     input_variables=["context_str", "question"], template=question_template
# )


pinecone.list_indexes()
index = pinecone.Index(index_name)
# docsearch = Pinecone(index, hypothetical_embeddings.embed_query, 'text')




# doc_chain_refine = load_qa_with_sources_chain(chat_llm, chain_type="refine",question_prompt=question_prompt, refine_prompt=refine_prompt)

question_generator = LLMChain(llm=basic_llm, prompt=CONDENSE_QUESTION_PROMPT)

docsearch_base = Pinecone(index, base_embeddings.embed_query, 'text')

doc_chain_refine_base_chat = load_qa_with_sources_chain(chat_llm, chain_type="stuff")

qa_base1 = ChatVectorDBChain(vectorstore=docsearch_base, combine_docs_chain=doc_chain_refine_base_chat, question_generator=question_generator, return_source_documents=True, top_k_docs_for_context=7)


# query = "How to lose fat? Answer in bullet points"
# result = qa({"question": query, "chat_history": chat_history})

# print(result['answer'])

chain = qa_base1

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Ask Huberman Lab", page_icon=":robot:")
st.header("Ask Huberman Lab")


st.info("Sample queries: ")
st.success("How to overcome the afternoon slump?")
st.warning("What is fat mobilization?")
st.success("Give me 5 actionable tips to lose fat")
st.warning("What's a common precursor molecule to both dopamine and adrenaline?")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


with st.form("my_form"):
   query = st.text_input("Query")
   # Every form must have a submit button.
   submitted = st.form_submit_button("Ask")
   if submitted:       
        st.write("You asked: ", query, " ... Running...")

        print('getting output')
        output = chain({"question": query, "chat_history": []})['answer']
        st.session_state.past.append(query)

        st.session_state.generated.append(output)




if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        lastop= st.session_state["generated"][i].split('\n')
        for j in range(len(lastop)):
            if(lastop[j]=="" or lastop[j]==" "):
                continue
            message(lastop[j], key=f'{i}_{j}')
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
