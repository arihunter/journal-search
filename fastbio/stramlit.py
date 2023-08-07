import streamlit as st
import os
import asyncio
import time
from pubmed_manager import PubmedManager
from llama_index import (
    load_index_from_storage, 
    ServiceContext, 
    StorageContext, 
    LangchainEmbedding,
    LLMPredictor
)
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.llms import OpenAI
import openai
from llama_index.query_engine import FLAREInstructQueryEngine
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.query_engine.multistep_query_engine import MultiStepQueryEngine
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.question_gen.types import SubQuestion
from llama_index.indices.query.query_transform.base import StepDecomposeQueryTransform
import pandas as pd
import multiprocessing
from typing import List, Optional
from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document
from llama_index import VectorStoreIndex
from joblib import Parallel, delayed


st.set_page_config(page_title="fastbio")
col1,col2,col3 = st.columns([1,1,1])
col2.title("FastBio")
st.divider()

if "search" not in st.session_state:
    st.session_state["search"] = False

if "query" not in st.session_state:
    st.session_state["query"] = None

if "response" not in st.session_state:
    st.session_state["response"] = None

if "feedbackRating" not in st.session_state:
    st.session_state["feedbackRating"] = None

if "feedbackText" not in st.session_state:
    st.session_state["feedbackText"] = None

if "apikey" not in st.session_state:
    st.session_state["apikey"] = None

if "engine" not in st.session_state:
    st.session_state["engine"] = None

if "references" not in st.session_state:
    st.session_state["references"] = []

if "searchobj" not in st.session_state:
    st.session_state["searchobj"] = None

# if "rag_fail" not in st.session_state:
#     st.session_state["rag_fail"] = False



apiKey = st.sidebar.text_input("OpenAI API Key", type="password")
st.session_state.apikey = apiKey
openai.api_key = st.session_state.apikey
if not st.session_state.apikey:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

@st.cache_resource(show_spinner=False)
class SearchBackend1():
  def __init__(self,persistDir):
    self.debugger = LlamaDebugHandler(print_trace_on_end=True)
    self.callbackManager = CallbackManager([self.debugger])
    self.queryEmbedModel = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
    self.storageContext = StorageContext.from_defaults(persist_dir=persistDir)
    self.serviceContext = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0, chunk_size=512), embed_model=self.queryEmbedModel, callback_manager=self.callbackManager)
    self.index = load_index_from_storage(self.storageContext, service_context=self.serviceContext) 
    self.queryEngine = self.index.as_query_engine()
    self.pubMedManager = PubmedManager()
  
  
  def update_index(self,nodes):
      for node in nodes:
          self.index.insert(node)

  @st.cache_data(show_spinner=False)    
  def get_nodes(_self,query):
      nodes = _self.pubMedManager.load_data(query)
      return nodes

  
  def search(self,query):
    docs = self.get_nodes(query)
    self.update_index(docs)
    response = self.queryEngine.query(query)
    citations = []
    for node in response.source_nodes:
        print(node)
        break
        title = node.node.metadata["Title of this paper"]
        url = node.node.metadata["URL"]
        citations.append((title,url))
    response = str(response)
    return response,citations




def searchButtonCallback():
    st.session_state.search = True




if st.session_state["search"] == False:
    engine = st.selectbox('Select Engine',["Engine1","Engine2","Engine3"])   
    st.session_state["engine"] = engine
    if st.session_state.query == None:
        userInput = st.text_input("Search with papers")
    else:
        userInput = st.text_input("Search with papers",value=st.session_state.query)
    st.session_state.query = userInput
    buttonClick = st.button("Ask",on_click=searchButtonCallback)




dfPath = "/Users/arihantbarjatya/Documents/GitHub/Personal_Data_LLM/Personal_Data_LLM/interfacingLLMs/logs.csv"
#searchObj = SearchBackend(persistDir)
persistDir = "/Users/arihantbarjatya/Documents/GitHub/Personal_Data_LLM/Personal_Data_LLM/database_storage/stored_embeddings/pubmed"
searchObj1 = SearchBackend1(persistDir)

def rebootandlog():
    # currentRow = [st.session_state["query"],st.session_state["response"],st.session_state["feedbackRating"],st.session_state["feedbackText"],st.session_state["references"],st.session_state["engine"]]
    # df = pd.concat([df, pd.DataFrame([currentRow])], ignore_index=True)
    # df.to_csv(dfPath)

    st.session_state["search"] = False
    st.session_state["query"] = None
    st.session_state["response"] = None
    st.session_state["feedbackRating"] = None
    st.session_state["feedbackText"] = None
    st.session_state["engine"] = None
    st.session_state["references"] = []
    #st.session_state["rag_fail"] = False

def editcallback():
    st.session_state["search"] = False
    st.session_state["response"] = None
    st.session_state["feedbackRating"] = None
    st.session_state["feedbackText"] = None
    st.session_state["engine"] = None
    st.session_state["references"] = []


if st.session_state.search:
    
    response,citations = searchObj1.search(st.session_state.query)
    st.session_state.response = str(response)    
    col1,col2 = st.columns([0.8,0.2])
    st.session_state.references = citations
    with col1:
        st.subheader("Query")
        st.markdown(st.session_state.query)
    with col2:
        st.button("Edit Query",on_click = editcallback)
    st.subheader("Response")
    st.markdown(st.session_state.response)
    with st.expander("Citations"):
        for reference in citations:
            st.caption(f"Title: {reference[0]}")
            st.caption(f"Link: {reference[1]}")
            

    st.divider()
    st.write("Feedback")
    responseFeedback = st.radio('',options=('Correct Response, No Hallucinations','Hallucinations','No Response'))
    st.session_state["feedbackRating"] = responseFeedback
    if responseFeedback:
        feedbackText = st.text_area("Please help us understand your feedback better")
        st.session_state["feedbackText"] = feedbackText
    st.markdown("")
    st.markdown("") 
    col1, col2, col3 = st.columns([1,1,1])
    col2.button("Search Again!", on_click=rebootandlog)

    

    # with st.spinner("Creating the best result for you"):
    #     response,references = searchObj.search(st.session_state.query,st.session_state.engine)
    # st.session_state.response = response
    # st.session_state.references = references
    # col1,col2 = st.columns([0.8,0.2])
    # with col1:
    #     st.subheader("Query")
    #     st.markdown(st.session_state.query)
    # with col2:
    #     st.button("Edit Query",on_click = editcallback)
    # st.subheader("Response")
    # st.markdown(st.session_state.response)
    # with st.expander("How was this response generated ?"):
    #     for i,reference in enumerate(references):
    #         st.caption(reference)

    # # showSources = st.checkbox("*Click here to see how we generated the response*")
    # # if showSources:
    # #     # for i,reference in enumerate(references):
    # #     #     st.caption(f"Subquestion{i} : {reference[0]}")
    # #     #     st.caption(f"Response : {reference[1]}")
    # st.divider()
    # st.subheader("Feedback")
    # responseFeedback =  st.radio('',options=('Correct Response, No Hallucinations','Hallucinations','No Response'))
    # st.session_state["feedbackRating"] = responseFeedback
    # if responseFeedback:
    #     feedbackText = st.text_area("Please help us understand your feedback better")
    #     st.session_state["feedbackText"] = feedbackText
    # st.markdown("")
    # st.markdown("") 
    # col1, col2, col3 = st.columns([1,1,1])
    # col2.button("Search Again!", on_click=rebootandlog)

    # with st.form("my_form"):
    #     st.session_state.response = response
    #     st.markdown(f"Query: {st.session_state.query}")
    #     st.markdown(f"Response: {response}")
    #     showSources = st.checkbox("References")
    #     if showSources:
    #         for i,reference in enumerate(references):
    #             st.caption(f"{i}-> {reference}")
    #     responseFeedback = st.radio('Rate the response',options=('Good','Bad','Ugly'))
    #     st.session_state["feedbackRating"] = responseFeedback
    #     if responseFeedback:
    #         feedbackText = st.text_area("Please help us understand your feedback better")
    #         st.session_state["feedbackText"] = feedbackText
    #     newSearch = st.form_submit_button("Search Again")
    #     if newSearch:
    #         st.write(st.session_state)
    #         reboot()





        



