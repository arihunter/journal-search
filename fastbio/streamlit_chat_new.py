import streamlit as st
import os
import asyncio
import time
from biorxiv_manager import BioRxivManager  
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

from typing import List, Optional
from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document
from llama_index import VectorStoreIndex


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

# if "rag_fail" not in st.session_state:
#     st.session_state["rag_fail"] = False

class PubmedReader(BaseReader):
    """Pubmed Reader.

    Gets a search query, return a list of Documents of the top corresponding scientific papers on Pubmed.
    """

    def load_data(
        self,
        search_query: str,
        max_results: Optional[int] = 10,
    ) -> List[Document]:
        """Search for a topic on Pubmed, fetch the text of the most relevant full-length papers.
        Args:
            search_query (str): A topic to search for (e.g. "Alzheimers").
            max_results (Optional[int]): Maximum number of papers to fetch.
        Returns:
            List[Document]: A list of Document objects.
        """
        import time
        import xml.etree.ElementTree as xml

        import requests

        pubmed_search = []
        parameters = {"tool": "tool", "email": "email", "db": "pmc"}
        parameters["term"] = search_query
        parameters["retmax"] = max_results
        resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=parameters,
        )
        root = xml.fromstring(resp.content)

        for elem in root.iter():
            if elem.tag == "Id":
                _id = elem.text
                url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?id={_id}&db=pmc"
                print(url)
                try:
                    resp = requests.get(url)
                    info = xml.fromstring(resp.content)

                    raw_text = ""
                    title = ""
                    journal = ""
                    for element in info.iter():
                        if element.tag == "article-title":
                            title = element.text
                        elif element.tag == "journal-title":
                            journal = element.text

                        if element.text:
                            raw_text += element.text.strip() + " "

                    pubmed_search.append(
                        {
                            "title": title,
                            "journal": journal,
                            "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{_id}/",
                            "text": raw_text,
                        }
                    )
                    time.sleep(1)  # API rate limits
                except Exception as e:
                    print(f"Unable to parse PMC{_id} or it does not exist:", e)
                    pass

        # Then get documents from Pubmed text, which includes abstracts
        pubmed_documents = []
        for paper in pubmed_search:
            pubmed_documents.append(
                Document(
                    text=paper["text"],
                    extra_info={
                        "Title of this paper": paper["title"],
                        "Journal it was published in:": paper["journal"],
                        "URL": paper["url"],
                    },
                )
            )

        return pubmed_documents

apiKey = st.sidebar.text_input("OpenAI API Key", type="password")
st.session_state.apikey = apiKey
openai.api_key = st.session_state.apikey
if not st.session_state.apikey:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

@st.cache_resource(show_spinner=False)
class SearchBackend1():
  def __init__(self):
    self.indexCreated = False
    self.index = None
    self.queryEngine = None
    self.pubObj = PubmedReader()


  #fetch k(20) relevant PubMed docs
  @st.cache_data(show_spinner=False)
  def fetch_docs(_self,query):
    docs = _self.pubObj.load_data(query,5)
    return docs
  
  def update_index(self,docs):  
    for doc in docs:
      self.index.insert(docs)

  @st.cache_data(show_spinner=False)
  def main(_self,query):
    currentDocs = _self.fetch_docs(query)
    if _self.indexCreated == False:
        _self.index = VectorStoreIndex.from_documents(currentDocs)
        _self.indexCreated = True
        _self.queryEngine = _self.index.as_query_engine()
    else:
        _self.update_index(currentDocs)
    
    response = _self.queryEngine.query(query)
    return response



@st.cache_resource(show_spinner=False)
class SearchBackend:
    def __init__(self,persistDir):
        self.debugger = LlamaDebugHandler(print_trace_on_end=True)
        self.callbackManager = CallbackManager([self.debugger])
        self.queryEmbedModel = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
        self.serviceContext = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0, chunk_size=512), embed_model=self.queryEmbedModel, callback_manager=self.callbackManager)
        self.storageContext = StorageContext.from_defaults(persist_dir=persistDir)
        self.biorxivDocsIndex = load_index_from_storage(self.storageContext, service_context=self.serviceContext)   
        self.biorxivDocsEngine = self.biorxivDocsIndex.as_query_engine(similarity_top_k=5, service_context=self.serviceContext)
        self.searchEngine1 = FLAREInstructQueryEngine(
            query_engine=self.biorxivDocsEngine,
            service_context=self.serviceContext,
            verbose=True
        )
        self.queryEngineTools = [
            QueryEngineTool(
                query_engine=self.biorxivDocsEngine,
                metadata=ToolMetadata(name="biorxiv_docs_engine", description="Provides information about the recent BioRxiv papers") 
            ), 
        ]
        self.searchEngine2 = SubQuestionQueryEngine.from_defaults(query_engine_tools=self.queryEngineTools, service_context=self.serviceContext, response_synthesizer=get_response_synthesizer(streaming=True), verbose=True, use_async=False)
        #self.queryEngine = self.biorxivDocsEngine.as_query_engine(service_context=self.serviceContext)
        self.stepDecomposeTransform = StepDecomposeQueryTransform(LLMPredictor(llm=OpenAI(model="gpt-3.5-turbo", temperature=0))) 
        self.indexSummary = "used to answer biology related questions from the context provided"
        self.searchEngine3 = MultiStepQueryEngine(
            query_engine=self.biorxivDocsEngine,
            query_transform=self.stepDecomposeTransform,
            index_summary=self.indexSummary,
        )

    @st.cache_data(show_spinner=False)
    def search(_self,query,engine):
        citations = []
        if engine == "Engine1":
            response = _self.searchEngine1.query(query)
            responseText = str(response)
            citedSources = response.source_nodes
            #citations = []
            #citedSourcesIds = []
            for node in citedSources:
                #citedSourcesText.append(node.node.text)
                citations.append(node.node.doc_id)
        if engine == "Engine2":
            #citations = []
            response = _self.searchEngine2.query(query)
            responseText = str(response)
            #citedSources = response.source_nodes
            # for node in citedSources:
            #     #citedSourcesText.append(node.node.text)
            #     citedSourcesText.append(node.node.doc_id)
            for i, (start_event, end_event) in enumerate(_self.debugger.get_event_pairs(CBEventType.SUB_QUESTION)):
                qa_pair = end_event.payload[EventPayload.SUB_QUESTION]
                question = qa_pair.sub_q.sub_question.strip()
                answer = qa_pair.answer.strip()
                sources = qa_pair.sources
                for node in sources:
                    citations.append(node.node.text)
                #citedSourcesText.append(sources)
                # if answer.contains("The given context information does not provide any information about"):
                #     st.session_state["rag_fail"] = True
        if engine == "Engine3":
            response = _self.searchEngine3.query(query)
            responseText = str(response)
            citedNodes = response.source_nodes
            for node in citedNodes:
                citations.append(node.node.text)
        return responseText,citations



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

persistDir = "/Users/arihantbarjatya/Documents/GitHub/Personal_Data_LLM/database_storage/stored_embeddings/biorxiv"
dfPath = "/Users/arihantbarjatya/Documents/GitHub/Personal_Data_LLM/Personal_Data_LLM/interfacingLLMs/logs.csv"
searchObj = SearchBackend(persistDir)
searchObj1 = SearchBackend1()

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
    
    response = searchObj1.main(st.session_state.query)
    st.session_state.response = str(response)
    citations = []

    for node in response.source_nodes:
        title = node.node.metadata["Title of this paper"]
        url = node.node.metadata["URL"]
        citations.append((title,url))
    
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
        for i,reference in enumerate(citations):
            st.caption(reference[0])
            st.caption(reference[1])
            

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





        



