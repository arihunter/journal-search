import streamlit as st
from pubmed_manager import PubmedManager
from llama_index import VectorStoreIndex
from llama_index.evaluation import DatasetGenerator
from llama_index.readers.schema.base import Document
import os 
import openai
import uuid


#initialisation
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
if "references" not in st.session_state:
    st.session_state["references"] = []


#streamlit code

st.set_page_config(page_title="fastbio")
col1,col2,col3 = st.columns([1,1,1])
col2.title("FastBio")
st.divider()

#sidebar
apiKey = st.sidebar.text_input("OpenAI API Key", type="password")
st.session_state.apikey = apiKey
if not apiKey:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()
openai.api_key = apiKey


#main content




#add logic to create a userId
#link userId to VectorStoreIndices
#@st.cache_resource(show_spinner=False)
class SearchBackend1():
  def __init__(self):
    #   self.indexCreated = False
    self.index = None
    #self.queryEngine = None
    #self.persistDir = "/Users/arihantbarjatya/Documents/fastbio/database_storage/stored_embeddings/pubmed"
    #self.userId = str(uuid.uuid4())
    self.pubObj = PubmedManager()


  #@st.cache_data(show_spinner=False)
  def fetch_docs(_self,query):
    docs = _self.pubObj.fetch_details(query)
    return docs
  
  def update_index(self,docs):  
    for doc in docs:
      self.index.insert(doc)


  # add userId logic so once a users DB is created its remembered
  @st.cache_data(show_spinner=False)
  def main(_self,query):
    with st.spinner("Creating the best response for you"):
        currentDocs,pubmedPapers = _self.fetch_docs(query)
        
        _self.index = VectorStoreIndex.from_documents(currentDocs)
        _self.queryEngine = _self.index.as_query_engine()
        response = _self.queryEngine.query(query)
        
        citations = []
        for node in response.source_nodes:
            title = node.node.metadata["Title of this paper"]
            url = node.node.metadata["URL"]
            citations.append((title,url))
    
    return response,citations,pubmedPapers

def searchButtonCallback():
    st.session_state.search = True


if st.session_state["search"] == False:
    # engine = st.selectbox('Select Engine',["Engine1","Engine2","Engine3"])   
    # st.session_state["engine"] = engine
    if st.session_state.query == None:
        userInput = st.text_input("Search with papers")
    else:
        userInput = st.text_input("Search with papers",value=st.session_state.query)
    st.session_state.query = userInput
    buttonClick = st.button("Ask",on_click=searchButtonCallback)

def log():
    pass

def editcallback():
    st.session_state["search"] = False
    st.session_state["response"] = None
    st.session_state["feedbackRating"] = None
    st.session_state["feedbackText"] = None
    #st.session_state["engine"] = None
    st.session_state["references"] = []

def reboot():
    st.session_state["search"] = False
    st.session_state["query"] = None
    st.session_state["response"] = None
    st.session_state["feedbackRating"] = None
    st.session_state["feedbackText"] = None
    #st.session_state["engine"] = None
    st.session_state["references"] = []

def generatedQuestionCallback(newQuery):
    log()
    st.session_state["query"] = newQuery
    st.session_state["response"] = None
    st.session_state["feedbackRating"] = None
    st.session_state["feedbackText"] = None
    st.session_state["references"] = []
    st.session_state["search"] = True



@st.cache_data(show_spinner=False,experimental_allow_widgets=True)
def createNewQuestions(query,response):
    responseDoc = Document(text=response,extra_info={"Original Query":query})
    dataGenerator = DatasetGenerator.from_documents([responseDoc])
    newQuestions = dataGenerator.generate_questions_from_nodes()
    #print(newQuestions)
    numberOfQuestions = 3        
    
    try:
        newQuestions = newQuestions[:numberOfQuestions]
    except Exception as e:
        pass
    
    newQuestions = sorted(newQuestions,key=len)     
    n = len(newQuestions)

    col1,col2,col3 = st.columns([0.3,0.3,0.4])

    try:
        col1.button(newQuestions[0],on_click=generatedQuestionCallback,args=[newQuestions[0]])
    except Exception as e:
        pass

    try:
        col2.button(newQuestions[1],on_click=generatedQuestionCallback,args=[newQuestions[1]])
    except Exception as e:
        pass

    try:
        col3.button(newQuestions[2],on_click=generatedQuestionCallback,args=[newQuestions[2]])
    except Exception as e:
        pass
    

searchObj1 = SearchBackend1()

if st.session_state.search:

    response,citations,pubmedPapers = searchObj1.main(st.session_state.query)
    st.session_state.response = str(response)
    st.session_state.references = citations


    tab1,tab2 = st.tabs(["Home","More Info!"])
    
    with tab1:
        queryCol1,queryCol2 = st.columns([0.8,0.2])
        with queryCol1:
            #st.subheader("Query")
            st.write(f'<p style="font-size:30px"><b>Query</b></p>',unsafe_allow_html=True)
            #st.markdown(st.session_state.query)
            st.write(f'{st.session_state.query}',
unsafe_allow_html=True)
        with queryCol2:
            st.button("Edit Query",on_click = editcallback)
        st.write(f'<p style="font-size:30px"><b>Response</b></p>',unsafe_allow_html=True)
        #st.markdown(f"*:{st.session_state.response}:*")
        st.write(f'<i>{st.session_state.response}</i>',
unsafe_allow_html=True)
        st.markdown("")
        st.markdown("")
        otherPapercheck = []
        with st.expander("Citations"):
            for i,reference in enumerate(citations):
                citationsCol1,citationsCol2 = st.columns([0.9,0.1])
                with citationsCol1:
                    st.caption(reference[0])
                    st.caption(reference[1])
                    otherPapercheck.append(str(reference[1]))
                with citationsCol2:
                    st.button(":thumbsdown:",key=f"Citations{i}")    

        st.markdown("")
        if response != None:
            createNewQuestions(st.session_state.query,st.session_state.response) 


        st.divider()
        st.subheader("Feedback")

        responseFeedback = st.radio('Choose for the generated response',options=('Correct Response, No Hallucinations','Hallucinations','Didnt Like the Response','No Response'))
        st.session_state["feedbackRating"] = responseFeedback
        if responseFeedback:
            feedbackText = st.text_area("Please help us understand your feedback better")
            st.session_state["feedbackText"] = feedbackText
        st.markdown("")
        st.markdown("") 
        finalCol1, finalCol2, finalCol3 = st.columns([1,1,1])
        finalCol2.button("Search Again!", on_click=reboot,type="primary")

    
    with tab2:
        with st.expander("Other relevant papers"):
            for i,data in enumerate(pubmedPapers):
                url = data["url"]
                url = str(url)
                relevantCol1,relevantCol2 = st.columns([0.9,0.1])
                if url not in otherPapercheck:
                    with relevantCol1:
                        st.caption(data["title"])
                        st.caption(url)
                    with relevantCol2:
                        st.button(":thumbsup:",key=f"{i}")



# import streamlit as st
# from pubmed_manager import PubmedManager
# from llama_index import VectorStoreIndex
# from llama_index.evaluation import DatasetGenerator
# from llama_index.readers.schema.base import Document
# import os 
# import openai
# import uuid


# #initialisation
# if "search" not in st.session_state:
#     st.session_state["search"] = False
# if "query" not in st.session_state:
#     st.session_state["query"] = None
# if "response" not in st.session_state:
#     st.session_state["response"] = None
# if "feedbackRating" not in st.session_state:
#     st.session_state["feedbackRating"] = None
# if "feedbackText" not in st.session_state:
#     st.session_state["feedbackText"] = None
# if "apikey" not in st.session_state:
#     st.session_state["apikey"] = None
# if "references" not in st.session_state:
#     st.session_state["references"] = []


# #streamlit code

# st.set_page_config(page_title="fastbio")
# col1,col2,col3 = st.columns([1,1,1])
# col2.title("FastBio")
# st.divider()

# #sidebar
# apiKey = st.sidebar.text_input("OpenAI API Key", type="password")
# st.session_state.apikey = apiKey
# if not apiKey:
#     st.info("Please add your OpenAI API key to continue.")
#     st.stop()
# openai.api_key = apiKey


# #main content




# #add logic to create a userId
# #link userId to VectorStoreIndices
# #@st.cache_resource(show_spinner=False)
# class SearchBackend1():
#   def __init__(self):
#     #   self.indexCreated = False
#     self.index = None
#     self.queryEngine = None
#     #self.persistDir = "/Users/arihantbarjatya/Documents/fastbio/database_storage/stored_embeddings/pubmed"
#     #self.userId = str(uuid.uuid4())
#     self.pubObj = PubmedManager()


#   #@st.cache_data(show_spinner=False)
#   def fetch_docs(_self,query):
#     docs,papers = _self.pubObj.fetch_details(query)
#     return docs,papers
  
#   def update_index(self,docs):  
#     for doc in docs:
#       self.index.insert(doc)


#   # add userId logic so once a users DB is created its remembered
#   @st.cache_data(show_spinner=False)
#   def main(_self,query):
#     with st.spinner("Creating the best response for you"):
#         currentDocs,ogPapers = _self.fetch_docs(query)
#         _self.index = VectorStoreIndex.from_documents(currentDocs)
#         _self.queryEngine = _self.index.as_query_engine()
        
#         # if self.indexCreated == False:
#         #     self.index = VectorStoreIndex.from_documents(currentDocs)
#         #     self.index.set_index_id(self.userId)
#         #     indexPath = self.persistDir+"/"+self.userId
#         #     if not os.path.exists(indexPath):
#         #         os.makedirs(indexPath)
#         #     self.index.storage_context.persist(indexPath)
#         #     self.indexCreated = True
#         #     self.queryEngine = self.index.as_query_engine()
#         # else:
#         #     self.update_index(currentDocs)
#         #     self.queryEngine = self.index.as_query_engine()
    
#         response = _self.queryEngine.query(query)
#         citations = []
#         for node in response.source_nodes:
#             title = node.node.metadata["Title of this paper"]
#             url = node.node.metadata["URL"]
#             citations.append((title,url))
    
#     return response,citations,ogPapers

# def searchButtonCallback():
# 	st.session_state.search = True


# if st.session_state["search"] == False:
#     # engine = st.selectbox('Select Engine',["Engine1","Engine2","Engine3"])   
#     # st.session_state["engine"] = engine
#     if st.session_state.query == None:
#         userInput = st.text_input("Search with papers")
#     else:
#         userInput = st.text_input("Search with papers",value=st.session_state.query)
#     st.session_state.query = userInput
#     buttonClick = st.button("Ask",on_click=searchButtonCallback)


# def editcallback():
#     st.session_state["search"] = False
#     st.session_state["response"] = None
#     st.session_state["feedbackRating"] = None
#     st.session_state["feedbackText"] = None
#     #st.session_state["engine"] = None
#     st.session_state["references"] = []

# def log():
#     pass

# def reboot():
#     log()
#     st.session_state["search"] = False
#     st.session_state["query"] = None
#     st.session_state["response"] = None
#     st.session_state["feedbackRating"] = None
#     st.session_state["feedbackText"] = None
#     #st.session_state["engine"] = None
#     st.session_state["references"] = []

# def feedbackEngine(_id):
#     key = f"feedbackEngine{_id}"
#     if key not in st.session_state:
#         st.session_state[key] = False
#     st.session_state[key] = True

# def feedbackPubmed(_id):
#     key = f"feedbackPubmed{_id}"
#     if key not in st.session_state:
#         st.session_state[key] = False
#     st.session_state[key] = True

# def generatedQuestionCallback(newQuery):
#     log()
#     st.session_state["query"] = newQuery
#     st.session_state["response"] = None
#     st.session_state["feedbackRating"] = None
#     st.session_state["feedbackText"] = None
#     st.session_state["references"] = []
#     st.session_state["search"] = True



# @st.cache_data(show_spinner=False,experimental_allow_widgets=True)
# def createNewQuestions(query,response):
#     responseDoc = Document(text=response,extra_info={"Original Query":query})
#     dataGenerator = DatasetGenerator.from_documents([responseDoc])
#     newQuestions = dataGenerator.generate_questions_from_nodes()
#     numberOfQuestions = 3        
    
#     try:
#         newQuestions = newQuestions[:numberOfQuestions]
#     except Exception as e:
#         pass
    
#     newQuestions = sorted(newQuestions,key=len)     
#     n = len(newQuestions)
#     for question in newQuestions:
#         st.button(question,on_click=generatedQuestionCallback,args=[question])

# searchObj1 = SearchBackend1()

# if st.session_state.search:

#     response,citations,pubmedPapers = searchObj1.main(st.session_state.query)
#     st.session_state.response = str(response)
#     st.session_state.references = citations
#     urlsOnly = [x[1] for x in citations]


#     tab1,tab2 = st.tabs[("Home","More Info!")]
    
#     with tab1:

#         queryCol1,queryCol2 = st.columns([0.8,0.2])
#         with queryCol1:
#             st.subheader("Query")
#             st.markdown(st.session_state.query)
#         with queryCol2:
#             st.button("Edit Query",on_click = editcallback)
#         st.subheader("Response")
#         st.markdown(st.session_state.response)


#         #homePageBaseDisplay(st.session_state.query,st.session_state.response)
#         if st.session_state.response != None:
#             createNewQuestions(st.session_state.query,st.session_state.response)
        
#         st.divider()

#         with st.expander("Citations for response"):
#             for i,reference in enumerate(citations):
#                 col1,col2 = st.columns([0.9,0.1])
#                 with col1:
#                     st.caption(reference[0])
#                     st.caption(reference[1])
#                 #otherPapercheck.append(str(reference[1]))
#                 with col2:
#                     st.button(":thumbsdown:",key=f"Citations{i}")

#         #homePageCitation(st.session_state.references)
#         st.divider()
        
#         st.subheader("Feedback")
#         responseFeedback = st.radio('Choose for the generated response',options=('Correct Response, No Hallucinations','Hallucinations','No Response'))
#         st.session_state["feedbackRating"] = responseFeedback
#         if responseFeedback:
#             feedbackText = st.text_area("Please help us understand your feedback better")
#             st.session_state["feedbackText"] = feedbackText
#         #homePageFeedback()
#         st.markdown("")
#         st.markdown("") 
        
#         col1, col2, col3 = st.columns([1,1,1])
#         col2.button("Search Again!",type="primary",on_click=reboot)


#     with tab2:

#         with st.expander("Other relevant papers"):
#             for i,data in enumerate(pubmedPapers):
#                 url = data["url"]
#                 url = str(url)
#                 col1,col2 = st.columns([0.9,0.1])
#                 if url not in otherPapercheck:
#                     with col1:
#                         st.caption(data["title"])
#                         st.caption(url)
#                     with col2:
#                         st.button(":thumbsup:",key=f"{i}")
        


#                 #st.button(label = ':thumbsdown')



        










    




