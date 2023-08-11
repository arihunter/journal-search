from typing import List, Optional
from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document
import asyncio
from Bio import Entrez
import time
import xml.etree.ElementTree as xml
import requests
from metapub import PubMedFetcher
import os

class PubmedManager(BaseReader):

    def __init__(self):
        os.environ["NCBI_API_KEY"] = "71836676d726923c4d2847443faae0fef308"
        self.cachePath = "/Users/arihantbarjatya/Documents/fastbio/database_storage/stored_embeddings/pubmed"
    
    def search(self,query):
        Entrez.email = 'arihantbadjatya@gmail.com'
        handle = Entrez.esearch(db='pubmed',sort='relevance',retmax='5',retmode='xml',term=query)
        results = Entrez.read(handle)
        return results

    def fetch_details(self,query):
        fetch = PubMedFetcher(cachedir=self.cachePath)
        response = self.search(query)
        pmids = response["IdList"]
        pubmed_search = []

        for pmid in pmids:
            try:
                article = fetch.article_by_pmid(pmid)
            except Exception as e:
                print(e)
                continue
            title = article.title
            abstract = article.abstract
            url = "https://pubmed.ncbi.nlm.nih.gov/"+pmid+"/"
            pubmed_search.append({
                "title":title,
                "abstract":abstract,
                "url":url
            })
        print(pubmed_search)
        pubmed_documents = []
        for paper in pubmed_search:
            if paper == None or paper["abstract"] == None:
                continue
            pubmed_documents.append(
                Document(
                    text=paper["abstract"],
                    extra_info={
                    "Title of this paper": paper["title"],
                    "URL": paper["url"]
                    }
                )
            )

        return pubmed_documents,pubmed_search






# class PubmedManager(BaseReader):
#     """Pubmed Reader.

#     Gets a search query, return a list of Documents of the top corresponding scientific papers on Pubmed.
#     """

#     def __init__(self):
#         self.embedder = EmbedNodes()

#     def load_data(
#         self,
#         search_query: str,
#         max_results: Optional[int] = 10,
#     ) -> List[Document]:
#         """Search for a topic on Pubmed, fetch the text of the most relevant full-length papers.
#         Args:
#             search_query (str): A topic to search for (e.g. "Alzheimers").
#             max_results (Optional[int]): Maximum number of papers to fetch.
#         Returns:
#             List[Document]: A list of Document objects.
#         """
#         import time
#         import xml.etree.ElementTree as xml

#         import requests

#         pubmed_search = []
#         parameters = {"tool": "tool", "email": "email", "db": "pmc"}
#         parameters["term"] = search_query
#         parameters["retmax"] = max_results
#         resp = requests.get(
#             "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
#             params=parameters,
#         )
#         root = xml.fromstring(resp.content)

#         for elem in root.iter():
#             if elem.tag == "Id":
#                 _id = elem.text
#                 url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?id={_id}&db=pubmed"
#                 print(url)
#                 try:
#                     resp = requests.get(url)
#                     info = xml.fromstring(resp.content)

#                     raw_text = ""
#                     title = ""
#                     journal = ""
#                     for element in info.iter():
#                         if element.tag == "article-title":
#                             title = element.text
#                         elif element.tag == "journal-title":
#                             journal = element.text

#                         if element.text:
#                             raw_text += element.text.strip() + " "

#                     pubmed_search.append(
#                         {
#                             "title": title,
#                             "journal": journal,
#                             "url":f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?id={_id}&db=pubmed",
#                             "text": raw_text,
#                         }
#                     )
#                     time.sleep(1)  # API rate limits
#                 except Exception as e:
#                     print(f"Unable to parse PMC{_id} or it does not exist:", e)
#                     pass

#         # Then get documents from Pubmed text, which includes abstracts
#         pubmed_documents = []
#         for paper in pubmed_search:
#             pubmed_documents.append(
#                 Document(
#                     text=paper["text"],
#                     extra_info={
#                         "Title of this paper": paper["title"],
#                         "Journal it was published in:": paper["journal"],
#                         "URL": paper["url"],
#                     },
#                 )
#             )

#         return pubmed_documents
    
    
#     def process(self,docs):
#         nodes = convert_documents_into_nodes(list(docs.values()))
#         #self.nodes = nodes
#         return nodes

#     async def embed_nodes(self):
#         ds = ray.data.from_items([node for node in self.nodes], parallelism=20) 
#         embedded_nodes = ds.map_batches(
#             self.embedder,  
#             batch_size=1, 
#             # num_cpus=1,
#             num_gpus=1,
#             compute=ActorPoolStrategy(size=1), 
#         )
#         nodes = [node["embedded_nodes"] for node in embedded_nodes.iter_rows()]
#         return nodes
    
#     def load(self,query):
#         currDocs = self.load_data(query,7)
#         nodes = self.process(currDocs)
#         loop = asyncio.get_event_loop()
#         nodes = loop.run_until_complete(self.embed_nodes()) 
#         return nodes


# def search(query):
#     Entrez.email = 'arihantbadjatya@gmail.com'
#     handle = Entrez.esearch(db='pubmed',
#                             sort='relevance',
#                             retmax='3',
#                             retmode='xml',
#                             term=query)
#     results = Entrez.read(handle)
#     return results

# def fetch_details(query):
#     # ids = ','.join(id_list)
#     # Entrez.email = 'arihantbadjatya@gmail.com'
#     # handle = Entrez.efetch(db='pubmed',
#     #                        retmode='xml',
#     #                        id=ids)
#     # results = Entrez.read(handle)
#     fetch = PubMedFetcher()
#     pubmed_search = []

#     # get the  PMID for first 3 articles with keyword sepsis
#     response = search(query)
#     pmids = response["IdList"]
#     # get  articles
#     for pmid in pmids:
#         article = fetch.article_by_pmid(pmid)
#         title = article.title
#         abstract = article.abstract
#         url = "https://pubmed.ncbi.nlm.nih.gov/"+pmid+"/"
#         pubmed_search.append({
#             "title":title,
#             "abstract":abstract,
#             "url":url
#         })

#     print(pubmed_search)
#     pubmed_documents = []
#     for paper in pubmed_search:
#         pubmed_documents.append(
#             Document(
#                 text=paper["abstract"],
#                 extra_info={
#                 "Title of this paper": paper["title"],
#                 "URL": paper["url"]
#                 }
#             )
#         )

#     return pubmed_documents

        

    
#     # for _id in id_list:
#     #     url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?id={_id}&db=pubmed"
#     #     print(url)
#     #     try:
#     #         resp = requests.get(url)
#     #         info = xml.fromstring(resp.content)

#     #         raw_text = ""
#     #         title = ""
#     #         journal = ""
#     #         for element in info.iter():
#     #             if element.tag == "article-title":
#     #                 title = element.text
#     #             elif element.tag == "journal-title":
#     #                 journal = element.text

#     #             if element.text:
#     #                 raw_text += element.text.strip() + " "

#     #         pubmed_search.append(
#     #             {
#     #                 "title": title,
#     #                 "journal": journal,
#     #                 "url":f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?id={_id}&db=pubmed",
#     #                 "text": raw_text,
#     #             }
#     #         )
#     #         time.sleep(1)  # API rate limits
#     #     except Exception as e:
#     #         print(f"Unable to parse PMC{_id} or it does not exist:", e)
#     #         pass

#     #     # Then get documents from Pubmed text, which includes abstracts
        
#     #     print(pubmed_search)

#     #     pubmed_documents = []
#     #     # for paper in pubmed_search:
#     #     #     pubmed_documents.append(
#     #     #         Document(
#     #     #             text=paper["text"],
#     #     #             extra_info={
#     #     #                 "Title of this paper": paper["title"],
#     #     #                 "Journal it was published in:": paper["journal"],
#     #     #                 "URL": paper["url"],
#     #     #             },
#     #     #         )
#     #     #     )

#     # return pubmed_documents





# if __name__ == "__main__":
#     #responseIds = search("What is DNA")
#     #print(responseIds)
#     #print(ids)
#     papers = fetch_details("What is DNA")
#     #print(papers[0])

