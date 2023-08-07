from typing import List, Optional
from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document
from llama_index import VectorStoreIndex
from parser import load_and_parse_json, convert_documents_into_nodes
from ray.data import Dataset, ActorPoolStrategy
import ray
from embed_nodes import EmbedNodes
import asyncio

class PubmedManager(BaseReader):
    """Pubmed Reader.

    Gets a search query, return a list of Documents of the top corresponding scientific papers on Pubmed.
    """

    def __init__(self):
        self.embedder = EmbedNodes()

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
    
    
    def process(self,docs):
        nodes = convert_documents_into_nodes(list(docs.values()))
        #self.nodes = nodes
        return nodes

    async def embed_nodes(self):
        ds = ray.data.from_items([node for node in self.nodes], parallelism=20) 
        embedded_nodes = ds.map_batches(
            self.embedder,  
            batch_size=1, 
            # num_cpus=1,
            num_gpus=1,
            compute=ActorPoolStrategy(size=1), 
        )
        nodes = [node["embedded_nodes"] for node in embedded_nodes.iter_rows()]
        return nodes
    
    def load(self,query):
        currDocs = self.load_data(query,7)
        nodes = self.process(currDocs)
        loop = asyncio.get_event_loop()
        nodes = loop.run_until_complete(self.embed_nodes()) 
        return nodes




if __name__ == "main":
    obj = PubmedManager()
    response = obj.load_data("What is DNA")
    print(response)