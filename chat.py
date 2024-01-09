import os
import openai
from qdrant_client import QdrantClient
from llama_index.vector_stores import QdrantVectorStore
from llama_index import SimpleDirectoryReader
from llama_index import Document
from llama_index.node_parser import get_leaf_nodes
from llama_index.node_parser import HierarchicalNodeParser
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, StorageContext
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.retrievers import AutoMergingRetriever
from llama_index.query_engine import RetrieverQueryEngine

class CodeSpaceHandler:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = QdrantClient("http://localhost:6333")
        self.vector_store = QdrantVectorStore(
            collection_name="db_2",
            client=self.client,
        )
        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128]
        )
        self.LLM = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
        self.auto_merging_context = ServiceContext.from_defaults(
            llm=self.LLM,
            embed_model="local:BAAI/bge-small-en-v1.5",
            node_parser=self.node_parser,
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.automerging_index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=self.storage_context,
            service_context=self.auto_merging_context,
        )
        self.automerging_retriever = self.automerging_index.as_retriever(
            similarity_top_k=12
        )
        self.retriever = AutoMergingRetriever(
            self.automerging_retriever,
            self.automerging_index.storage_context,
            verbose=True,
        )
        self.rerank = SentenceTransformerRerank(top_n=6, model="BAAI/bge-reranker-base")
        self.auto_merging_engine = RetrieverQueryEngine.from_args(
            self.automerging_retriever, node_postprocessors=[self.rerank]
        )

    def query_auto_merging(self, query):
        response = self.auto_merging_engine.query(query)
        return response.response
