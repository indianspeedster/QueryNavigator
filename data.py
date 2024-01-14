from qdrant_client import QdrantClient
from llama_index.vector_stores import QdrantVectorStore
from llama_index import ServiceContext
from llama_index.node_parser import get_leaf_nodes
from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex, StorageContext
from llama_index.schema import Document
from llama_index.llms import OpenAI
from llama_index.node_parser import HierarchicalNodeParser
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

client = QdrantClient("http://localhost:6333")

vector_store = QdrantVectorStore(
    collection_name="db_2",
    client=client,
)

# ref: https://docs.llamaindex.ai/en/stable/examples/data_connectors/simple_directory_reader.html
# Load all (top-level) files from directory
documents = SimpleDirectoryReader(
    input_dir="docs",   
    # input_files = ["invading_the_sacred.pdf"]
).load_data()

document = Document(text = "\n\n".join([doc.text for doc in documents]))

node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes = [2048, 512, 128]
)

LLM = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

service_context = ServiceContext.from_defaults(
    embed_model="local:BAAI/bge-large-en-v1.5",
    node_parser=node_parser,
)

auto_merging_context = ServiceContext.from_defaults(
    llm = LLM,
    embed_model="local:BAAI/bge-small-en-v1.5",
    node_parser=node_parser,
)

nodes = node_parser.get_nodes_from_documents([document])
leaf_nodes = get_leaf_nodes(nodes)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
storage_context.docstore.add_documents(nodes)

automerging_index = VectorStoreIndex(
    leaf_nodes, storage_context = storage_context, service_context = auto_merging_context
)


