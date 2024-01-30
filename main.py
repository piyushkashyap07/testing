import os
from dotenv import load_dotenv
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.response.pprint_utils import pprint_response
import os.path
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage

def setup_openai_api_key():
    # Load environment variables from .env file
    load_dotenv()

    # Check if OPENAI_API_KEY is set in the environment
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if openai_api_key is not None:
        # Set OPENAI_API_KEY in the environment
        os.environ['OPENAI_API_KEY'] = openai_api_key
        print("OPENAI_API_KEY set successfully.")
    else:
        print("Error: OPENAI_API_KEY environment variable is not set.")


def create_index_and_query_engine(directory_path):
    
    try:
        # Create a SimpleDirectoryReader to load documents from the specified directory
        reader = SimpleDirectoryReader(directory_path)

        # Load data/documents from the directory
        documents = reader.load_data()
        print("Documents loaded successfully.")

        # Create a VectorStoreIndex from the loaded documents
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        print("VectorStoreIndex created successfully.")

        # Create a QueryEngine from the VectorStoreIndex
        query_engine = index.as_query_engine()
        print("QueryEngine created successfully.")

        return index, query_engine

    except Exception as e:
        print(f"Error creating index and query engine: {e}")
        return None, None


def create_query_engine_and_query(query_text, index, similarity_top_k=4, similarity_cutoff=0.80):
    try:
        # Create a VectorIndexRetriever with specified parameters
        retriever = VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)

        # Create a SimilarityPostprocessor with specified similarity cutoff
        postprocessor = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)

        # Create a RetrieverQueryEngine with the retriever and postprocessor
        query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[postprocessor])

        # Query the engine with the provided text
        response = query_engine.query(query_text)

        # Print the response using pprint_response
        pprint_response(response, show_source=True)

        return response

    except Exception as e:
        print(f"Error creating query engine or querying: {e}")
        return None
    

def create_or_load_index_and_query(query_text, directory_path, persist_dir="./storage"):
    try:
        if not os.path.exists(persist_dir):
            # Load the documents and create the index
            documents = SimpleDirectoryReader(directory_path).load_data()
            index = VectorStoreIndex.from_documents(documents)

            # Store the index for later
            index.storage_context.persist(persist_dir=persist_dir)
            print("Index created and stored successfully.")
        else:
            # Load the existing index
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            print("Index loaded from storage.")

        # Query the index
        query_engine = index.as_query_engine()
        response = query_engine.query(query_text)

        # Print the response
        print(response)

        return response

    except Exception as e:
        print(f"Error creating or loading index and querying: {e}")
        return None

if __name__ == "__main__":
    # Call the function to set up OpenAI API key
    setup_openai_api_key()
    # Specify the directory path
    pdf_directory_path = r"C:\Piyush\Scripts\generative-ai-projects\LLMS\RAG-LLm-pro-1\pdf"

    # Call the function to create index and query engine
    created_index, created_query_engine = create_index_and_query_engine(pdf_directory_path)

    # Specify the directory path and query text
    pdf_directory_path = r"C:\Piyush\Scripts\generative-ai-projects\LLMS\RAG-LLm-pro-1\pdf"
    query_text = "What are Nonprobability Sampling?"

    # Call the function to create query engine and query
    created_response = create_query_engine_and_query("What is Training Data ?", created_index)
    # Call the function to create or load index and query
    created_or_loaded_response = create_or_load_index_and_query(query_text, pdf_directory_path)