# Chatbot using RAG and Vector Databases with OPENAI API

This is a Flask application that serves as a Chatbot built on top of RAG (Retrieval-Augmented Generation) and Vector Databases, utilizing the OPENAI API. The Chatbot is designed to answer queries using information stored in a knowledge base derived from a PDF file.

## Prerequisites

Before running the application, ensure you have the following prerequisites installed and configured:

1. **Docker**: You need to run a Docker container for the QDRANT Vector Database. Use the following command to start the Docker container:

```
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```


2. **Python and pip**: Make sure you have Python 3.x installed along with pip for package management.

3. **Required Python Packages**: Install the necessary Python packages by running the following command:

```
pip install -r requirements.txt

```


4. **PDF Knowledge Base**: You will need a PDF file that contains the knowledge base information. This file will be used by the Chatbot to answer queries.

5. **OpenAI API Key**: Set the `OPENAI_API_KEY` environment variable to your OpenAI API key:

```
export OPENAI_API_KEY=your_api_key_here
```

## Getting Started

Follow these steps to run the Flask application:

1. Start the QDRANT Vector Database Docker container (if not already running):

```
docker start qdrant
```

2. Install the required Python packages as mentioned above.

3. Add the PDF knowledge base file (e.g., `knowledge_base.pdf`) to the appropriate directory.

4. Configure the application to use the PDF file as the knowledge base. You can set the file path in the application configuration.

5. Run the Flask application:

```
python app.py
```


The Flask application should now be running and accessible at `http://localhost:5000`.

## Usage

You can access the Chatbot interface by visiting `http://localhost:5000` in your web browser. Enter your queries, and the Chatbot will utilize the RAG model and the knowledge base from the PDF file to provide responses.




