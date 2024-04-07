from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os

app = Flask(__name__)
CORS(app, resources={r"/query": {"origins": "https://soft-belekoy-df3409.netlify.app/"}})

# Set up Hugging Face token
HF_token = "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"
#os.environ['HuGGINGFACEHUB_API_TOKEN'] = HF_token

# Load data from the provided URL
URL = "https://docs.google.com/spreadsheets/d/1CCiYSa2ZReP4gcAIbKflING7eVmwEXAx/edit?usp=sharing&ouid=114895549194268657828&rtpof=true&sd=true"
data = WebBaseLoader(URL)
content = data.load()

# Split the content into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
chunking = text_splitter.split_documents(content)

# Get embeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_token,
    model_name="pinecone/bert-retriever-squad2"
)

# Create a vector store
vectorstore = Chroma.from_documents(chunking, embeddings)

# Set up the Hugging Face model
model = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1",
                        model_kwargs={"temperature": 0.5, "max_new_tokens": 512, "max_length": 64},
                        huggingfacehub_api_token="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")

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 0})

qa = RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type="stuff")


@app.route('/query', methods=['POST'])
def process_query():
    try:
        print("Request received")
        # Get the query from the request data
        data = request.get_json()
        query = data.get('randomQuery')
        print(query)
        
        response = qa.invoke(query)
        output = response['result']
        start_index = output.find("Helpful Answer:") + len("Helpful Answer:")
        helpful_answer = output[start_index:].strip()

        
        #return email_content
        return helpful_answer
        print("Helpful Answer", helpful_answer)
        #return jsonify({'helpful_answer': helpful_answer})

    except Exception as e:
        # Handle exceptions gracefully
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True,port=5001)
