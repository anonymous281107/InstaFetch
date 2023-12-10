from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import json
from django.conf import settings
from langchain.chat_models import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field, validator
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import JSONLoader
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
from uuid import uuid4
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma

import chromadb


tiktoken.encoding_for_model('gpt-3.5-turbo')

# Initialize tokernizer
tokenizer = tiktoken.get_encoding('cl100k_base')

# Creating a local index for vector storage
index_name = 'autodetail'


def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


# Initiate Chunking modules
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=30,
    length_function=tiktoken_len
)

# Initialize openAI embedder to encode user queries
embeddings = OpenAIEmbeddings(
    model='text-embedding-ada-002', openai_api_key=settings.OPENAI_API_KEY)


client = chromadb.HttpClient(host='localhost', port=8000)
print("Chroma DB ", client.heartbeat())


@api_view(["POST"])
# This api returns the Description, Technical Specification or User Review section
def getSectionData(request, *args, **kwrgs):
    section = request.data['section']
    product = request.data['product']
    data = json.load(open('data/'+product+'.json'))
    items = []
    for category in data['sections']:
        if category == section:
            for item in category["items"]:
                items.append(item)
    newData = {"items": items}
    return Response(newData)


# This section handles user query
@api_view(["POST"])
def handleUserQuery(request, *args, **kwrgs):
    query = request.data['query']
    product = request.data['product']
    productCollection = client.get_collection(name=product)
    llm = ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )
    print("Items on Collection ", productCollection.peek())
    print("Number of items : ", productCollection.count())
    db = Chroma(
        client=client,
        collection_name=product,
        embedding_function=embeddings,
    )
    query = query
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr")
    )
    prompt = """
        Use the information available to construct a valid response to the user query  : If no valid response is available return "Sorry the answer to that question is not available"

        Example user query with solution : 
        {
            "query" : "Does this TV model support Dolby Digital Plus audio, and what accessibility features does it offer?"
            "response" : "Yes, the TV model supports Dolby Digital Plus audio and offers various accessibility features including VoiceView, Screen Magnifier, Text Banner, closed captioning, Audio Description, and Bluetooth support for select hearing aids and devices."
            "reasoningSteps" : "Step 1 Identify Dolby Digital Plus Support : (Description Section) The TV model description mentions "Audio support: Dolby Digital Plus with passthrough of Dolby-encoded audio." Step 2 : Confirm Accessibility Features: (Technical Specification) The provided information states accessibility features like VoiceView, a screen reader for visually impaired users, a Screen Magnifier, Text Banner, closed captioning for videos, and Audio Description for verbal descriptions of on-screen activities, along with support for select compatible Bluetooth hearing aids and devices."
        }
        """
    result = qa.run(query + prompt)
    return Response(json.loads(result))

# This API encodes and caches HTML data obtained from extrator module and caches it


@api_view(["POST"])
def embeddHTMLData(request, *args, **kwrgs):
    product = request.data["product"]
    data = json.load(open('data/'+product+'.json'))
    print("Inserting Data to Chroma")
    texts = []
    metadatas = []
    metadata = {
        'product': data['product']
    }
    record_texts = []
    for cat in data['sections']:
        record_texts.append(json.dumps(cat))

    record_metadatas = [{
        "chunk": j, "text": text, **metadata
    } for j, text in enumerate(record_texts)]
    texts.extend(record_texts)
    metadatas.extend(record_metadatas)
    productCollection = client.create_collection(
        name=data['product'], embedding_function=embeddings)
    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embeddings.embed_documents(texts)
        productCollection.add(
            documents=texts,
            embeddings=embeds,
            ids=ids
        )

    return Response("Chroma Upload Complete")
