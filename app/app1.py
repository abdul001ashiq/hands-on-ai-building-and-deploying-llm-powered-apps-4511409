# Chroma compatibility issue resolution
# https://docs.trychroma.com/troubleshooting#sqlite
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from tempfile import NamedTemporaryFile
import logging

import chainlit as cl
from chainlit.types import AskFileResponse

import chromadb
from chromadb.config import Settings
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain.chains.base import Chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStore
from prompt import EXAMPLE_PROMPT, PROMPT, WELCOME_MESSAGE
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_file(*, file: AskFileResponse) -> list:
    if file.type != "application/pdf":
        raise TypeError("Only PDF files are supported")
    
    try:
        with NamedTemporaryFile() as tempfile:
            tempfile.write(file.content)

            loader = PDFPlumberLoader(tempfile.name)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Reduced chunk size for better retrieval
                chunk_overlap=200  # Increased overlap for better context
            )

            docs = text_splitter.split_documents(documents)

            for i, doc in enumerate(docs):
                doc.metadata["source"] = f"source_{i}"

            if not docs:
                raise ValueError("PDF file parsing failed.")

            return docs
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise

def create_search_engine(*, file: AskFileResponse) -> VectorStore:
    try:
        # Process and save data in the user session
        docs = process_file(file=file)
        cl.user_session.set("docs", docs)
        
        encoder = OpenAIEmbeddings(
            model="text-embedding-ada-002"
        )
        
        # Initialize Chromadb client and settings
        client = chromadb.EphemeralClient()
        client_settings = Settings(
            allow_reset=True,
            anonymized_telemetry=False
        )

        search_engine = Chroma.from_documents(
            client=client,
            documents=docs,
            embedding=encoder,
            client_settings=client_settings 
        )

        return search_engine
    except Exception as e:
        logger.error(f"Error creating search engine: {str(e)}")
        raise 

@cl.on_chat_start
async def start():
    try:
        files = None
        while files is None:
            files = await cl.AskFileMessage(
                content=WELCOME_MESSAGE,
                accept=["application/pdf"],
                max_size_mb=20,
            ).send()
      
        file = files[0]
        msg = cl.Message(content=f"Processing `{file.name}`...")
        await msg.send()

        search_engine = await cl.make_async(create_search_engine)(file=file)

        llm = ChatOpenAI(
            model='gpt-3.5-turbo-16k-0613',
            temperature=0,
            streaming=True
        )

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=search_engine.as_retriever(max_tokens_limit=4097),
            chain_type_kwargs={
                "prompt": PROMPT,
                "document_prompt": EXAMPLE_PROMPT
            },
        )

        msg.content = f"`{file.name}` processed. You can now ask questions!"
        await msg.update()

        cl.user_session.set("chain", chain)
    except Exception as e:
        logger.error(f"Error in chat start: {str(e)}")
        await cl.Message(content=f"An error occurred: {str(e)}").send()
        return

@cl.on_message
async def main(message: cl.Message):
    try:
        chain = cl.user_session.get("chain")
        if not chain:
            await cl.Message(content="Please upload a PDF file first.").send()
            return

        cb = cl.AsyncLangchainCallbackHandler()
        response = await chain.acall(message.content, callbacks=[cb])
        answer = response["answer"]
        sources = response["sources"].strip()
        source_elements = []

        # Get the documents from the user session
        docs = cl.user_session.get("docs")
        if not docs:
            await cl.Message(content="No documents found in session. Please upload a PDF file.").send()
            return

        metadatas = [doc.metadata for doc in docs]
        all_sources = [m["source"] for m in metadatas]

        # Adding sources to the answer
        if sources:
            found_sources = []

            # Add the sources to the message
            for source in sources.split(","):
                source_name = source.strip().replace(".", "")
                try:
                    index = all_sources.index(source_name)
                    text = docs[index].page_content
                    found_sources.append(source_name)
                    # Create the text element referenced in the message
                    source_elements.append(cl.Text(content=text, name=source_name))
                except ValueError:
                    logger.warning(f"Source {source_name} not found in documents")
                    continue

            if found_sources:
                answer += f"\nSources: {', '.join(found_sources)}"
            else:
                answer += "\nNo sources found"

        await cl.Message(content=answer, elements=source_elements).send()
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        await cl.Message(content=f"An error occurred while processing your message: {str(e)}").send() 