from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
import chainlit as cl
from chainlit.types import AskFileResponse
from langchain.chains import RetrievalQAWithSourcesChain
import os

os.environ['OPENAI_API_KEY'] = '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'

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

welcome_message = """Welcome to the Chainlit PDF QA demo! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""


def process_file(file: AskFileResponse):
    import tempfile

    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    # with tempfile.NamedTemporaryFile() as tempfile:
    #     tempfile.write(file) 
    loader = Loader(file.path)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    return docs


def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    # Save data in the user session
    cl.user_session.set("docs", docs)

    # Create a unique namespace for the file

    docsearch = Chroma.from_documents(
        docs, embeddings
    )
    return docsearch


@cl.on_chat_start
async def start():
    # Sending an image with the local file path
    await cl.Message(content="You can now chat with your pdfs.").send()
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    # print(str(file.path))

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # No async implementation in the Pinecone client, fallback to sync
    docsearch = await cl.make_async(get_docsearch)(file)
    
    msg = cl.Message(content=f"clear 1")
    await msg.send()

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(max_tokens_limit=4097),
    )

    # Let the user know that the system is ready
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])

    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    # Get the documents from the user session
    docs = cl.user_session.get("docs")
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = docs[index].page_content
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()

# import os
# from typing import List
# from langchain.document_loaders import PyPDFLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores.pinecone import Pinecone
# from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ChatMessageHistory, ConversationBufferMemory
# from langchain.docstore.document import Document

# import pinecone

# import chainlit as cl
# from chainlit.types import AskFileResponse
# os.environ['PINECONE_API_KEY']='f13b0921-3e0d-4a52-93cd-3a375ed0469a'
# # os.environ["PINECONE_ENV"]=YOUR_PINECONE_ENV
# os.environ['OPENAI_API_KEY']= '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'
# pinecone.init(
#     api_key=os.environ.get("PINECONE_API_KEY"),
#     environment=os.environ.get("PINECONE_ENV"),
# )

# index_name = "langchain-demo"
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# embeddings = OpenAIEmbeddings()

# namespaces = set()

# welcome_message = """Welcome to the Chainlit PDF QA demo! To get started:
# 1. Upload a PDF or text file
# 2. Ask a question about the file
# """


# def process_file(file: AskFileResponse):
#     if file.type == "text/plain":
#         Loader = TextLoader
#     elif file.type == "application/pdf":
#         Loader = PyPDFLoader

#         loader = Loader(file.path)
#         documents = loader.load()
#         docs = text_splitter.split_documents(documents)
#         for i, doc in enumerate(docs):
#             doc.metadata["source"] = f"source_{i}"
#         return docs


# def get_docsearch(file: AskFileResponse):
#     docs = process_file(file)

#     # Save data in the user session
#     cl.user_session.set("docs", docs)

#     # Create a unique namespace for the file
#     namespace = file.id

#     if namespace in namespaces:
#         docsearch = Pinecone.from_existing_index(
#             index_name=index_name, embedding=embeddings, namespace=namespace
#         )
#     else:
#         docsearch = Pinecone.from_documents(
#             docs, embeddings, index_name=index_name, namespace=namespace
#         )
#         namespaces.add(namespace)

#     return docsearch


# @cl.on_chat_start
# async def start():
#     await cl.Avatar(
#         name="Chatbot",
#         url="https://avatars.githubusercontent.com/u/128686189?s=400&u=a1d1553023f8ea0921fba0debbe92a8c5f840dd9&v=4",
#     ).send()
#     files = None
#     while files is None:
#         files = await cl.AskFileMessage(
#             content=welcome_message,
#             accept=["text/plain", "application/pdf"],
#             max_size_mb=20,
#             timeout=180,
#         ).send()

#     file = files[0]

#     msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
#     await msg.send()

#     # No async implementation in the Pinecone client, fallback to sync
#     docsearch = await cl.make_async(get_docsearch)(file)

#     message_history = ChatMessageHistory()

#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         output_key="answer",
#         chat_memory=message_history,
#         return_messages=True,
#     )

#     chain = ConversationalRetrievalChain.from_llm(
#         ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
#         chain_type="stuff",
#         retriever=docsearch.as_retriever(),
#         memory=memory,
#         return_source_documents=True,
#     )

#     # Let the user know that the system is ready
#     msg.content = f"`{file.name}` processed. You can now ask questions!"
#     await msg.update()

#     cl.user_session.set("chain", chain)


# @cl.on_message
# async def main(message: cl.Message):
#     chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
#     cb = cl.AsyncLangchainCallbackHandler()
#     res = await chain.acall(message.content, callbacks=[cb])
#     answer = res["answer"]
#     source_documents = res["source_documents"]  # type: List[Document]

#     text_elements = []  # type: List[cl.Text]

#     if source_documents:
#         for source_idx, source_doc in enumerate(source_documents):
#             source_name = f"source_{source_idx}"
#             # Create the text element referenced in the message
#             text_elements.append(
#                 cl.Text(content=source_doc.page_content, name=source_name)
#             )
#         source_names = [text_el.name for text_el in text_elements]

#         if source_names:
#             answer += f"\nSources: {', '.join(source_names)}"
#         else:
#             answer += "\nNo sources found"

#     await cl.Message(content=answer, elements=text_elements).send()