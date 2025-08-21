from importlib.metadata import metadata
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


import streamlit as st
import os
import tempfile
import pandas as pd
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
import json
from langchain.schema import messages_from_dict, messages_to_dict

# import logging
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.document_loaders import PyPDFLoader, CSVLoader,DataFrameLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain, StuffDocumentsChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langsmith import traceable



# Load environment variables from .env file
load_dotenv()

# Access the key
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]="fhgfhfhfhfrytrygff"
os.environ["LANGCHAIN_PROJECT"]="Testbot1"
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load and split PDF
def load_pdf(file_path):
    df = pd.read_csv(file_path)
    df = df.applymap(
        lambda x: x.lower().strip() if isinstance(x, str) else x)
    df["combined"] = df.apply(lambda row: ', '.join(f"{col}: {row[col]}" for col in df.columns), axis=1)
    loader = DataFrameLoader(df, page_content_column="combined")
    # loader = CSVLoader(file_path)
    # loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)
    return docs

# Create vectorstore with Ollama Embeddings
def create_vectorstore(pages, persist_directory):
    embeddings = OllamaEmbeddings(model="all-minilm")
    vectordb = Chroma.from_documents(pages, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb


custom_prompt = PromptTemplate(input_variables=["question", "context","chat_history"],
    template=
    """
# Role and Objective
- You are a friendly, conversational customer service assistant. Your primary goal is to help users purchase a mobile phone that best matches their preferences.
# Instructions
- Interact only in English.
- Use an informal, approachable tone, avoiding robotic or overly formal language.
- Rely exclusively on the Product-list for phone recommendations; do not use outside knowledge.
- Reference chat history to identify and respect user-stated preferences (such as budget, brand, RAM, etc.), and avoid repeating questions already answered.
- Ask concise, clarifying follow-up questions if the user's request lacks necessary information. Only ask one question at a time.
- If the user updates or adds brands, update and append new brand preferences accordingly.
- Suggest phone options that fit all known preferences and are under the user's maximum budget where possible.
- Provide recommendations only after at least two distinct preferences are known.
- If no phones in the Product-list match all preferences, respond with: "No phones match your preferences." and suggest changing or relaxing some filters.
- Do not speculate or use world knowledge beyond what is in the Product-list.
# Workflow Checklist
Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level:
1. Identify and recall user preferences from chat history.
2. Confirm whether at least two preferences are stated.
3. If details are missing, ask a single clarifying question.
4. Filter Product-list for matching phones.
5. Restate user’s goals in your own words (briefly).
6. Recommend matching phones if found; if none, advise adjusting preferences.
# Reasoning and Validation
- Think step by step using provided context to deduce the most relevant options.
- After each recommendation, briefly validate that the suggestions match all known user preferences and maximum budget. If not, self-correct.
# Output Format
- Plain text chat responses in an informal, friendly tone.
- Use markdown (e.g., bulleted lists) when listing phone options.
- Output recommendations only when 2+ preferences are clarified.
# Verbosity and Updates
- Keep responses brief and engaging. Add more detail only when necessary for clarity.
- At each major step (e.g., new recommendations), provide a 1-2 sentence micro-update on progress or any blockers (e.g., no matching phones).
# Stop Conditions
- Respond after each user input.
- Pause or escalate only if no phones match the stated preferences.
# Preambles
- Before providing recommendations, briefly restate the current understanding of the user's goals in your own words.
# Context
Product-list:
{context}
Chat History:
{chat_history}
User:
{question}
Assistant:
"""
)

# Create QA chain using Ollama LLM
def create_conversational_chain(vectordb, model_name="gpt-4o-mini", open_ai_api_key=openai_api_key):
    # llm = ChatOllama(model=model_name)
    llm = ChatOpenAI(model=model_name, max_tokens=2000, temperature=0.7)
    base_retriever = vectordb.as_retriever(search_kwargs={
        "k": 10}, search_type = "mmr")  # base retriever
    # zip_retriver = LLMChainExtractor.from_llm(llm=llm)

    # retriever = ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=zip_retriver)
    question_prompt = PromptTemplate(input_variables=["chat_history", "question"], template="""
    Extract the preferences from chat history and then combine it with the question to create a standalone question.
    chat history:
    {chat_history}
    Question:
    {question}
    Standalone question:
    """)
    question_generator = LLMChain(llm=llm, prompt=question_prompt)
    doc_llm_chain = LLMChain(llm=llm, prompt=custom_prompt)
    doc_chain = StuffDocumentsChain(llm_chain=doc_llm_chain, document_variable_name="context")


    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
    qa_chain = ConversationalRetrievalChain(
        combine_docs_chain= doc_chain,
        question_generator=question_generator,
        retriever=base_retriever,
        memory=memory,
        verbose=True,
        # combine_docs_chain_kwargs={"prompt": custom_prompt},
        rephrase_question=False,
    )
    return qa_chain

# Streamlit UI
st.set_page_config(page_title="📄 Test bot", layout="wide")
st.title("🤖 Mobile shopping assistant")
DB_PATH = "./ollama_pdf_db"
filters = {}
if os.path.exists(DB_PATH ):
    vectordb = Chroma(persist_directory=DB_PATH , embedding_function=OllamaEmbeddings(model="all-minilm"))
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = create_conversational_chain(vectordb)
    st.success("✅ Vector DB loaded. You can start chatting!")

else:
    uploaded_file = st.file_uploader("Upload a CSV", type="csv")

    if uploaded_file and "qa_chain" not in st.session_state:
        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name


            pages = load_pdf(tmp_path)
            vectordb = create_vectorstore(pages, persist_directory="ollama_pdf_db")

            # Store chain in session so it doesn't reprocess
            st.session_state.qa_chain = create_conversational_chain(vectordb)
            st.success("PDF processed! You can ask questions now 👇")

# Question
if "qa_chain" in st.session_state:
    user_question = st.text_input("Shopper assistant:")
    if user_question:
        with st.spinner("Thinking...") and get_openai_callback() as cb:
            answer = st.session_state.qa_chain.invoke({"question": user_question})
            
            st.info(
                f"📊 Token Usage → Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens}, "
                f"Total: {cb.total_tokens}")
            print(answer.keys())
            @traceable
            def print_docs_and_scores(user_question):
                docs_and_scores = vectordb.similarity_search_with_score(user_question, k=10)
                return docs_and_scores
            print_docs_and_scores(user_question)

            st.markdown(f"**Answer:** {answer['answer']}")
            # st.markdown(f"**Answer:** {answer}")

if "qa_chain" in st.session_state:
    st.sidebar.subheader("Chat History")
    for message in st.session_state.qa_chain.memory.chat_memory.messages:
        st.sidebar.markdown(f"**{message.type.capitalize()}:** {message.content}")

if st.sidebar.button("Upload New file"):
    uploaded_file = st.file_uploader("Upload a PDF", type="csv")

    if uploaded_file and "qa_chain" not in st.session_state:
        with st.spinner("Processing CSV..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            pages = load_pdf(tmp_path)
            vectordb = create_vectorstore(pages, persist_directory="ollama_pdf_db")

            # Store chain in session so it doesn't reprocess
            st.session_state.qa_chain = create_conversational_chain(vectordb)
            st.success("CSV processed! You can ask questions now 👇")
    with st.expander("🔍 Chat Memory Debug"):
        # st.write(st.session_state.qa_chain.memory.chat_memory.messages)
        st.write(st.session_state.memory)

if st.button("🔄 Reset Chat"):
    # Clear memory and reassign a new memory object
    new_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.qa_chain.memory = new_memory
    st.success("Chat history reset!")


