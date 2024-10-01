# !!! THIS CODE WONT BE RUNNED AS IS REQUIRES GPU !!! #


import logging
import sys
import streamlit as st
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Model names (make sure you have access on HF)
LLAMA2_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"

# Define system prompt and LLM
SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.
"""

query_wrapper_prompt = PromptTemplate(
    "[INST]<>\n" + SYSTEM_PROMPT + "<>\n\n{query_str}[/INST] "
)

# Initialize the LLM
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=LLAMA2_7B_CHAT,
    model_name=LLAMA2_7B_CHAT,
    device_map="auto",  # Use "auto" or specify "cpu"
    model_kwargs={"torch_dtype": torch.float32},  # Change to float32 for CPU
)


# Initialize the embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Set the LLM and embed model in Settings
Settings.llm = llm
Settings.embed_model = embed_model

# Load documents
documents = SimpleDirectoryReader("./doc").load_data()
index = VectorStoreIndex.from_documents(documents)

# Set up Streamlit app
st.title("GyAni!")

# Initialize chat history if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input from the user
if prompt := st.chat_input("Ask anything!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query the LLM using the index
    query_engine = index.as_query_engine()
    response = query_engine.query(prompt)

    # Display the response
    with st.chat_message("GyAnI!"):
        st.markdown(response)
    
    st.session_state.messages.append({"role": "GyAnI!", "content": response})
