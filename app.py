import streamlit as st
import re
import os
import shutil
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

# ---------------------------------------------------------
# Page Configuration & Setup
# ---------------------------------------------------------
st.set_page_config(page_title="YouTube Video Chatbot", page_icon="🎥", layout="centered")
st.title("YouTube Video Chatbot 🎥")

# ---------------------------------------------------------
# Core Functions from youtube_video_chatbot.py
# ---------------------------------------------------------
def get_video_id(url):
    """Extract YouTube video ID from URL."""
    patterns = [
        r"(?:v=)([\w-]{11})",
        r"(?:youtu\.be/)([\w-]{11})",
        r"(?:shorts/)([\w-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    if re.fullmatch(r"[\w-]{11}", url):
        return url
    raise ValueError("Invalid YouTube URL")

def get_youtube_transcript(url):
    """Extract YouTube transcript and return it as a single string."""
    video_id = get_video_id(url)
    api = YouTubeTranscriptApi()
    transcripts = list(api.list(video_id))
    
    if not transcripts:
        raise Exception("No transcript available for this video.")

    preferred_languages = ["en", "hi", "kn"]
    selected = None

    for language in preferred_languages:
        for transcript in transcripts:
            if transcript.language_code == language:
                selected = transcript
                break
        if selected:
            break

    if selected is None:
        selected = transcripts[0]

    transcript = selected.fetch()
    result = []
    
    for item in transcript:
        start = int(item.start)
        minutes = start // 60
        seconds = start % 60
        timestamp = f"[{minutes:02d}:{seconds:02d}]"
        result.append(f"{timestamp} {item.text}")

    return "\n".join(result)

@st.cache_resource
def process_video_and_setup_retriever(url):
    """Fetches transcript, chunks it, and sets up the vector store in-memory."""
    text = get_youtube_transcript(url)
    docs = [Document(page_content=text)]
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = SemanticChunker(embeddings=embeddings)
    chunks = text_splitter.split_documents(docs)

    # REMOVED: os and shutil folder deletion logic
    # Create the Chroma vectorstore in-memory by dropping persist_directory
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )
    return retriever

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "OpenRouter API Key", 
        type="password", 
        value="" # Default from original script
    )
    video_url = st.text_input("YouTube URL")
    process_btn = st.button("Process Video")

# ---------------------------------------------------------
# State Management
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if process_btn:
    if video_url:
        with st.spinner("Extracting transcript and building knowledge base..."):
            try:
                st.session_state.retriever = process_video_and_setup_retriever(video_url)
                st.session_state.messages = [] # Clear history on new video
                st.success("Video processed successfully! You can now chat.")
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
    else:
        st.warning("Please enter a valid YouTube URL.")

# ---------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------
# Render existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the video..."):
    if "retriever" not in st.session_state:
        st.warning("⚠️ Please process a YouTube video in the sidebar first.")
    elif not api_key:
        st.warning("⚠️ Please provide an OpenRouter API key.")
    else:
        # Add user prompt to state and UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            try:
                retriever = st.session_state.retriever
                docs = retriever.invoke(prompt)
                context = "\n\n".join(doc.page_content for doc in docs)
                
                system_prompt = f"""
                You are a knowledgeable and friendly AI assistant. Answer the user's questions using only the relevant context or history from the provided video transcript.
                
                Context:
                {context}
                
                If the answer is not found in or related to the context, politely say so. Do not make up information.
                """
                
                history_messages = [{"role": h["role"], "content": h["content"]} for h in st.session_state.messages[:-1]]
                final_messages = [{"role": "system", "content": system_prompt}] + history_messages + [{"role": "user", "content": prompt}]
                
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
                
                with st.spinner("Thinking..."):
                    response = client.chat.completions.create(
                        model="openrouter/free",
                        messages=final_messages
                    )
                
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"An error occurred during generation: {str(e)}")