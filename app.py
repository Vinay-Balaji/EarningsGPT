# Import necessary libraries
import streamlit as st
import os
from indexer import download_audio, transcribe_audio, split_transcript, embed_documents, save_index, load_index, create_faiss_index
from query import retrieve_documents, generate_response
from sentence_transformers import SentenceTransformer
import numpy as np

# Define paths for audio and index
AUDIO_PATH = "audio"  # Directory to save downloaded audio files
INDEX_PATH = "faiss_index.idx"  # File path to save/load FAISS index

# Ensure directories exist
os.makedirs(AUDIO_PATH, exist_ok=True)  # Create audio directory if it doesn't exist

embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # Load sentence transformer model

# Load FAISS index if it exists, otherwise initialize as None
if os.path.exists(INDEX_PATH):
    faiss_index = load_index(INDEX_PATH)
else:
    faiss_index = None  # Initialize faiss_index as None

if 'chunks' not in st.session_state:
    st.session_state.chunks = None  # Initialize chunks as None in session state

st.title("Earnings Call Analyzer")  # Set title of the app

url = st.text_input("Enter YouTube URL")  # Text input for YouTube URL

# Button to analyze the earnings call
if st.button("Analyze"):
    with st.spinner("Downloading audio..."):
        audio_file = download_audio(url, AUDIO_PATH)  # Download audio from YouTube URL

    with st.spinner("Transcribing audio..."):
        transcript = transcribe_audio(audio_file)  # Transcribe downloaded audio
        st.session_state.chunks = split_transcript(transcript)  # Split transcript into chunks and store in session state

    with st.spinner("Embedding documents..."):
        embeddings = embed_documents(st.session_state.chunks, embed_model)  # Embed transcript chunks
        if faiss_index is None:
            faiss_index = create_faiss_index(embeddings)  # Create new FAISS index if it doesn't exist
        else:
            faiss_index.add(np.array(embeddings))  # Add embeddings to existing FAISS index
        save_index(faiss_index, INDEX_PATH)  # Save FAISS index to file

    st.success("Analysis complete! You can now ask questions.")  # Show success message

# Input field for question about the earnings call
question = st.text_input("Ask a question about the earnings call")  # Text input for user's question
openai_api_key = "enter_open_AI_API_KEY"  # Enter your OpenAI API key here

# Button to get an answer to the user's question
if st.button("Get Answer"):
    if st.session_state.chunks is not None and faiss_index is not None:  # Check if chunks and faiss_index are defined
        with st.spinner("Retrieving documents..."):
            retrieved_docs = retrieve_documents(question, faiss_index, st.session_state.chunks, embed_model)  # Retrieve relevant documents

        with st.spinner("Generating response..."):
            answer = generate_response(retrieved_docs, question, openai_api_key)  # Generate response using OpenAI API
            st.write(answer)  # Display the answer
    else:
        st.error("Please analyze an earnings call first.")  # Show error message if analysis hasn't been done
