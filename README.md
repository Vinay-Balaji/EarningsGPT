# EarningsGPT: Earnings Call Analyzer with AI-Powered Insights

## Overview

**EarningsGPT** is an advanced web application designed to help investors and analysts extract meaningful insights from earnings call recordings. The application leverages state-of-the-art speech-to-text transcription, text embedding, and AI-driven question-answering using Retrieval Augmented Generation (RAG) to provide users with a comprehensive analysis of earnings calls. By inputting a YouTube link to an earnings call, users can transcribe the audio, split the transcript into manageable chunks, and ask specific questions to get detailed answers, all within an intuitive web interface.

RAG combines the power of retrieval systems and large language models (LLMs) to handle data not originally included in the LLM's training. This allows EarningsGPT to offer up-to-date insights from the latest earnings calls, ensuring investors have the most relevant information at their fingertips.

## Features

1. **Audio Download and Transcription**
   - Downloads audio from YouTube earnings call videos.
   - Transcribes the audio using AssemblyAI to convert speech into text

2. **Text Chunking and Embedding**
   - Splits the transcribed text into chunks for efficient processing
   - Embeds the text chunks using SentenceTransformer to create vector representations for each chunk.

3. **AI-Powered Question Answering with RAG**
   - Utilizes a FAISS index to retrieve relevant chunks based on user queries.
   - Generates detailed answers to user questions using OpenAI's GPT-3.5-turbo model.

4. **Interactive Web Interface**
   - Provides a user-friendly interface for entering YouTube URLs, analyzing earnings calls, and asking questions.
   - Displays analysis progress with status messages and spinners.

## How It Works
### Indexing Pipeline
The indexing pipeline is responsible for all data pre-processing and for storing the pre-processed data in the so-called “index” of a vector database.

1. **Audio Download**
   - The application takes a YouTube URL as input and uses yt-dlp to download the audio file.
   - The downloaded audio is converted to MP3 format using ffmpeg.

2. **Transcription**
   - The MP3 audio file is uploaded to AssemblyAI for transcription.
   - The resulting transcript is retrieved and stored for further processing.

3. **Text Processing**
   - The transcript is split into chunks of manageable size.
   - Each chunk is embedded into a vector representation using SentenceTransformer.

4. **Indexing**
   - The embedded chunks are indexed using FAISS for efficient similarity search.
  
### Query Pipeline
The query pipeline allows users to ask questions about the earnings calls. It includes two main steps: retrieval and generation.

1. **Retrieval**
   - The retrieval step selects the documents (chunks) to pass on to the language model.
   - A sophisticated retrieval system is used, incorporating both vector and semantic retrieval.
   - Initially, 40 candidate documents are fetched and then reranked using a ranking model to determine the most relevant documents.

2. **Generation**
   - The top 10 documents are used to prompt GPT-3.5 for an answer to the user’s question.
   - GPT-3.5 generates a conversational, human-like response based on the provided documents.

## Setup and Installation
1. Clone the Repository (git clone https://github.com/yourusername/EarningsGPT.git)
2. Create and activiate a virtual environment
      - python -m venv .venv
      - source .venv/bin/activate   # On Windows use `.venv\Scripts\activate`
3. Install the required packages (pip install -r requirements.txt)
4. Replace 'assemblyaiapikey' and 'enter_open_AI_API_Key
5. Run the application (streamlit run app.py) 
