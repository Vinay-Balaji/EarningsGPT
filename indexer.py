# Import necessary libraries
from yt_dlp import YoutubeDL
import os
import requests
import time
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Set API keys and paths
ASSEMBLYAI_API_KEY = "assemblyaiapikey"  # AssemblyAI API key
FFMPEG_PATH = "C:\\Users\\vinay\\Desktop\\ffmpeg-2024-08-01-git-bcf08c1171-essentials_build\\bin\\ffmpeg.exe"  # Path to ffmpeg executable

# Function to download audio from YouTube
def download_audio(url, output_path):
    # Set options for YoutubeDL
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'ffmpeg_location': FFMPEG_PATH,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    # Download audio using YoutubeDL
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Extract and format file name
    info_dict = ydl.extract_info(url, download=False)
    filename = ydl.prepare_filename(info_dict).replace('.webm', '.mp3').replace('.m4a', '.mp3')
    return filename

# Function to transcribe audio using AssemblyAI
def transcribe_audio(audio_path):
    headers = {'authorization': ASSEMBLYAI_API_KEY}
    upload_url = 'https://api.assemblyai.com/v2/upload'
    transcribe_url = 'https://api.assemblyai.com/v2/transcript'

    # Function to read audio file in chunks
    def read_file(filename, chunk_size=5242880):
        with open(filename, 'rb') as _file:
            while True:
                data = _file.read(chunk_size)
                if not data:
                    break
                yield data

    # Upload audio file to AssemblyAI
    response = requests.post(upload_url, headers=headers, data=read_file(audio_path))
    audio_url = response.json()['upload_url']

    # Request transcription from AssemblyAI
    response = requests.post(transcribe_url, headers=headers, json={"audio_url": audio_url})
    transcript_id = response.json()['id']

    # Poll for transcription completion
    while True:
        response = requests.get(f"{transcribe_url}/{transcript_id}", headers=headers)
        if response.json()['status'] == 'completed':
            return response.json()['text']
        elif response.json()['status'] == 'failed':
            raise Exception("Transcription failed")
        time.sleep(5)

# Function to split transcript into chunks
def split_transcript(transcript, max_chunk_size=512):
    words = transcript.split()  # Split transcript into words
    chunks = []
    for i in range(0, len(words), max_chunk_size):
        chunks.append(" ".join(words[i:i + max_chunk_size]))  # Create chunks of specified size
    return chunks

# Function to embed documents using a sentence transformer model
def embed_documents(documents, model):
    print("Embedding documents...")
    embeddings = model.encode(documents)  # Generate embeddings for documents
    return np.array(embeddings, dtype=np.float32)  # Return embeddings as NumPy array

# Function to create a FAISS index from embeddings
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Get dimension of embeddings
    index = faiss.IndexFlatL2(dimension)  # Create FAISS index with L2 distance
    index.add(embeddings)  # Add embeddings to index
    return index

# Function to save FAISS index to a file
def save_index(index, file_path):
    faiss.write_index(index, file_path)  # Save index to file

# Function to load FAISS index from a file
def load_index(file_path):
    return faiss.read_index(file_path)  # Load index from file
