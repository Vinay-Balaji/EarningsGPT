# Import necessary libraries
from sentence_transformers import SentenceTransformer
import openai

# Function to retrieve documents based on query
def retrieve_documents(query, index, documents, model, top_k=40):
    query_embedding = model.encode([query])  # Encode the query
    distances, indices = index.search(query_embedding, top_k)  # Search for top_k closest documents

    # Ensure indices are within the range of available documents
    max_index = len(documents) - 1
    valid_indices = [i for i in indices[0] if i <= max_index]

    retrieved_docs = [documents[i] for i in valid_indices]  # Retrieve documents based on valid indices
    return retrieved_docs

# Function to generate a response based on retrieved documents and question
def generate_response(retrieved_docs, question, api_key):
    openai.api_key = api_key  # Set OpenAI API key
    prompt = "Answer the question based on the following documents:\n\n"
    prompt += "\n\n".join(retrieved_docs[:10])  # Top 10 docs for prompt
    prompt += f"\n\nQuestion: {question}\nAnswer:"

    # Generate response using OpenAI's ChatCompletion
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response.choices[0].message['content'].strip()  # Return the generated response
