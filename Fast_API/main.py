from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from scipy.spatial import distance
import ast
import openai
from transformers import GPT2TokenizerFast
from decouple import config

# Initialize FastAPI app
app = FastAPI()

# Define the GPT-3 model and other parameters
GPT_MODEL = "gpt-3.5-turbo"
api_key = config('API_KEY')  # Replace with your actual OpenAI API key
openai.api_key = api_key

# Load the CSV file with embeddings
embeddings_file_path = '/Users/vivekhanagoji/Documents/BigDataAssignment02/Streamlit_FastAPI/Fast_API/pdf_data.csv'  # Update with the path to your CSV file
df = pd.read_csv(embeddings_file_path)

# Convert the embeddings from string to list
df['Embedding'] = df['Embedding'].apply(ast.literal_eval)


def num_tokens(text):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    encoding = tokenizer.encode(text, add_special_tokens=False)
    return len(encoding)

# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    return 1 - distance.cosine(embedding1, embedding2)

# Define a search function
def strings_ranked_by_relatedness(query, df, relatedness_fn=cosine_similarity, top_n=100):
    query_embedding = generate_text_embeddings(query)  # Implement this function using the OpenAI Text Embedding API
    strings_and_relatednesses = [
        (row['Chunk Text'], relatedness_fn(query_embedding, row['Embedding']))
        for _, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

# Define a function to generate embeddings from text using OpenAI Text Embedding API
def generate_text_embeddings(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(model=model, input=text)
    return response['data'][0]['embedding']

token_budget = 4096 - 500  # Adjust the token budget as needed
# Function to create a query message from the user's question
def query_message(query, df, token_budget):
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below PDFs on the SEC forms to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f'\n\nQuestion: {query}'
    message = introduction

    # Process each section separately
    for string in strings:
        # Split the content into smaller sections, e.g., paragraphs
        sections = string.split('\n\n')  # You can use a more appropriate separator
        
        for section in sections:
            next_section = f'\n\nSection:\n"""\n{section}\n"""'
            if num_tokens(message + next_section + question) > token_budget:
                break
            else:
                message += next_section

    # return message
    return message + question

# Function to answer questions using GPT
def ask(query, df, GPT_MODEL, token_budget, print_message=False):
    message = query_message(query, df, token_budget=token_budget)
    if print_message:
        # print(message)
        messages = [
            {"role": "system", "content": "You answer questions about the SEC pdfs"},
            {"role": "user", "content": message},
        ]
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=0
        )
        response_message = response["choices"][0]["message"]["content"]
        # answer = response_message.split("Section:\n")[0]
    return response_message


# FastAPI route to answer questions
class Question(BaseModel):
    query: str

class Answer(BaseModel):
    answer: str

@app.post("/ask", response_model=Answer)
def get_answer(question: Question):
    response = ask(question.query, df, GPT_MODEL, token_budget=4096 - 500,print_message=True)  # Adjust token budget as needed
    return {"answer": response}
