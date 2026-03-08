import re
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests
import sys

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def clean_text(text):
    
    text = re.sub(r'\s+',' ',text)
    return text.strip()

def process_pdf(pdf_path,chunk_size = 150,overlap = 30):
    
    doc = fitz.open(pdf_path)
    all_chunks = []
    
    for page_num in range(len(doc)):
        
        page= doc.load_page(page_num)
        text = page.get_text("text")
        text = clean_text(text)
        
        if not text:
            continue
        
        word = text.split()
        step = chunk_size - overlap
        
        for i in range(0,len(word),step):
            chunk_words = word[i : i+ chunk_size]
            chunk_string= " ".join(chunk_words)
            
            all_chunks.append({
                "text": chunk_string,
                "page_num": page_num + 1,
                "source": pdf_path
                })
        
            
    return all_chunks


def generate_embedding(chunks_list):
    
    if len(chunks_list) == 0:
        print("\nERROR:No text could be read from the PDF.!")
        sys.exit()
    
    print(f"A total of {len(chunks_list)} chunks are processed.")
    
    text_to_encode= [chunk["text"] for chunk in chunks_list]
    
    embedding = embedding_model.encode(text_to_encode,convert_to_numpy=True)
    
    embeddings = np.array(embedding, dtype=np.float32)
    
    if len(embeddings.shape) == 1:
        embeddings = np.expand_dims(embeddings, axis=0)
    
    faiss.normalize_L2(embeddings)

    return chunks_list,embeddings

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)
    
    return index

def search_in_index(query,index,chunks_list,k =5):
    
    query_embedding = embedding_model.encode([query],convert_to_numpy= True)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    
    faiss.normalize_L2(query_embedding)
    
    distances, indices = index.search(query_embedding, k)
    
    for i, (idx,score) in enumerate(zip(indices[0],distances[0])):
        print(f"   -> {i+1}. Chunk Score: {score:.4f} (Page: {chunks_list[idx]['page_num']})")

    results = []
    
    for idx in indices[0]:
        results.append(chunks_list[idx])
    
    return results

def generated_answer(query,retrieved_chunks):
    context_text = ""
    for i,chunk in enumerate(retrieved_chunks):
        context_text += f"\n--- Source {i+1} (Page: {chunk['page_num']}) ---\n"
        context_text += f"{chunk['text']}\n"
    
    system_prompt = """
        You are a strict academic assistant.

        You must ONLY answer using the provided context.

        Rules:
        - If the answer is not in the context say:
          "The document does not contain this information."
        - Always cite the page number like: (Page 3)
        - Do not use external knowledge.
        - Answer in English.
        """
    user_prompt = f"""
        Context:{context_text}

        User Question:{query}
        """
    
    url = "http://127.0.0.1:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    data = {
        "model": "phi3", 
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,  
        "max_tokens": 1500    
    }
    
    print("Waiting for a answer...\n")
    try:
        response = requests.post(url, headers=headers, json=data,timeout=120)
        
        response.raise_for_status()
        
        results = response.json()
        
        answer = results["choices"][0]["message"]["content"]
        return answer
    except requests.exceptions.RequestException as e:
         return f"Unable to access the API: {e}?"
        
def main():
    print("Starting RAG System")
    
    pdf_path = r"data/sample.pdf"
    
    try:
        print(f"\n(1/4) Reading and chunking PDF: {pdf_path}")
        chunks_list = process_pdf(pdf_path, chunk_size=150, overlap=30)
        
        print("\n(2/4) Converting text chunks into embeddings")
        chunks_list, embeddings = generate_embedding(chunks_list)
        
        print("\n(3/4) Creating FAISS index on CPU")
        index = create_faiss_index(embeddings)
        
        print("PDF loaded. Waiting for your questions.")
        print("(Type 'q' or 'quit' to exit)")
        
    except FileNotFoundError:
        print(f"\nError! A file named '{pdf_path}' could not be found.")
        sys.exit()
    
    while(True):
        user_querry = input("\n;You:")
        
        if user_querry.lower() in ["q","quit"]:
            break
        
        if not user_querry.strip():
            continue
        
        retrieved_chunks = search_in_index(user_querry, index, chunks_list,k=5)
        
        answer = generated_answer(user_querry, retrieved_chunks)
        
        print(f"\nPhi-3: {answer}")
        
if __name__ == "__main__":
    main()









