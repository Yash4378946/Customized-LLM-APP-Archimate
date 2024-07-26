import gradio as gr
from huggingface_hub import InferenceClient
from typing import List, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss
import os

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

class MyApp:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_pdf("Architecture_Design_Basics.pdf")
        self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such file: '{file_path}'")
        doc = fitz.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        print("PDF processed successfully!")

    def build_vector_db(self) -> None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = model.encode([doc["content"] for doc in self.documents])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        results = [self.documents[i]["content"] for i in I[0]]
        return results if results else ["No relevant documents found."]

app = MyApp()

def respond(
    message: str,
    history: List[Tuple[str, str]],
    system_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    system_message = ("You are an expert in architecture. You provide accurate and concise information about different architectural styles, techniques, and famous buildings.")
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    retrieved_docs = app.search_documents(message)
    context = "\n".join(retrieved_docs)
    messages.append({"role": "system", "content": "Relevant documents: " + context})

    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

demo = gr.Blocks()

with demo:
    gr.Markdown("🏛️ **Architecture Mate**")
    gr.Markdown(
        "‼️Disclaimer: This chatbot is based on a publicly available architecture reference book. "
        "We are not professional architects, and the use of this chatbot is at your own risk. For professional advice, please consult a qualified architect.‼️"
    )
    
    chatbot = gr.ChatInterface(
        respond,
        examples=[
            ["Can you provide information on Gothic architecture?"],
            ["What are the key features of Modernist architecture?"],
            ["Tell me about famous Baroque buildings."],
            ["How did the Renaissance influence architecture?"],
            ["What are some notable examples of Brutalist architecture?"],
            ["Can you explain the principles of Sustainable architecture?"],
            ["What are the characteristics of Art Deco architecture?"],
            ["How does Contemporary architecture differ from other styles?"],
            ["What are the basics of designing a sustainable building?"],
            ["How can I incorporate natural light into my building design?"],
            ["What are the key principles of minimalist architecture?"],
            ["Can you provide tips for designing energy-efficient homes?"],
            ["What materials are commonly used in modern architecture?"],
            ["How can I create an open floor plan in my home design?"]
        ],
        title='Architecture Mate 🏛️'
    )

if __name__ == "__main__":
    demo.launch()

