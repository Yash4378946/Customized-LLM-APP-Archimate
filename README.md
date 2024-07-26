# Architecture Mate Chatbot

## Overview

The "Architecture Mate" chatbot is designed to assist users with architectural design queries, providing guidance on various aspects of architecture, including design principles, materials, styles, and construction techniques. It aims to be a helpful companion for architects, students, and enthusiasts.

## Features

- **Design Principles**: Offers information on key architectural design principles.
- **Material Guidance**: Provides insights on different construction materials and their uses.
- **Style Exploration**: Helps users explore various architectural styles and their characteristics.
- **Construction Techniques**: Explains common construction techniques and best practices.
- **User-Friendly Interface**: Intuitive chatbot interface built with Gradio.

## Installation

1. **Clone the repository**:
    ```sh
    git clone <repo-url>
    cd <repo-directory>
    ```

2. **Create and activate a virtual environment**:
    ```sh
    python3 -m venv env
    source env/bin/activate
    ```

3. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the chatbot**:
    ```sh
    python app.py
    ```

2. **Interact with the chatbot**:
    - Open your web browser and go to the local URL provided by Gradio.
    - Start asking questions about architecture to get detailed responses.

## Project Structure

- `app.py`: The main application file that contains the code for the chatbot.
- `requirements.txt`: The file containing the required Python packages.

## Example Questions

- "What are the principles of sustainable architecture?"
- "Can you explain the characteristics of Gothic architecture?"
- "What are the best materials for eco-friendly buildings?"
- "How do I design a space with good natural lighting?"
- "What are some modern construction techniques?"

## Customization

- **System Message**: Modify the initial system message in `app.py` to change the chatbot's introduction and purpose.
- **Parameters**: Adjust the `max_tokens`, `temperature`, and `top_p` sliders in the Gradio interface to customize the chatbot's responses.

## Dependencies

- `gradio`: For creating the web interface.
- `huggingface_hub`: For interfacing with the Hugging Face Inference API.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing the Inference API.
- [Gradio](https://www.gradio.app/) for the user-friendly web interface.


Building a Retrieval-Augmented Generation (RAG) bot can significantly enhance the capabilities of a language model by incorporating external knowledge to generate more accurate and contextually relevant responses. This guide will walk you through creating a simple RAG bot using Gradio and the Hugging Face APIs.

But how does RAG enhance LLM’s performance?

RAG improves the performance of language models by augmenting them with external documents. This method retrieves relevant documents based on the user query and combines them with the original prompt before passing them to the language model for response generation. This approach ensures that the language model can access up-to-date and domain-specific information without the need for extensive retraining.



A common scenario of RAG helping LLM (Source)

The basic steps in RAG can be simplified as follows:

Input: The question to which the LLM system responds is referred to as the input. If no RAG is used, the LLM is directly used to respond to the question.

Indexing: If RAG is used, then a series of related documents are indexed by chunking them first, generating embeddings of the chunks, and indexing them into a vector store. At inference, the query is also embedded in a similar way.


Basic retrieval steps in RAG. (Source)

Retrieval: The relevant documents are obtained by comparing the query against the indexed vectors, also denoted as “Relevant Documents”.

Generation: The relevant documents are combined with the original prompt as additional context. The combined text and prompt are then passed to the model for response generation which is then prepared as the final output of the system to the user.

In the example provided, using the model directly fails to respond to the question due to a lack of knowledge of current events. On the other hand, when using RAG, the system can pull the relevant information needed for the model to answer the question appropriately. (Source)

Now Let’s Build a Chatbot using RAG:

I have used Zephyr LLM model and all-MiniLM-L6-v2 sentence transformer model. This sentence-transformers model maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.

The all-* models were trained on all available training data (more than 1 billion training pairs) and are designed as general purpose models. The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality. Toggle All models to see all evaluated original models.

We need the following ingredients:

1. A PDF as your knowledgebase

2. A requirements.txt file

3. An app.py file

4. An account on Hugging Face (See this blog to learn about building a LLM chatbot in Hugging Face)
