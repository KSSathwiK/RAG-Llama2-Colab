# RAG-llama2-Google-Colab
This project, RAG-llama2-Google-Colab, leverages a Retrieval-Augmented Generation (RAG) architecture with the LLaMA-2 model for enhanced information retrieval and response generation. The setup is hosted on Google Colab, allowing for easy access and integration of LLaMA-2's capabilities with a custom knowledge base.

The primary objective of this project is to utilize a folder path containing domain-specific documents (e.g., PDFs, text files, or other formats) to construct a knowledge base. The RAG architecture then effectively combines retrieval and generation to provide accurate and contextually relevant answers based on the given information.


## Environment Variables

You'll need to create a `.env` file in the root of the project directory with your Hugging Face API token:

```
HF_API_TOKEN=your_huggingface_api_token_here
```

## Running the App

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Activate on Windows
   venv\Scripts\activate
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
