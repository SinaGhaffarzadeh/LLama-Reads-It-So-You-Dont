
# 🧠📄 Retrieval-Augmented Generation (RAG) PDF QA System

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline to answer natural language questions over a collection of PDF documents. It combines **semantic retrieval** using embedding models with **generative reasoning** powered by large language models (LLMs).  

Built with **LlamaIndex**, **Hugging Face Transformers**, and **SentenceTransformers**, the system enables intelligent document understanding — all locally and with GPU support.

---

## 🧱 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a hybrid NLP architecture that enhances language models by grounding their responses in external knowledge sources.

### 🔁 RAG Workflow

```mermaid
graph TD
    A[User Query] --> B[Retriever (Embeddings + Vector DB)]
    B --> C[Top-K Relevant Chunks]
    C --> D[LLM (Generator)]
    D --> E[Answer]
```

- **Retriever**: Uses embedding-based similarity to fetch relevant chunks from a vector store (e.g., Chroma, FAISS).
- **Generator**: A decoder-only LLM that synthesizes a natural language answer using the retrieved context.
- **Result**: More factual, up-to-date, and domain-adaptable responses.

---

## ⚙️ How to Build a RAG Pipeline

1. **Document Ingestion**: Load and chunk your PDFs.
2. **Embedding**: Convert each chunk into vector representations using a SentenceTransformer model.
3. **Indexing**: Store vectors in a memory index for quick retrieval.
4. **Querying**: Accept user questions and embed them.
5. **Retrieval**: Find the most relevant text chunks.
6. **Generation**: Pass the retrieved content and question to an LLM to generate a response.

---

## 📦 Project Details

### 🔧 Features

- ✅ **End-to-End Local RAG Pipeline**
- 🧾 Multiple PDF ingestion support
- 🔍 Natural language querying
- ⚡ GPU acceleration with CUDA
- 🛠️ Robust `try-except` error handling

### 📁 Structure

```
├── Data/                      # PDF files go here
├── main.py                   # Main application logic
├── requirements.txt          # Python dependencies
└── README.md                 # You're here!
```

---

## 🧠 Models Used

| Role | Model | Type | Strengths |
|------|-------|------|-----------|
| **Retriever** | `sentence-transformers/all-MiniLM-L6-v2` | Bi-Encoder | Lightweight, fast, excellent for small-scale retrieval |
| **Generator** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Decoder-only LLM | Handles long context (2048+ tokens), fast, locally runnable |

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/llm-pdf-qa.git
cd llm-pdf-qa
```

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Authenticate with Hugging Face

```python
from huggingface_hub import login
login(token="your_huggingface_token")
```

Or via environment variable:

```bash
export HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

---

## 📥 Usage

1. Place your PDF files inside the `Data/` directory.
2. Run the application:

```bash
python main.py
```

3. The system will:
   - Parse PDFs
   - Embed and index the text
   - Use a built-in query (e.g., `"What is motion?"`)
   - Return a generated response based on retrieved documents

---

## 📊 Model Comparison

### 🔍 Embedding Models

| Model | Dim | Speed | Accuracy | Best For |
|-------|-----|-------|----------|----------|
| `all-MiniLM-L6-v2` | 384 | ✅ Fast | ⚠️ Good | Lightweight RAG on local systems |
| `e5-base-v2` | 768 | ⚠️ Medium | ✅ High | Instruction-aware retrieval |
| `bge-base-en-v1.5` | 768 | ⚠️ Medium | ✅ High | Open-source alternative to OpenAI |
| `text-embedding-3-small` (OpenAI) | 1536 | ✅ Fast | ✅ Very High | Commercial APIs (Paid) |

### 🧠 LLMs for Generation

| Model | Type | Size | Max Tokens | Quality | Notes |
|-------|------|------|------------|---------|-------|
| `TinyLlama-1.1B-Chat` | Decoder-only | 1.1B | 2048+ | ✅ High | Efficient for local inference |
| `flan-t5-base` | Encoder-Decoder | 250M | 512 | ⚠️ Medium | Limited by token size |
| `LLaMA-2-7B-Chat` | Decoder-only | 7B | ~4K | ✅ High | Heavier, accurate |
| `Mistral-7B-Instruct` | Decoder-only | 7B | 8192 | ✅ High | Great reasoning |
| `gpt-3.5-turbo` | API | ~20B | 8K+ | 🏆 Very High | Paid, cloud only |

> ⚠️ **Note:** flan-t5-base often fails in RAG due to short token limits (512). Prefer decoder-only models like TinyLlama or Mistral for longer queries.

---

## 🧪 Example Output

```bash
Cuda is available! True
The version of Cuda is: 12.1
{'name': 'your-username', 'email': 'your@email.com'}
Response: Motion is defined as...
```

---

## 💡 Future Roadmap

- [ ] Gradio/Streamlit Web UI
- [ ] User-input-based querying
- [ ] DOCX/TXT document support
- [ ] Persistent vector DB (FAISS, ChromaDB)
- [ ] Re-ranking and multi-round generation

---

## 🤝 Contributions

Feel free to fork and contribute. PRs are welcome!

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for more details.

---

## ✍️ Author

**Your Name**  
[GitHub](https://github.com/your-username) • [LinkedIn](https://linkedin.com/in/your-profile)
