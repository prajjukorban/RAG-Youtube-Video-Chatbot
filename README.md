# 🎥 RAG YouTube Video Chatbot

A Streamlit-based Retrieval-Augmented Generation (RAG) application that allows you to chat directly with YouTube videos. Provide a YouTube link, and the app will extract the transcript, index it using a vector database, and answer your questions based on the video's content using large language models.

## ✨ Features

* **YouTube Transcript Extraction:** Automatically fetches video transcripts (prioritizing English, Hindi, and Kannada).
* **Smart Semantic Chunking:** Uses HuggingFace embeddings and LangChain's SemanticChunker to logically break down long transcripts.
* **In-Memory Vector Search:** Utilizes ChromaDB in-memory for lightning-fast, conflict-free Maximal Marginal Relevance (MMR) retrieval.
* **OpenRouter Integration:** Connects to OpenRouter's free API to generate intelligent, context-aware answers.
* **Interactive UI:** Built entirely in Streamlit for a clean, user-friendly chat interface.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **RAG Framework:** LangChain (`langchain-core`, `langchain-huggingface`, `langchain-chroma`, `langchain-experimental`)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Vector Database:** ChromaDB
* **LLM Provider:** OpenAI API (routed via OpenRouter)
* **Transcript API:** `youtube-transcript-api`

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed on your machine. You will also need a free API key from [OpenRouter](https://openrouter.ai/).

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/prajjukorban/RAG-Youtube-Video-Chatbot.git
   cd RAG-Youtube-Video-Chatbot


2. **Create a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```



### Usage

1. **Run the Streamlit application:**
```bash
streamlit run app.py

```


2. **Configure the app:**
* Open the provided local URL (usually `http://localhost:8501`) in your browser.
* In the left sidebar, enter your **OpenRouter API Key**.
* Paste the **YouTube URL** of the video you want to analyze.
* Click **Process Video**.


3. **Start Chatting:**
* Once the video is processed, use the chat input at the bottom to ask questions about the video content. The AI will respond based exclusively on the video's transcript.



## 📝 Folder Structure

```text
RAG-Youtube-Video-Chatbot/
│
├── app.py                  
├── requirements.txt
├── Youtube_video_chatbot.ipynb
└── README.md               

```

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a Pull Request if you have suggestions for improvements, new features, or bug fixes.

## 📄 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE&utm_source=gemini).

```
