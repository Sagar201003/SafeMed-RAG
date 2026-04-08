# Safe Medical RAG Bot 🛡️🏥

> **Note:** This project is currently an interactive **Demo / Work in Progress**. It is strictly designed to showcase the architectural integration of safety pipelines within Retrieval-Augmented Generation (RAG) applications for academic presentations.

## Overview
The Safe Medical RAG Bot is a highly secure, interactive web application built with Streamlit and Groq's fast `llama-3.1-8b-instant`. The primary objective of this application is to demonstrate a modern, multi-layered security framework capable of sanitizing untrusted user inputs, preventing prompt injections, isolating poisoned document embeddings, and verifying the mathematical safety grounding of Large Language Model responses.

## 🚀 The Safety Guardrail Pipeline
As AI agents gain broader access to critical environments—such as healthcare and medicine—security becomes the number one priority. This RAG framework natively routes every single interaction through three critical choke points:

1. **Input Stage Guardrails:**
   - Evaluates incoming queries natively for toxic terminology, implicit self-harm, and off-topic rejection filtering *prior* to database execution to save compute.
   - Detects advanced structural Prompt Injections (e.g., hidden "Do Anything Now" / DAN payloads).

2. **Retrieval Stage Guardrails (Anti-Poisoning Defense):**
   - Encodes text inputs locally using `sentence-transformers/all-MiniLM-L6-v2`.
   - Before forwarding embedding contexts to the LLM, the retrieval filter actively scans the raw chunks for implicitly poisoned payloads. If an adversarial chunk is retrieved from a compromised origin within the vector database, the system draws a red line through it and rigorously discards it.

3. **Output Stage Guardrails (Anti-Hallucination & PII Scrubbing):**
   - Intercepts Llama 3.1's synthetic generation payload prior to user deployment.
   - Mathematically calculates the **Grounding Score** (semantic token overlap with the raw retrieved contextual database text) to identify if the LLM hallucinated outside data.
   - Runs deterministic regular expressions to securely `[REDACT]` globally identified PII (SSNs, Phone Numbers, Emails) before natively rendering onto the screen.

## 💻 Tech Stack
- **Frontend / UI System:** Streamlit
- **LLM Engine:** Groq API (`llama-3.1-8b-instant`)
- **Fast Vector Encoding:** PyTorch & `sentence-transformers`
- **Dynamic Credential Handling:** `python-dotenv`

## ⚙️ Running Locally
1. Clone the repository natively.
2. Install the necessary dependencies (we enforce stable CPU compilation tracking for Windows PyTorch mapping):
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables locally. Ensure you create a `.env` file tracking your Groq API credentials:
   ```env
   GROQ_API_KEY="gsk_..."
   ```
4. Run the Streamlit Dashboard natively on port 8501:
   ```bash
   python -m streamlit run app.py
   ```
