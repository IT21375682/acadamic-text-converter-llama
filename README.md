# 🧠 Academic Text Converter (LLaMA-2 + LoRA)

This project transforms informal academic content into IEEE-compliant academic text using a fine-tuned LLaMA-2 model with a LoRA adapter. It supports semantic citation reinsertion using `sentence-transformers`.

---

## 🚀 Features

- Converts informal text into formal academic language
- Merges LoRA adapter with LLaMA-2
- Reinserts citations based on semantic similarity
- Runs entirely on CPU (optional GPU support with tweaks)

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/IT21375682/acadamic-text-converter-llama.git
cd acadamic-text-converter-llama
```

### 2. Create `.env` File

Create a `.env` file in the root directory with your Hugging Face token:
```clone
https://huggingface.co/Shandeep201/llama-2
```

```env
HF_TOKEN=your_huggingface_token_here
```

> 🔒 **Important:** Never commit `.env` files or secrets to GitHub!

---

### 3. Create a Python Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
# Or: source venv/bin/activate   # On macOS/Linux
```

---

### 4. Install Dependencies

Create a `requirements.txt` file with the following content:

```text
fastapi
uvicorn
transformers
torch
peft
sentence-transformers
nltk
python-dotenv
nest_asyncio
```

Then install:

```bash
pip install -r requirements.txt
```

---

### 5. Run the Server

```bash
python main.py
```

Then access the API at:  
📍 `http://localhost:8000/convert`

---

## 🔁 API Usage

### POST `/convert`

**Request Body:**

```json
{
  "section": "Abstract",
  "content": "This study talks about how machine learning is kinda useful for predicting stuff...[1]"
}
```

**Response:**

```json
{
  "converted_text": "This study investigates the efficacy of machine learning in predictive analytics... [1]"
}
```

---

## 🧪 Example Prompt Format

```text
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Transform the following informal text into formal academic language, ensuring compliance with IEEE formatting standards. Retain the original meaning but do not include any citation markers in the output:

### Input:
The research kinda shows that AI helps a lot in many stuff.

### Response:
The research demonstrates that artificial intelligence significantly contributes to a wide range of applications.
```

---

## 📚 License

This project is for educational and research purposes only.

---

## ✍️ Author

- **IT21375682** – SLIIT Final Year Research  
- 💬 Contact: [GitHub Profile](https://github.com/IT21375682)
