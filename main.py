import os
import re
import json
import torch
import threading
import uvicorn
import nest_asyncio


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# Additional imports for semantic matching
from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
#from nltk.tokenize import sent_tokenizefrom pydantic import BaseModel
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel

# ----------------------------------------------
# 1) Load environment variables from a .env file
# ----------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# Retrieve the Hugging Face token from the environment
hf_token =process.env.get("HF_TOKEN") or os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set!")

# -------------------------------------------------
# 2) Set up a Transformers cache directory (optional)
# -------------------------------------------------
cache_dir = "/tmp/hf_cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["XDG_CACHE_HOME"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir

# -------------------------------------------------
# 3) Import from transformers + peft
# -------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Model repository and config
adapter_model_path = "Shandeep201/llama-2"
# base_model_name = "unsloth/llama-2-7b-bnb-4bit"
base_model_name = "meta-llama/Llama-2-7b-chat-hf"
max_seq_length = 2048

print("Loading base model on CPU (float32)...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32,    # standard CPU float
    token=hf_token
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    use_auth_token=hf_token
)

print("Attaching LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    adapter_model_path,
    use_auth_token=hf_token
)

print("Merging LoRA adapter into the model ...")
model = model.merge_and_unload()
model.eval()





print("Model loaded on CPU.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class TextRequest(BaseModel):
    section: str  # e.g., "Abstract", "Methodology"
    content: str  # The paragraph to convert

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def remove_citations(text: str) -> str:
    """Remove citation markers from the text using regex."""
    return re.sub(r'\[[^\]]+\]', '', text)

def reinsert_citations_semantic(original_text: str, generated_text: str) -> str:
    """
    Reinserts citations into the generated_text using semantic matching.
    Splits the original text into sentences, extracts citations from those sentences,
    and then uses SentenceTransformer embeddings to locate the best insertion point in the generated text.
    """
    # Split texts into sentences
    orig_sentences = sent_tokenize(original_text)
    gen_sentences = sent_tokenize(generated_text)

    # Prepare a list of (clean sentence, list of citations) tuples from the original text
    citation_data = []
    for sent in orig_sentences:
        citations = re.findall(r'\[[^\]]+\]', sent)
        if citations:
            clean_sent = re.sub(r'\[[^\]]+\]', '', sent)
            citation_data.append((clean_sent.strip(), citations))

    sem_model = SentenceTransformer('all-MiniLM-L6-v2')
    gen_embeddings = sem_model.encode(gen_sentences, convert_to_tensor=True)

    # Map generated sentence indices to citations
    insertion_map = {}
    for clean_sent, citations in citation_data:
        orig_embedding = sem_model.encode(clean_sent, convert_to_tensor=True)
        cos_scores = util.cos_sim(orig_embedding, gen_embeddings)[0]
        best_idx = int(cos_scores.argmax())
        insertion_map.setdefault(best_idx, []).extend(citations)

    # Reassemble generated sentences, appending citations where needed
    new_gen_sentences = []
    for idx, sent in enumerate(gen_sentences):
        if idx in insertion_map:
            sent = sent.strip() + " " + " ".join(insertion_map[idx])
        new_gen_sentences.append(sent)

    final_text = " ".join(new_gen_sentences)
    return final_text

@app.post("/convert")
async def convert_text(request: TextRequest):
    try:
        # Remove citations from input text to prevent model from re-generating them
        clean_content = remove_citations(request.content)

        # Use the clean text in the prompt
        prompt = alpaca_prompt.format(
            "Transform the following informal text into formal academic language, ensuring compliance with IEEE formatting standards. Retain the original meaning but do not include any citation markers in the output:",
            clean_content,
            ""
        )

        inputs = tokenizer([prompt], return_tensors="pt").to("cpu")
        outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
        full_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        if "### Response:" in full_response:
            converted_text = full_response.split("### Response:")[1].strip()
        else:
            converted_text = full_response.strip()

        # Reinsert the original citations based on semantic matching
        final_text = reinsert_citations_semantic(request.content, converted_text)
        return {"converted_text": final_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Apply nest_asyncio to fix event loop issue in Google Colab
nest_asyncio.apply()

# You can either run uvicorn directly from code:
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)