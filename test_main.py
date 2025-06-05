# test_main.py

import os
import sys
import types
import re
import pytest
import torch
import nest_asyncio
from fastapi.testclient import TestClient
import warnings
from _pytest.deprecated import PytestDeprecationWarning

# --------------------------------------------------------------------------------
# Silence pytest-asyncio deprecation warning about fixture loop scope
warnings.filterwarnings("ignore", category=PytestDeprecationWarning)

# --------------------------------------------------------------------------------
# 1) Stub out heavy modules *before* importing main.py so that
#    import main does not attempt to load real models or external services.

for name in [
    'uvicorn',                  # server runner
    'google', 'google.colab', 'google.colab.drive',  # Colab drive mount
]:
    sys.modules[name] = types.ModuleType(name)

# --------------------------------------------------------------------------------
# Stub out the transformers library
tf = types.ModuleType('transformers')
class DummyAutoModel:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        # Return a dummy model instance
        return DummyAutoModel()
    def generate(self, **kwargs):
        # Return a dummy tensor to simulate generation
        return torch.tensor([[0,1,2]])
class DummyAutoTok:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        # Return a dummy tokenizer instance
        return DummyAutoTok()
    def __call__(self, texts, return_tensors="pt"):
        # Simulate tokenization output and support .to(device)
        class DummyInputs(dict):
            def to(self, device):
                return self
        return DummyInputs(input_ids=torch.tensor([[0,1,2]]))
    def batch_decode(self, outputs, skip_special_tokens=True):
        # Simulate decoding to text containing our marker
        return ["Prelude\n### Response: Dummy out."]
tf.AutoModelForCausalLM = DummyAutoModel
tf.AutoTokenizer       = DummyAutoTok
sys.modules['transformers'] = tf

# --------------------------------------------------------------------------------
# Stub out PEFT adapter library
peft_mod = types.ModuleType('peft')
class DummyPeft:
    @staticmethod
    def from_pretrained(base, path, **kw):
        return DummyPeft()
    def merge_and_unload(self):
        return self
    def eval(self):
        pass
peft_mod.PeftModel = DummyPeft
sys.modules['peft'] = peft_mod

# --------------------------------------------------------------------------------
# Stub out sentence-transformers semantic matching
st_mod = types.ModuleType('sentence_transformers')
util_mod = types.ModuleType('sentence_transformers.util')
class DummyST:
    def __init__(self, *a, **k):
        pass
    def encode(self, texts, convert_to_tensor=False):
        # Return a simple tensor [0], [1], ... for each sentence
        return torch.arange(len(texts), dtype=torch.float32).unsqueeze(1)
# Simulate cosine similarity by broadcasting embeddings
util_mod.cos_sim = lambda a, b: b.T
st_mod.SentenceTransformer = lambda *a, **k: DummyST()
st_mod.util = util_mod
sys.modules['sentence_transformers'] = st_mod
sys.modules['sentence_transformers.util'] = util_mod

# --------------------------------------------------------------------------------
# 2) Now import the unmodified main.py; all heavy imports are stubbed above
import main

# --------------------------------------------------------------------------------
# 3) Apply nest_asyncio so that FastAPI’s TestClient works in this environment
nest_asyncio.apply()

# --------------------------------------------------------------------------------
# 4) Create a TestClient for end-to-end HTTP tests
client = TestClient(main.app)


#########################
# Unit tests for helper functions
#########################

def test_remove_citations_basic():
    """
    remove_citations should strip out any [n] citations, but otherwise leave
    punctuation (including spaces before punctuation) unchanged per current behavior.
    """
    out = main.remove_citations("Hello [1], world [2]!")
    assert "[1]" not in out and "[2]" not in out
    assert out.startswith("Hello") and out.endswith(" !")

def test_reinsert_citations_semantic_contains_all():
    """
    reinsert_citations_semantic should reinsert all citations from the original
    into the generated text, under our dummy semantic stub.
    """
    orig = "First [A] x. Second [B] y."
    gen  = "First x. Second y."
    out  = main.reinsert_citations_semantic(orig, gen)
    assert "[A]" in out and "[B]" in out

def test_no_citations_input():
    """
    When there are no citations in the input, remove_citations returns the same text,
    and reinsert_citations_semantic returns the generated text unchanged.
    """
    text = "No citations here."
    assert main.remove_citations(text) == text
    assert main.reinsert_citations_semantic(text, text) == text


#########################
# Integration tests for /convert endpoint
#########################

@pytest.fixture(autouse=True)
def patch_tokenizer_and_model(monkeypatch):
    """
    Before each integration test, monkey-patch main.tokenizer and main.model
    with our dummy classes so that no real model is loaded.
    """
    monkeypatch.setattr(main, "tokenizer", DummyAutoTok())
    monkeypatch.setattr(main, "model",     DummyAutoModel())

def test_convert_happy_path():
    """
    A valid POST to /convert should return status 200 and include both the dummy
    LLM output ("Dummy out.") and the original citation marker in the response.
    """
    r = client.post("/convert", json={"section":"Intro", "content":"Hi [Z]."})
    assert r.status_code == 200
    data = r.json()
    assert "Dummy out." in data["converted_text"]
    assert "[Z]" in data["converted_text"]

def test_convert_empty_also_returns_dummy():
    """
    Although content is whitespace-only, main.py does not 400 on empty content,
    so we expect a dummy conversion with status 200.
    """
    r = client.post("/convert", json={"section":"Empty", "content":"   "})
    assert r.status_code == 200
    assert "Dummy out." in r.json()["converted_text"]

def test_missing_content_field():
    """
    Omitting the required 'content' field should trigger a 422 Unprocessable Entity
    with an error detailing that 'content' is required.
    """
    r = client.post("/convert", json={"section":"OnlySection"})
    assert r.status_code == 422
    errs = r.json()["detail"]
    assert any(e["loc"][-1] == "content" for e in errs)

def test_invalid_content_type():
    """
    Providing a non-string type for 'content' should also yield a 422 error.
    """
    r = client.post("/convert", json={"section":"Sec", "content": 123})
    assert r.status_code == 422

def test_method_not_allowed():
    """
    Sending a GET request to /convert (which only accepts POST) should return 405.
    """
    r = client.get("/convert")
    assert r.status_code == 405

def test_long_document_reinsertion():
    """
    For a longer document with multiple citations, reinsert_citations_semantic
    should still reinsert all citation markers.
    """
    orig = " ".join(f"Sent{i} [{i}]." for i in range(10))
    gen  = " ".join(f"Sent{i}." for i in range(10))
    out = main.reinsert_citations_semantic(orig, gen)
    for i in range(10):
        assert f"[{i}]" in out

def test_response_schema():
    """
    The /convert endpoint's JSON response should contain exactly one key:
    'converted_text', whose value is a string.
    """
    r = client.post("/convert", json={"section":"S", "content":"X [1]."})
    assert r.status_code == 200
    j = r.json()
    assert set(j.keys()) == {"converted_text"}
    assert isinstance(j["converted_text"], str)
