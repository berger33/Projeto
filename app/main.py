from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .agent import FashionStoreAgent, LANGCHAIN_AVAILABLE

BASE = Path(__file__).resolve().parents[1]
agent = FashionStoreAgent(BASE / "docs")
app = FastAPI(title="Aurora Moda - Agente Inteligente", version="1.0.0")


class Ask(BaseModel):
    question: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents": len(agent.kb.documents),
        "chunks": len(agent.kb.chunks),
        "langchain_available": LANGCHAIN_AVAILABLE,
    }


@app.post("/api/ask")
def ask(payload: Ask):
    return agent.answer(payload.question)


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(
        '''<!doctype html>
<meta charset="utf-8">
<title>Aurora Moda</title>
<style>
body{font-family:Arial;background:#f6f4f1;margin:40px}
.card{max-width:850px;margin:auto;background:#fff;padding:32px;border-radius:20px;box-shadow:0 12px 40px #0001}
textarea{width:100%;height:90px;padding:12px;box-sizing:border-box}
button{padding:12px 18px;background:#111;color:white;border:0;border-radius:10px;cursor:pointer}
.a{margin-top:16px;padding:16px;background:#f3f3f3;border-radius:10px;white-space:pre-wrap}
</style>
<div class="card">
  <h1>Assistente Aurora Moda</h1>
  <p>Respostas baseadas exclusivamente na documentação oficial.</p>
  <textarea id="q" placeholder="Digite sua pergunta"></textarea><br><br>
  <button onclick="askAgent()">Perguntar</button>
  <div id="a" class="a">Pronto para responder.</div>
</div>
<script>
async function askAgent(){
  const response = await fetch('/api/ask', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question:document.getElementById('q').value})
  });
  const data = await response.json();
  const sources = (data.sources || []).map(x => x.document).join(', ');
  document.getElementById('a').innerText = data.answer + (sources ? '\n\nFontes: ' + sources : '');
}
</script>'''
    )
