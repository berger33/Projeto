from __future__ import annotations

import re
from pathlib import Path

from .config import Settings
from .documents import load_chunks
from .domain import RAGAnswer, SourceRef
from .embeddings import HashEmbeddingProvider, OllamaEmbeddingProvider
from .generation import ExtractiveGenerator, OllamaGenerator
from .retrieval import VectorIndex

STOPWORDS = {"qual", "quais", "como", "para", "com", "uma", "uns", "das", "dos", "que", "por", "ser", "sao", "são", "esta", "está", "meu", "minha"}


def _terms(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text) if len(token) > 2 and token.lower() not in STOPWORDS}


class RAGService:
    def __init__(self, docs_dir: str | Path, settings: Settings):
        self.settings = settings
        chunks = load_chunks(docs_dir)
        if settings.rag_mode == "ollama":
            embeddings = OllamaEmbeddingProvider(settings.ollama_base_url, settings.embedding_model)
            self.generator = OllamaGenerator(settings.ollama_base_url, settings.generation_model)
        else:
            embeddings = HashEmbeddingProvider()
            self.generator = ExtractiveGenerator()
        self.index = VectorIndex(chunks, embeddings)

    @property
    def chunk_count(self) -> int:
        return len(self.index.chunks)

    def answer(self, question: str) -> RAGAnswer:
        question = question.strip()
        if not question:
            raise ValueError("A pergunta não pode ser vazia.")
        ranked = self.index.search(question, k=self.settings.retrieval_k)
        question_terms = _terms(question)
        selected = [item for item in ranked if item.score >= self.settings.min_score and question_terms & _terms(item.chunk.text)]
        if not selected:
            return RAGAnswer(
                answer="Não encontrei informação suficiente na documentação oficial da Aurora Moda Online.",
                sources=[],
                confidence="baixa",
                mode=self.generator.mode,
            )
        answer = self.generator.generate(question, selected)
        if answer.lower().startswith("não encontrei informação suficiente"):
            return RAGAnswer(answer=answer, sources=[], confidence="baixa", mode=self.generator.mode)
        sources: list[SourceRef] = []
        for item in selected[:3]:
            ref = SourceRef(document=item.chunk.source, page=item.chunk.locator.get("page"), row=item.chunk.locator.get("row"))
            if ref not in sources:
                sources.append(ref)
        top_score = selected[0].score
        return RAGAnswer(answer=answer, sources=sources, confidence="alta" if top_score >= 0.45 else "média", mode=self.generator.mode)
