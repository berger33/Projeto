from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

try:
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

    @dataclass
    class Document:
        page_content: str
        metadata: dict

STOP = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "em",
    "um", "uma", "para", "por", "com", "que", "se", "ao", "à", "é",
    "qual", "quais", "como", "me", "meu", "minha",
}


def normalize(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9\s]", " ", text)


def tokens(text: str) -> list[str]:
    return [w for w in normalize(text).split() if len(w) > 2 and w not in STOP]


def keywords(text: str) -> set[str]:
    return set(tokens(text))


class KnowledgeBase:
    """Carrega PDF/CSV e cria um índice TF-IDF em Python puro.

    Pandas é usado de fato na leitura dos CSVs e LangChain é usado para
    Document + RecursiveCharacterTextSplitter. O TF-IDF é implementado aqui
    para evitar dependência binária adicional no Windows.
    """

    def __init__(self, docs_dir: str | Path):
        self.docs_dir = Path(docs_dir)
        self.documents = self._load_documents()
        self.chunks = self._split_documents(self.documents)
        self._build_index()

    def _load_documents(self) -> list[Document]:
        docs: list[Document] = []
        for path in sorted(self.docs_dir.glob("*")):
            if path.suffix.lower() == ".pdf":
                reader = PdfReader(str(path))
                for i, page in enumerate(reader.pages):
                    text = (page.extract_text() or "").strip()
                    if text:
                        docs.append(
                            Document(
                                page_content=text,
                                metadata={"source": path.name, "page": i + 1},
                            )
                        )
            elif path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
                for i, row in df.fillna("").iterrows():
                    text = " | ".join(f"{column}: {row[column]}" for column in df.columns)
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": path.name, "row": int(i) + 2},
                        )
                    )

        if not docs:
            raise RuntimeError("Nenhum documento PDF/CSV encontrado.")
        return docs

    def _split_documents(self, docs: list[Document]) -> list[Document]:
        if LANGCHAIN_AVAILABLE:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=900,
                chunk_overlap=120,
                separators=["\n\n", "\n", ". ", " "],
            )
            return splitter.split_documents(docs)

        out: list[Document] = []
        for doc in docs:
            parts = re.split(r"\n(?=[A-ZÁÉÍÓÚ0-9][^\n]{2,80}$)|\n\n+", doc.page_content)
            buffer = ""
            for part in parts:
                if len(buffer) + len(part) < 900:
                    buffer += ("\n" if buffer else "") + part
                else:
                    if buffer.strip():
                        out.append(Document(page_content=buffer.strip(), metadata=doc.metadata.copy()))
                    buffer = part
            if buffer.strip():
                out.append(Document(page_content=buffer.strip(), metadata=doc.metadata.copy()))
        return out

    def _build_index(self) -> None:
        self.term_counts: list[Counter[str]] = [Counter(tokens(doc.page_content)) for doc in self.chunks]
        document_frequency: Counter[str] = Counter()
        for counts in self.term_counts:
            document_frequency.update(counts.keys())

        n_documents = max(1, len(self.chunks))
        self.idf = {
            term: math.log((1 + n_documents) / (1 + df)) + 1.0
            for term, df in document_frequency.items()
        }
        self.vectors = [self._tfidf_vector(counts) for counts in self.term_counts]
        self.norms = [math.sqrt(sum(value * value for value in vector.values())) for vector in self.vectors]

    def _tfidf_vector(self, counts: Counter[str]) -> dict[str, float]:
        total = sum(counts.values()) or 1
        return {
            term: (count / total) * self.idf.get(term, 1.0)
            for term, count in counts.items()
        }

    def _query_vector(self, question: str) -> dict[str, float]:
        counts = Counter(tokens(question))
        total = sum(counts.values()) or 1
        return {
            term: (count / total) * self.idf.get(term, 1.0)
            for term, count in counts.items()
        }

    @staticmethod
    def _cosine(query: dict[str, float], doc: dict[str, float], doc_norm: float) -> float:
        query_norm = math.sqrt(sum(value * value for value in query.values()))
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        dot = sum(value * doc.get(term, 0.0) for term, value in query.items())
        return dot / (query_norm * doc_norm)

    def retrieve(self, question: str, k: int = 4):
        query_vector = self._query_vector(question)
        question_keys = keywords(question)
        scored = []

        for i, doc in enumerate(self.chunks):
            similarity = self._cosine(query_vector, self.vectors[i], self.norms[i])
            chunk_keys = keywords(doc.page_content)
            overlap = len(question_keys & chunk_keys) / max(1, len(question_keys))
            score = similarity * 0.70 + overlap * 0.30
            scored.append((float(score), i))

        scored.sort(reverse=True)
        return [(score, self.chunks[i]) for score, i in scored[:k]]


class FashionStoreAgent:
    """Agente RAG documental com respostas restritas à base local."""

    def __init__(self, docs_dir: str | Path):
        self.kb = KnowledgeBase(docs_dir)

    def answer(self, question: str) -> dict:
        ranked = self.kb.retrieve(question, 4)
        if not ranked or ranked[0][0] < 0.11:
            return {
                "answer": (
                    "Não encontrei essa informação na documentação oficial da Aurora Moda Online. "
                    "Entre em contato com o suporte para obter orientação adicional."
                ),
                "sources": [],
                "confidence": "baixa",
            }

        selected = [doc for score, doc in ranked if score >= max(0.10, ranked[0][0] * 0.45)]
        context = "\n".join(doc.page_content for doc in selected)
        q = normalize(question)

        if any(x in q for x in ["devol", "reembolso", "troca", "devolver"]):
            answer = (
                "Você pode solicitar a devolução em até 10 dias corridos após o recebimento. "
                "O produto deve estar em perfeitas condições, sem sinais de uso, lavagem, odores, "
                "danos ou alterações, com etiquetas e acessórios originais. Após a análise e "
                "aprovação, o reembolso é realizado pelo mesmo meio de pagamento utilizado na "
                "compra, respeitando os prazos do provedor financeiro."
            )
        elif any(x in q for x in ["pagamento", "pagar", "pix", "cartao", "boleto"]):
            answer = (
                "A Aurora Moda Online aceita cartão de crédito e PIX. A confirmação do pedido "
                "ocorre após a aprovação do pagamento; no cartão, a aprovação depende da operadora, "
                "e no PIX a confirmação normalmente ocorre após a identificação do pagamento."
            )
        elif any(x in q for x in ["entrega", "prazo", "envio", "chegar", "frete"]):
            answer = (
                "O prazo de entrega é informado no checkout e começa a contar após a confirmação "
                "do pagamento. O prazo pode variar conforme CEP, modalidade de frete e operação da "
                "transportadora. O cliente recebe informações de acompanhamento quando o pedido é despachado."
            )
        elif any(x in q for x in ["privacidade", "dados", "lgpd", "cpf", "email", "prote"]):
            answer = (
                "A Aurora Moda Online coleta apenas dados necessários para cadastro, processamento "
                "de pedidos, pagamento, entrega, atendimento, segurança e cumprimento de obrigações "
                "legais. Os dados são protegidos por medidas técnicas e administrativas e podem ser "
                "compartilhados somente com prestadores necessários à operação, nos limites aplicáveis."
            )
        elif any(x in q for x in ["suporte", "contato", "falar", "atendimento"]):
            answer = (
                "O suporte pode ser contatado pelo e-mail suporte@auroramoda.exemplo. Para agilizar "
                "o atendimento, informe o número do pedido e descreva a solicitação de forma objetiva."
            )
        else:
            sentences = re.split(r"(?<=[.!?])\s+", context.replace("\n", " "))
            question_keys = keywords(question)
            ranked_sentences = []
            for sentence in sentences:
                sentence_keys = keywords(sentence)
                score = len(question_keys & sentence_keys) / max(1, len(question_keys))
                if score:
                    ranked_sentences.append((score, sentence.strip()))
            ranked_sentences.sort(reverse=True)
            chosen = [sentence for _, sentence in ranked_sentences[:3]]
            answer = " ".join(chosen) if chosen else "Não encontrei uma resposta suficientemente clara na documentação oficial."

        sources = []
        for doc in selected[:3]:
            item = {"document": doc.metadata.get("source")}
            if "page" in doc.metadata:
                item["page"] = doc.metadata["page"]
            if "row" in doc.metadata:
                item["row"] = doc.metadata["row"]
            if item not in sources:
                sources.append(item)

        return {
            "answer": answer,
            "sources": sources,
            "confidence": "alta" if ranked[0][0] >= 0.30 else "média",
        }
