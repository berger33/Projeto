"""Normalização e tokenização de texto em PT-BR, compartilhadas pelo pipeline.

Ponto único para o que antes estava triplicado (Fase 2, G-15). Por enquanto é usado pela detecção de
recusa e pela verificação de sustentação (P1-01); a busca lexical (P1-04) passa a usar este módulo.

Todas as funções operam sobre texto **normalizado**: NFKD sem marcas diacríticas, minúsculas, espaços
colapsados. ``devolução``/``devolucao``/``DEVOLUÇÃO`` viram o mesmo token ``devolucao``.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WS_RE = re.compile(r"\s+")
_MARKDOWN_RE = re.compile(r"[*_`#>]+")

# Palavras funcionais do PT-BR (já normalizadas em ``STOPWORDS_PT``). Lista deliberadamente
# conservadora: só termos que não carregam conteúdo para busca ou verificação de sustentação.
_STOPWORDS_RAW = """
a à às ao aos aquela aquelas aquele aqueles aquilo as até com como da das de dela delas dele deles
depois do dos e é ela elas ele eles em entre era eram essa essas esse esses esta está estamos estão
estas estava estavam este esteja estejam estes esteve estive estivemos estiver estiveram estou eu foi
fomos for foram fosse fossem fui há isso isto já lhe lhes mais mas me mesmo mesma meu meus minha
minhas muito muitos muitas na não nas nem no nos nós nossa nossas nosso nossos num numa o os ou para
pela pelas pelo pelos por qual quais quando que quem se seja sejam sem ser será serão seu seus só
somos sou sua suas também te tem têm temos tenho ter teu teus tinha tinham tive tivemos tiver tiveram
tu tua tuas um uma você vocês vos pode podem poderá poderão deve devem deverá deverão sim caso onde
sobre após ainda cada ante contra desde durante perante sob algum alguma alguns algumas nenhum nenhuma
todo toda todos todas outro outra outros outras tal tais aqui ali lá então assim porém portanto pois
porque sendo sido tendo tido estar estando estado haver havia houve faz fazer fez feito vai vão ir
isso disso nisso desse dessa deste desta neste nesta nesse nessa daquele daquela ela ele lhe
"""
STOPWORDS_PT: frozenset[str] = frozenset()


def strip_accents(text: str) -> str:
    """Remove marcas diacríticas preservando o restante (``ç`` → ``c``, ``ã`` → ``a``)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize(text: str) -> str:
    """Minúsculas, sem acentos, sem marcação Markdown simples, espaços colapsados."""
    lowered = strip_accents(text).lower()
    lowered = _MARKDOWN_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", lowered).strip()


def tokenize(text: str) -> list[str]:
    """Tokens alfanuméricos do texto normalizado (mantém a ordem e as repetições)."""
    return _TOKEN_RE.findall(normalize(text))


def content_tokens(text: str, *, min_length: int = 3) -> list[str]:
    """Tokens com conteúdo: fora das stopwords e com ``min_length`` ou mais caracteres.

    Números são sempre mantidos (``10`` em "10 dias" é informação verificável).
    """
    return [
        token for token in tokenize(text) if token not in STOPWORDS_PT and (len(token) >= min_length or token.isdigit())
    ]


def _build_stopwords() -> frozenset[str]:
    return frozenset(normalize(word) for word in _STOPWORDS_RAW.split())


STOPWORDS_PT = _build_stopwords()
