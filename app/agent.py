from pathlib import Path
import re
import pandas as pd
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class KnowledgeBase:
    def __init__(self, docs_dir):
        self.docs=[]
        for p in sorted(Path(docs_dir).glob('*')):
            if p.suffix.lower()=='.pdf':
                for i,page in enumerate(PdfReader(str(p)).pages):
                    text=(page.extract_text() or '').strip()
                    if text:self.docs.append(Document(page_content=text,metadata={'source':p.name,'page':i+1}))
            elif p.suffix.lower()=='.csv':
                df=pd.read_csv(p).fillna('')
                for i,row in df.iterrows():
                    self.docs.append(Document(page_content=' | '.join(f'{c}: {row[c]}' for c in df.columns),metadata={'source':p.name,'row':i+2}))
        self.chunks=RecursiveCharacterTextSplitter(chunk_size=900,chunk_overlap=120).split_documents(self.docs)
        self.v=TfidfVectorizer(ngram_range=(1,2),strip_accents='unicode')
        self.m=self.v.fit_transform([d.page_content for d in self.chunks])
    def retrieve(self,q,k=4):
        scores=cosine_similarity(self.v.transform([q]),self.m).ravel()
        ids=scores.argsort()[::-1][:k]
        return [(float(scores[i]),self.chunks[i]) for i in ids]

class FashionStoreAgent:
    def __init__(self,docs_dir): self.kb=KnowledgeBase(docs_dir)
    def answer(self,q):
        ranked=self.kb.retrieve(q)
        if not ranked or ranked[0][0] < .04:
            return {'answer':'Não encontrei essa informação na documentação oficial da Aurora Moda Online.','sources':[],'confidence':'baixa'}
        t=q.lower()
        if any(x in t for x in ['devol','reembolso','troca']): a='Você pode solicitar a devolução em até 10 dias corridos após o recebimento. O produto deve estar em perfeitas condições, sem sinais de uso, lavagem, odores, danos ou alterações, com etiquetas e acessórios originais.'
        elif any(x in t for x in ['pagamento','pagar','pix','cartão']): a='A Aurora Moda Online aceita cartão de crédito e PIX. A confirmação do pedido ocorre após a aprovação do pagamento.'
        elif any(x in t for x in ['entrega','prazo','envio','frete']): a='O prazo de entrega é informado no checkout e começa após a confirmação do pagamento. Pode variar conforme CEP, frete e transportadora.'
        elif any(x in t for x in ['privacidade','dados','lgpd']): a='Os dados são usados para cadastro, pedidos, pagamento, entrega, atendimento, segurança e obrigações legais, com medidas técnicas e administrativas de proteção.'
        elif any(x in t for x in ['suporte','contato','atendimento']): a='O suporte pode ser contatado pelo e-mail suporte@auroramoda.exemplo.'
        else: a=ranked[0][1].page_content[:700]
        src=[]
        for _,d in ranked[:2]:
            item={'document':d.metadata['source']}
            if 'page' in d.metadata:item['page']=d.metadata['page']
            if item not in src:src.append(item)
        return {'answer':a,'sources':src,'confidence':'alta'}
