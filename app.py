"""Docora: conversa com documentos e fontes ao lado das respostas."""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

# Prioriza o pacote deste repositório no Streamlit Community Cloud.
_REPO_ROOT = Path(__file__).resolve().parent
if sys.path[:1] != [str(_REPO_ROOT)]:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from docora.assistant import ProviderError, answer_question
from docora.retrieval import KnowledgeIndex, demo_chunks, ingest_bytes


def server_setting(name):
    """Lê o ambiente ou os segredos opcionais, inclusive quando não há arquivo."""
    if value := os.environ.get(name):
        return value
    try:
        return st.secrets.get(name, "")
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        return ""


def load_example():
    """Seleciona a base demonstrativa sem recriar o índice ou apagar a conversa."""
    st.session_state["collection"] = "Loja demonstrativa"
    st.session_state["example_loaded"] = True


st.set_page_config(page_title="Docora · Conversas com fontes", page_icon="💬", layout="wide")
st.markdown("""<style>
.stApp{background:#fcfaf7;color:#303044}.block-container{max-width:1450px;padding-top:4rem}
h1,h2,h3{letter-spacing:-.035em;color:#353047}
.brand{font-size:25px;font-weight:800;color:#6848a3;letter-spacing:-1px}
.brand span{font-size:12px;letter-spacing:1px;color:#90879e;font-weight:500;margin-left:18px}
.welcome{padding:35px 0 20px}.welcome h2{font-family:Georgia,serif;font-weight:400;font-size:40px}
.welcome p{color:#81798a;max-width:530px;font-size:17px}
[data-testid="stChatMessage"]{border-radius:18px;background:#f2eef7;padding:20px;margin:12px 0}
[data-testid="stChatInput"]{border-radius:18px;border-color:#d1c3df}
@media(max-width:700px){.brand span{display:block;margin-left:0}.welcome h2{font-size:30px}}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="brand">Docora<span>SEU ESPAÇO DE CONSULTA</span></div>', unsafe_allow_html=True)
st.divider()
conversation, references = st.columns([2.15, 1], gap="large")

@st.cache_resource
def embedding_resource(_cache_bust=2):
    """Compartilha um único encoder local entre a demo e as bibliotecas enviadas."""
    from docora.embeddings import load_embedder
    return load_embedder()


def _build_demo_index(retriever, reranker):
    """Monta o índice da demo com fallback lexical se o híbrido falhar no boot."""
    retriever = str(retriever or "hybrid")
    reranker = str(reranker or "none")
    if retriever not in {"hybrid", "bm25", "tfidf"}:
        retriever = "hybrid"
    if reranker not in {"none", "rules", "crossencoder"}:
        reranker = "none"
    try:
        if retriever == "hybrid":
            model, reason = embedding_resource()
            index = KnowledgeIndex(demo_chunks(), retriever=retriever, embedder=model, reranker=reranker)
            if reason:
                index.fallback_reason = reason
            return index
        return KnowledgeIndex(demo_chunks(), retriever=retriever, embedder=None, reranker=reranker)
    except Exception as error:
        index = KnowledgeIndex(demo_chunks(), retriever="tfidf", embedder=None, reranker="none")
        index.fallback_reason = f"Falha ao iniciar a busca ({type(error).__name__}); usando TF-IDF."
        return index


@st.cache_resource
def demo_index(retriever, reranker, _cache_bust=2):
    """Cria o índice demonstrativo compartilhado pelo cache de recursos do Streamlit."""
    return _build_demo_index(retriever, reranker)

with references:
    sources_slot = st.container()
    with st.container(border=True):
        st.subheader("Sua biblioteca")
        retriever_names = {"hybrid": "Busca híbrida", "bm25": "BM25", "tfidf": "TF-IDF"}
        configured_retriever = server_setting("DOCORA_RETRIEVER") or "hybrid"
        configured_retriever = configured_retriever if configured_retriever in retriever_names else "hybrid"
        retriever = st.selectbox("Método de busca", list(retriever_names),
                                index=list(retriever_names).index(configured_retriever),
                                format_func=retriever_names.get, key="retriever")
        reranker_names = {"none": "Sem reordenação", "rules": "Regras locais", "crossencoder": "Cross-encoder local"}
        reranker = server_setting("DOCORA_RERANKER") or "none"
        reranker = reranker if reranker in reranker_names else "none"
        collection = st.radio("Documentos usados na consulta", ["Loja demonstrativa", "Meus documentos"], key="collection")
        st.button("Carregar exemplo", on_click=load_example, width="stretch")
        if st.session_state.pop("example_loaded", False):
            st.success("Exemplo carregado: Lume Casa. Escolha uma pergunta rápida ou escreva sua dúvida.")
        index = None
        if collection == "Loja demonstrativa":
            index = demo_index(str(retriever), str(reranker))
            st.caption("Lume Casa · loja fictícia. Conteúdo demonstrativo.")
        else:
            uploaded = st.file_uploader("Enviar documentos", type=["md", "txt", "pdf"], accept_multiple_files=True)
            st.caption("Até 5 arquivos de 2 MB. PDF com texto selecionável, até 30 páginas. Processamento em memória nesta sessão.")
            if not uploaded:
                st.info("Envie documentos para começar a consulta.")
            elif len(uploaded) > 5:
                st.error("Selecione no máximo 5 documentos.")
            else:
                upload_key = sha256((retriever + reranker).encode() + b"".join(f.name.encode() + f.getvalue() for f in uploaded)).hexdigest()
                try:
                    if st.session_state.get("upload_key") != upload_key:
                        chunks = [c for f in uploaded for c in ingest_bytes(f.name, f.getvalue())]
                        if retriever == "hybrid":
                            st.info("A busca híbrida calcula os vetores dos documentos nesta sessão. Isso pode levar alguns instantes.")
                            model, reason = embedding_resource()
                            with st.spinner("Preparando a busca dos documentos…"):
                                upload_index = KnowledgeIndex(chunks, retriever=retriever, embedder=model, reranker=reranker)
                            if reason:
                                upload_index.fallback_reason = reason
                        else:
                            upload_index = KnowledgeIndex(chunks, retriever=retriever, reranker=reranker)
                        st.session_state["upload_index"] = upload_index
                        st.session_state["upload_key"] = upload_key
                    index = st.session_state["upload_index"]
                except (ValueError, TypeError, RuntimeError) as error:
                    st.error(str(error))
        if index:
            st.caption(f"Busca ativa: {retriever_names[index.active_retriever]}")
            st.caption(f"Reordenação ativa: {reranker_names[index.active_reranker]}")
            if index.reranker_fallback_reason:
                st.caption(index.reranker_fallback_reason)
            if index.fallback_reason:
                st.caption("Modelo local ausente ou indisponível. A consulta usa TF-IDF.")
            documents = sorted({c.source for c in index.chunks})
            st.caption(f"{len(documents)} documentos disponíveis · {len(index.chunks)} trechos")
            with st.expander("Explorar documentos"):
                for document in documents:
                    st.markdown(f"**{document}**")
                    for chunk in (c for c in index.chunks if c.source == document):
                        with st.expander(chunk.section):
                            st.text(chunk.text)
        api_key = server_setting("OPENAI_API_KEY")
        model = server_setting("OPENAI_MODEL")
        mode_label = "Trechos dos documentos"
        if api_key:
            mode_label = st.radio("Formato da resposta", ["Trechos dos documentos", "Síntese com IA"])
        mode = "generative" if mode_label == "Síntese com IA" else "extractive"
        configured = bool(api_key and model)
        approved = mode == "extractive"
        if mode == "generative":
            if not configured:
                st.info("A síntese com IA está indisponível porque o modelo não foi configurado no servidor. A consulta aos trechos continua disponível.")
            approved = st.checkbox("Autorizar o envio da pergunta e dos trechos recuperados para a OpenAI. A chamada pode consumir créditos da API.")
        else:
            st.caption("Consulta local: respostas reproduzidas dos documentos, com referências.")

# Trocar base ou modo inicia outro histórico para não apresentar fontes de um contexto anterior.
context_key = sha256((mode + retriever + reranker + ("|".join(c.id for c in index.chunks) if index else "empty")).encode()).hexdigest()
if st.session_state.get("context_key") != context_key:
    st.session_state["history"] = []
    st.session_state["context_key"] = context_key
history = st.session_state["history"]
disabled = index is None or (mode == "generative" and (not configured or not approved))
with conversation:
    top, clear = st.columns([4, 1])
    top.caption("CONVERSA · " + ("BASE DEMONSTRATIVA" if collection == "Loja demonstrativa" else "SEUS DOCUMENTOS"))
    if clear.button("Nova conversa", disabled=not history):
        st.session_state["history"] = []
        st.rerun()
    question = None
    if not history:
        st.markdown('<div class="welcome"><h2>O que você quer descobrir?</h2><p>Pergunte aos seus documentos. A resposta vem acompanhada dos trechos que sustentam a informação.</p></div>', unsafe_allow_html=True)
        if collection == "Loja demonstrativa":
            for label, suggestion in [
                ("Prazo de entrega", "Qual é o prazo de entrega para São Paulo?"),
                ("Trocas e devoluções", "Como posso devolver um produto?"),
                ("Formas de pagamento", "Quais formas de pagamento são aceitas?"),
            ]:
                if st.button(label + "  ↗", key=label, disabled=disabled, width="stretch"):
                    question = suggestion
    messages = st.container()
    typed = st.chat_input("Pergunte sobre os documentos…", disabled=disabled, max_chars=1500)
    question = typed or question
    if question and not disabled:
        try:
            with st.spinner("Consultando a biblioteca…"):
                if mode == "generative":
                    answer = answer_question(question, index, mode, model=model, api_key=api_key)
                else:
                    answer = answer_question(question, index, mode)
            history.append({"question": question, "answer": answer.to_dict()})
            st.session_state["history"] = history[-6:]
            st.rerun()
        except (ValueError, ProviderError) as error:
            st.error(str(error))
    with messages:
        for turn in history:
            with st.chat_message("user"):
                st.write(turn["question"])
            response = turn["answer"]
            with st.chat_message("assistant"):
                st.caption(mode_label)
                if response["status"] == "needs_human":
                    st.info(response["message"])
                else:
                    st.markdown(response["message"])
    st.caption("Cada pergunta consulta a base de forma independente. Histórico de até seis consultas nesta sessão.")

with sources_slot:
    if history:
        st.subheader("Fontes da conversa")
        selected_turn = st.selectbox("Ver fontes de qual pergunta?", range(len(history)), index=len(history)-1,
                                     format_func=lambda i: history[i]["question"])
        response = history[selected_turn]["answer"]
        if not response["sources"]:
            st.caption("Esta resposta não tem trechos associados.")
        for source in response["sources"]:
            position = f"página {source['page']}" if source["page"] else f"linhas {source['line_start']}–{source['line_end']}"
            with st.container(border=True):
                st.markdown(f"**[{source['reference']}] {source['section']}**")
                st.caption(f"{source['source']} · {position}")
                with st.expander("Ler trecho original"):
                    st.text(source["text"])
                    st.caption(f"Trecho {source['id']} · escore de recuperação {source['score']:.3f}")
        st.download_button("Baixar esta consulta · JSON", json.dumps(history[selected_turn], ensure_ascii=False, indent=2),
                           "docora-consulta.json", mime="application/json")
with references:
    with st.expander("Sobre esta consulta"):
        st.write("A busca pode usar TF-IDF, BM25 ou combinar BM25 com vetores de um modelo multilíngue local. Recupera até três fontes. O modo local reproduz os trechos; a síntese com IA redige uma resposta e valida seus identificadores de referência.")
        st.write("Perguntas sem evidência recebem uma mensagem de ausência de informação. O assistente não acessa pedidos individuais. PDFs digitalizados precisam de OCR antes do envio.")
        active = index.active_retriever if index else "tfidf"
        report_path = _REPO_ROOT / (f"reports/{active}_original.json" if active != "tfidf" else "reports/baseline_tfidf_original.json")
        if report_path.exists():
            report = json.loads(report_path.read_text("utf-8"))
            st.caption(f"Referência original · {retriever_names[active]}: {report['total']} perguntas avaliadas · {report['pass_rate_pct']:.1f}% dos critérios atendidos. Avaliação local; não mede generalização nem um LLM externo.")
        st.caption("API: /health, /sources e /ask. Documentação em /docs.")
        st.link_button("Projeto no GitHub", "https://github.com/GustavoCurio/docora")
