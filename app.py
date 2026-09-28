import json
import os
from pathlib import Path

import streamlit as st
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL = "openai/gpt-oss-120b"
TOP_K = 5
MIN_RELEVANCE = 0.15

st.set_page_config(page_title="IBA Sukkur Assistant", page_icon="🎓", layout="centered")


def get_api_key():
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        key = os.environ.get("GROQ_API_KEY")
    return key


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )


@st.cache_data
def load_departments():
    path = Path(INDEX_DIR) / "manifest.json"
    if not path.exists():
        return []
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return sorted(manifest.get("departments", {}).keys())


def retrieve(db, question, department):
    kwargs = {"k": TOP_K}
    if department != "All departments":
        kwargs["filter"] = {"department": department}
        kwargs["fetch_k"] = 100
    results = db.similarity_search_with_score(question, **kwargs)

    hits = []
    for doc, dist in results:
        # embeddings are normalized, so cosine similarity = 1 - dist^2 / 2
        relevance = 1 - (float(dist) ** 2) / 2
        if relevance >= MIN_RELEVANCE:
            hits.append((doc, relevance))
    return hits


def build_prompt(question, hits):
    blocks = []
    for i, (doc, _) in enumerate(hits, start=1):
        m = doc.metadata
        header = "[{}] {} (page {})".format(i, m.get("filename"), m.get("page"))
        blocks.append(header + "\n" + doc.page_content)
    context = "\n\n".join(blocks)

    system = (
        "You are a helpful assistant for IBA Sukkur University. "
        "Answer the question using ONLY the context provided. "
        "If the context does not contain the answer, say you could not find "
        "it in the university documents. Be clear and concise. "
        "Cite the sources you used with their numbers, like [1] or [2]."
    )
    user = "Context:\n" + context + "\n\nQuestion: " + question
    return system, user


def generate_answer(client, system, user):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=1500,
        reasoning_effort="low",
    )
    return response.choices[0].message.content


def main():
    st.title("🎓 IBA Sukkur Assistant")
    st.caption("Ask questions about IBA Sukkur University documents.")

    api_key = get_api_key()
    if not api_key:
        st.error("GROQ_API_KEY is not configured. Add it to .streamlit/secrets.toml.")
        st.stop()

    if not Path(INDEX_DIR).exists() or not (Path(INDEX_DIR) / "index.faiss").exists():
        st.error("FAISS index not found in 'faiss_index/'. Run ingest.py first.")
        st.stop()

    db = load_db()
    client = Groq(api_key=api_key)

    departments = ["All departments"] + load_departments()
    department = st.selectbox("Department", departments)

    question = st.chat_input("Ask a question...")
    if not question:
        st.stop()

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            hits = retrieve(db, question, department)

        if not hits:
            st.write("I could not find relevant information in the documents.")
            st.stop()

        with st.spinner("Writing answer..."):
            system, user = build_prompt(question, hits)
            try:
                answer = generate_answer(client, system, user)
            except Exception as e:
                st.error("The answer service failed: " + str(e)[:200])
                st.stop()

        st.write(answer)

        st.markdown("**Sources**")
        for i, (doc, rel) in enumerate(hits, start=1):
            m = doc.metadata
            title = "[{}] {} · {} · p.{}".format(
                i, m.get("filename"), m.get("department"), m.get("page")
            )
            with st.expander(title):
                st.caption("Relevance: {:.0%} | {}".format(rel, m.get("source")))
                st.write(doc.page_content)


main()
