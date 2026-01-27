# =========================================================
# Fix PYTHONPATH (IMPORTANT)
# =========================================================
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# =========================================================
# Imports
# =========================================================
import streamlit as st
import requests
import tempfile
from pathlib import Path

from src.training.run_ingestion import run_ingestion


# =========================================================
# Config
# =========================================================
API_URL = "http://localhost:8000/ask"

st.set_page_config(
    page_title="Lecture RAG",
    layout="wide",
)

st.title("📚 Lecture RAG Assistant")

tab_chat, tab_train = st.tabs(["💬 Chat", "🧠 Train / Ingest"])


# =========================================================
#  TAB 1: CHAT
# =========================================================
with tab_chat:
    # حاوية لعرض الرسائل لضمان بقائها داخل التبويب
    chat_container = st.container()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # عرض المحادثة السابقة داخل الحاوية
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])


# =========================================================
#  TAB 2: TRAIN / INGEST
# =========================================================
with tab_train:
    st.header("🧠 Train / Build Knowledge Base for developer")

    st.markdown("""
    - ارفع **ملف PDF واحد**
    - أو **عدة ملفات PDF** (سيتم التعامل معها كمجلد واحد)
    """)

    uploaded_files = st.file_uploader(
        "📂 Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    chunk_size = st.number_input(
        "Chunk size",
        min_value=200,
        max_value=800,
        value=800
    )

    overlap = st.number_input(
        "Overlap",
        min_value=0,
        max_value=200,
        value=100
    )

    if st.button("🚀 Start Ingestion"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Processing documents..."):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir)

                    for f in uploaded_files:
                        file_path = tmp_path / f.name
                        file_path.write_bytes(f.read())

                    if len(uploaded_files) == 1:
                        input_path = tmp_path / uploaded_files[0].name
                    else:
                        input_path = tmp_path  

                    result = run_ingestion(
                        data_path=input_path,
                        chunk_size=chunk_size,
                        overlap=overlap
                    )

            st.success("✅ Ingestion completed successfully!")

            if isinstance(result, dict):
                st.markdown(f"""
                **Documents:** `{result.get("num_docs", "-")}`  
                **Chunks:** `{result.get("num_chunks", "-")}`  
                """)

            st.info("🔄 Restart FastAPI server to reload the new FAISS index.")


# =========================================================
#  CHAT INPUT (FIXED POSITION)
# =========================================================
# وضع المدخلات خارج الـ tabs يضمن تثبيتها في أسفل المتصفح
question = st.chat_input("اكتب سؤالك هنا / Ask your question")

if question:
    # 1. إضافة رسالة المستخدم وعرضها فوراً داخل التبويب
    st.session_state.messages.append({"role": "user", "content": question})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(question)

        # 2. طلب الرد من المساعد وعرضه
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        API_URL,
                        json={"question": question},
                        timeout=120
                    )
                    response.raise_for_status()
                    data = response.json()
                    answer = data.get("answer", "No answer returned.")
                except Exception as e:
                    data = {}
                    answer = f"❌ API error: {e}"

                st.markdown(answer)

                if data.get("citations"):
                    with st.expander("📌 Sources"):
                        seen = set()
                        for c in data["citations"].values():
                            key = (c["source"], c["page"])
                            if key not in seen:
                                seen.add(key)
                                st.markdown(f"- 📄 `{c['source']}` | page `{c['page']}`")

    # 3. حفظ رد المساعد في الـ session_state
    st.session_state.messages.append({"role": "assistant", "content": answer})