import streamlit as st
import requests

API_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("📚 Lecture RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض المحادثة السابقة
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("اكتب سؤالك هنا / Ask your question")

if question:
    # رسالة المستخدم
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    with st.chat_message("user"):
        st.markdown(question)

    # رسالة المساعد
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=120
                )
            except Exception as e:
                st.error(f"API error: {e}")
                answer = "⚠️ Failed to reach the server."
            else:
                if response.status_code != 200:
                    st.error(response.text)
                    answer = "⚠️ Server returned an error."
                else:
                    data = response.json()
                    answer = data.get("answer", "No answer returned.")

                    st.markdown(answer)

                    not_found_msg = "Not found in the provided lecture material."

                    if "citations" in data and data["citations"] and answer.strip() != not_found_msg:
                        with st.expander("📌 المراجع والمصادر (Sources)"):
                            unique_citations = {}
                            for key, info in data["citations"].items():
                                source_name = info.get("source", "Unknown")
                                page_num = info.get("page", "-")
                                cite_key = f"{source_name}_page_{page_num}"
                                if cite_key not in unique_citations:
                                    unique_citations[cite_key] = (source_name, page_num)

                            for src, pg in unique_citations.values():
                                st.markdown(f"📄 **الملف:** `{src}` | 📖 **الصفحة:** `{pg}`")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
