import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────────────────────
# Streamlit UI Setup
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Summarizer", page_icon="📄", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>📄 LangChain: Smart Summarizer</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Summarize YouTube videos or websites using LLMs 🔍</p>",
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────
# Sidebar - Branding and Inputs
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    groq_api_key = st.text_input("🔑 Enter Groq API Key", type="password")
    selected_model = st.selectbox("🤖 Choose Model", [
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768"
    ])
    st.markdown("---")
    st.markdown("Made with ❤️ using [LangChain](https://www.langchain.com/) and [Groq](https://console.groq.com/)")

# ─────────────────────────────────────────────────────────────
# Prompt Templates for map_reduce
# ─────────────────────────────────────────────────────────────
map_prompt_template = """
Summarize the following chunk of content:
{text}
"""
combine_prompt_template = """
Combine all the following partial summaries into a cohesive summary of around 300 words:
{text}
"""

map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

# ─────────────────────────────────────────────────────────────
# URL Input
# ─────────────────────────────────────────────────────────────
st.markdown("### 🔗 Enter a YouTube or Website URL to Summarize:")
generic_url = st.text_input("URL", placeholder="https://...", label_visibility="collapsed")

st.markdown(
    "<small>Supported: YouTube URLs or public websites (avoid login-restricted pages).</small>",
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────
# Main Logic on Button Press
# ─────────────────────────────────────────────────────────────
if st.button("🚀 Generate Summary"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("🚫 Please enter both a valid URL and API key.")
    else:
        try:
            with st.spinner("⏳ Loading and summarizing content..."):
                # Load website or YouTube content
                if "youtube.com" in generic_url.lower():
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                        }
                    )
                docs = loader.load()

                # Split text into manageable chunks
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                split_docs = splitter.split_documents(docs)

                # Initialize model
                llm = ChatGroq(model=selected_model, groq_api_key=groq_api_key)

                # Run summarization chain
                chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce",
                    map_prompt=map_prompt,
                    combine_prompt=combine_prompt,
                    verbose=False
                )
                output_summary = chain.run(split_docs)

                # Display result
                st.success("✅ Summary Generated Successfully")
                st.markdown("### ✨ Summary:")
                st.write(output_summary)

        except Exception as e:
            st.error("⚠️ An error occurred:")
            st.exception(e)

# Optional Footer
st.markdown("---")
st.markdown("<center><small>🚀 Built for quick summarization by LangChain + Groq</small></center>", unsafe_allow_html=True)
