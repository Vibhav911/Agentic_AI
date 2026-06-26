import validators
import streamlit as st
from urllib.parse import urlparse, parse_qs

# LangChain updated imports
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain

# Groq updated import
from langchain_groq import ChatGroq

# YouTube transcript
from youtube_transcript_api import YouTubeTranscriptApi


# ---------------- STREAMLIT UI ---------------- #
st.set_page_config(page_title="Summarize Text from YouTube or Websites")
st.title("🔍 Summarize Text from YouTube or Websites")

with st.sidebar:
    api_key = st.text_input("Enter Groq API Key", type="password")

generic_url = st.text_input("Enter URL")


# ---------------- LLM SETUP ---------------- #
def get_llm(api_key):
    return ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.3
    )


prompt_template = """
Provide a concise and informative summary of the following content in about 300 words.

Content:
{text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)


# ---------------- HELPERS ---------------- #
def extract_video_id(url):
    parsed_url = urlparse(url)

    if "youtube.com" in parsed_url.netloc:
        return parse_qs(parsed_url.query).get("v", [None])[0]
    elif "youtu.be" in parsed_url.netloc:
        return parsed_url.path[1:]
    return None


def get_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except Exception:
        raise Exception("❌ Could not fetch transcript. Video may not have captions.")


# ---------------- MAIN LOGIC ---------------- #
if st.button("Summarize"):
    if not api_key.strip() or not generic_url.strip():
        st.error("⚠️ Please provide both API key and URL.")
    
    elif not validators.url(generic_url):
        st.error("⚠️ Invalid URL.")
    
    else:
        try:
            with st.spinner("⏳ Processing..."):
                docs = []

                # -------- YouTube -------- #
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    video_id = extract_video_id(generic_url)

                    if not video_id:
                        st.error("❌ Invalid YouTube URL.")
                        st.stop()

                    transcript_text = get_youtube_transcript(video_id)

                    docs.append(
                        Document(
                            page_content=transcript_text,
                            metadata={"source": "YouTube"}
                        )
                    )

                # -------- Website -------- #
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False
                    )
                    docs = loader.load()

                # -------- Summarization -------- #
                llm = get_llm(api_key)

                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="stuff",
                    prompt=prompt
                )

                result = chain.invoke({"input_documents": docs})
                summary = result["output_text"]

                # -------- Output -------- #
                st.success("✅ Summary generated successfully!")
                st.text_area("Summary", summary, height=300)

        except Exception as e:
            st.error("⚠️ Error during processing")
            st.exception(e)