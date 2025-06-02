# backend.py

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Dict, Any




def get_youtube_transcript(url: str) -> str:
    """
    Given a YouTube video URL, returns the transcript as a plain text string.
    If transcripts are disabled, returns an error message.
    """
    try:
        # Extract the video ID from the URL
        video_id = url.split("v=")[1].split("&")[0]
        # Fetch the transcript (English)
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        # Flatten to plain text
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript
    except TranscriptsDisabled:
        return "No captions available for this video."
    except Exception as e:
        return f"Error: {e}"

# Usage example:
#url = "https://www.youtube.com/watch?v=p4pHsuEf4Ms"
#transcript = get_youtube_transcript(url)
#print(transcript)


def process_youtube_video(url: str, google_api_key: str) -> Dict[str, Any]:
    """
    Process YouTube video: extract transcript, split, embed, and create retriever.
    Returns a dictionary with retriever and chunks, or an error message.
    """
    transcript = get_youtube_transcript(url)
    if transcript.startswith("No captions available") or transcript.startswith("Error:"):
        return {"error": transcript}

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    #print(len(chunks))

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    return {
        "transcript": transcript,
        "chunks": chunks,
        "vector_store": vector_store,
        "retriever": retriever
    }

def get_context_from_question(retriever, question: str, k: int = 5) -> str:
    """
    Retrieve relevant context chunks for a given question.
    """
    docs = retriever.get_relevant_documents(question)
    return "\n".join([doc.page_content for doc in docs])

# Example usage for testing
if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=p4pHsuEf4Ms"
    print(fetch_youtube_transcript(url))
