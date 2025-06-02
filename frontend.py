# frontend.py

import backend
import youtube_transcript_api


import streamlit as st
google_api_key = st.secrets["GOOGLE_API_KEY"]
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load environment variables
#load_dotenv()
#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.title("🎬 YouTube Video Transcript Q&A (Chat Mode)")

# --- Session State Initialization ---
if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None
if 'chunks' not in st.session_state:
    st.session_state['chunks'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'video_processed' not in st.session_state:
    st.session_state['video_processed'] = False



# --- Video Processing Section ---
with st.sidebar:
    st.header("Step 1: Process a YouTube Video")

    # Use session_state to store the input and processed URL
    if 'youtube_url' not in st.session_state:
        st.session_state.youtube_url = ""
    if 'processed_url' not in st.session_state:
        st.session_state.processed_url = ""
    if 'input_disabled' not in st.session_state:
        st.session_state.input_disabled = False    

    # Input field for YouTube URL, value is from session_state
    youtube_url = st.text_input(
        "Enter YouTube Video URL:",
        value=st.session_state.youtube_url,
        key="youtube_url_input", 
        disabled=st.session_state.input_disabled
    )

    if st.button("Process Video"):
        # Disable and clear the input field immediately
        #st.session_state.input_disabled = True
        #st.session_state.youtube_url = ""
        
        if not youtube_url:
            st.warning("Please enter the YouTube URL.")
        elif not GOOGLE_API_KEY:
            st.error("Google API Key not found.")
        else:
            with st.spinner("Processing..."):
                result = backend.process_youtube_video(youtube_url, GOOGLE_API_KEY)
            if "error" in result:
                st.error(result["error"])
                st.session_state['retriever'] = None
                st.session_state['chunks'] = None
                st.session_state['chat_history'] = []
                st.session_state['video_processed'] = False
                st.session_state['processed_url'] = ""
            else:
                st.success("Transcript processed and embeddings generated!")
                st.write("First chunk preview:")
                st.write(result["chunks"][0].page_content)
                st.session_state['retriever'] = result['retriever']
                st.session_state['chunks'] = result['chunks']
                st.session_state['chat_history'] = []
                st.session_state['video_processed'] = True
                # Store the processed URL in session_state
                st.session_state['processed_url'] = youtube_url

            # Clear the input field after processing
            st.session_state.youtube_url = ""
            #st.rerun()  # Refresh to update the UI

    # Show the processed YouTube URL outside the input field
    #if st.session_state['processed_url']:
        #st.markdown(f"**Processed YouTube URL:** [{st.session_state['processed_url']}]({st.session_state['processed_url']})")


# --- Chat Section ---
if st.session_state['video_processed'] and st.session_state['retriever'] is not None:
    st.header("Step 2: Chat with the Video Transcript")
    
# Show the processed YouTube URL outside the input field
    if st.session_state['processed_url']:
        st.markdown(f"**Processed YouTube URL:** [{st.session_state['processed_url']}]({st.session_state['processed_url']})")

    # Display chat history
    for message in st.session_state['chat_history']:
        role = message['role']
        content = message['content']
        with st.chat_message(role):
            st.markdown(content)

    # Chat input for user's question
    user_input = st.chat_input("Ask a question about the video transcript...")

    if user_input:
        # Show user's message in chat
        st.session_state['chat_history'].append({'role': 'user', 'content': user_input})

        # Retrieve relevant context
        retriever = st.session_state['retriever']
        docs = retriever.get_relevant_documents(user_input)
        context_text = "\n".join([doc.page_content for doc in docs])

        # Prepare prompt
        prompt = PromptTemplate(
            template="""
            You are a helpful assistant.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.
            {context}
            Question: {question}
            """,
            input_variables=['context', 'question']
        )
        final_prompt = prompt.invoke({"context": context_text, "question": user_input})

        # LLM answer
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        answer = llm.invoke(final_prompt)
        answer_content = answer.content.strip()

        # Show assistant's message in chat
        st.session_state['chat_history'].append({'role': 'assistant', 'content': answer_content})

        # Rerun to display the new messages
        st.rerun()

else:
    st.info("Please process a YouTube video first using the sidebar.")

