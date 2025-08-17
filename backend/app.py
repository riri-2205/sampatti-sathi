from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
import os
import openai
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import langdetect
import base64
import io

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.urandom(24)  # Secure key for sessions
app.config['SESSION_TYPE'] = 'filesystem'   # Store sessions on filesystem
Session(app)

# Initialize RAG
def init_rag():
    try:
        loader = JSONLoader(
            file_path='knowledge_base.json',
            jq_schema='.[] | {content: .description + " Eligibility: " + (.eligibility | tostring), metadata: {name: .name}}',
            text_content=False
        )
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # Better context retention
        texts = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        if not os.path.exists('vector_db/index.faiss'):
            vectorstore = FAISS.from_documents(texts, embeddings)
            vectorstore.save_local('vector_db')
        else:
            vectorstore = FAISS.load_local('vector_db', embeddings, allow_dangerous_deserialization=True)
        return vectorstore
    except Exception as e:
        print(f"Error initializing RAG: {str(e)}")
        raise

vectorstore = init_rag()
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)  # Conversational flexibility

# Define a custom prompt for the conversational chain
prompt_template = """You are a proactive, friendly conversational agent assisting farmers in finding eligible government schemes in English, Hindi, Tamil, or Kannada. Follow these steps:
1. Review the chat history and user's current input to extract key details: land size (in hectares), crop, state, and farmer category (e.g., small, marginal).
2. If any detail is missing or unclear, ask a specific, relevant follow-up question based on the history (e.g., 'What state are you in?' or 'How large is your farm?').
3. Use the provided context from the knowledge base to inform partial responses as details accumulate, but avoid a full scheme list until all details (land size, crop, state, category) are confirmed.
4. Once all details are provided, use the context to suggest relevant schemes concisely.
5. Update the chat history with the current input and your response, then encourage further interaction.
6. Respond only in English, Hindi, Tamil, or Kannada based on the detected language, and avoid other languages.

Chat History:
{chat_history}
Context from knowledge base: {context}
User's input: {question}
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=prompt_template
)

# Set up ConversationalRetrievalChain with memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    session_id="farmer_chat"  # Unique session ID
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # Top 3 relevant docs
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    verbose=True  # For debugging
)

def detect_language(text):
    try:
        detected_lang = langdetect.detect(text)
        # Restrict to English, Hindi, Tamil, Kannada
        language_map = {'en': 'en', 'hi': 'hi', 'ta': 'ta', 'kn': 'kn'}
        return language_map.get(detected_lang, 'en')  # Default to English if unsupported
    except:
        return 'en'  # Fallback to English on error

def translate_text(text, target_lang='en'):
    try:
        # Restrict translation to supported languages
        supported_langs = {'en', 'hi', 'ta', 'kn'}
        if target_lang not in supported_langs:
            target_lang = 'en'
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Translate to {target_lang}: {text}"}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text  # Return original text on failure

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'text': 'No audio file provided.', 'audio': ''}), 400

        audio_data = request.files['audio'].read()
        language = request.form.get('language', 'auto')

        # STT with OpenAI Whisper
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.webm"
        try:
            with audio_file:
                transcript_response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    timeout=30
                )
            transcript = transcript_response.text
        except openai.InternalServerError as e:
            return jsonify({'text': 'Sorry, there was an issue processing your audio. Please try again later.', 'audio': ''}), 500
        except Exception as e:
            return jsonify({'text': f'Error transcribing audio: {str(e)}', 'audio': ''}), 500

        detected_lang = detect_language(transcript)

        # Translate to English for processing if needed (only supported languages)
        if detected_lang != 'en':
            transcript_en = translate_text(transcript, 'en')
        else:
            transcript_en = transcript

        # AI Reasoning with Conversational RAG
        result = qa_chain({"question": transcript_en})
        response_text_en = result['answer']

        # Translate back to detected language (only supported)
        response_text = translate_text(response_text_en, detected_lang)

        # TTS with OpenAI
        try:
            tts_response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=response_text
            )
            audio_bytes = tts_response.content
        except Exception as e:
            return jsonify({'text': f'Error generating audio: {str(e)}', 'audio': ''}), 500

        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        return jsonify({'text': response_text, 'audio': audio_base64})

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'text': 'An unexpected error occurred. Please try again.', 'audio': ''}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')