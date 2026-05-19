import streamlit as st
import sounddevice as sd
import numpy as np
import whisper
import queue
import time
from deep_translator import GoogleTranslator
from gtts import gTTS
import io

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="Pinnacle AI Translator",
    page_icon="??",
    layout="centered"
)

# Custom Professional UI Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
    .status-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #F3F4F6;
        border-left: 5px solid #3B82F6;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Pinnacle Polyglot AI App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Speech translation powered by Whisper & Deep-Translator</div>', unsafe_allow_html=True)

# --- Resource Cached Initializations ---
@st.cache_resource
def load_whisper_engine():
    model = whisper.load_model("base")
    fake_audio = np.zeros(int(16000 * 0.5), dtype=np.float32)
    _ = model.transcribe(fake_audio, fp16=False, language="en")
    return model

model = load_whisper_engine()

# Supported Target Languages Configuration Map
LANGUAGE_MAP = {
    "Spanish ????": {"code": "es"},
    "Hindi ????": {"code": "hi"},
    "Bengali ????": {"code": "bn"},
    "Japanese ????": {"code": "ja"},
    "Chinese (Simplified) ????": {"code": "zh-CN"}
}

# --- State Management Instantiation ---
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "translation" not in st.session_state:
    st.session_state.translation = ""
if "status" not in st.session_state:
    st.session_state.status = "Ready to record."

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

# --- UI Controls Layout ---
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Configuration")
    target_lang_label = st.selectbox("Select Target Language", list(LANGUAGE_MAP.keys()))
    selected_lang_data = LANGUAGE_MAP[target_lang_label]
    recording_duration = st.slider("Max Record Windows (Seconds)", min_value=3, max_value=12, value=6)
    record_clicked = st.button("?? Start Recording Session", type="primary")

# --- Streamlit Session Execution Logic ---
if record_clicked:
    st.session_state.status = "Initializing microphone hardware..."
    status_placeholder = st.empty()
    status_placeholder.markdown(f'<div class="status-box"><b>Status:</b> {st.session_state.status}</div>', unsafe_allow_html=True)
    
    while not audio_queue.empty():
        try: audio_queue.get_nowait()
        except queue.Empty: break
            
    sr = 16000
    chunk_duration = 0.2
    frames_per_chunk = int(sr * chunk_duration)
    recorded_chunks = []
    
    ambient_energies = []
    with sd.InputStream(samplerate=sr, channels=1, callback=audio_callback, blocksize=int(sr * 0.2), dtype='float32'):
        time.sleep(0.4)
        while not audio_queue.empty():
            chunk = audio_queue.get()
            ambient_energies.append(np.sqrt(np.mean(chunk**2)))
            
    silence_threshold = max(np.mean(ambient_energies) * 2.5, 0.004) if ambient_energies else 0.015

    st.session_state.status = "?? Speak now! The engine is tracking speech active patterns..."
    status_placeholder.markdown(f'<div class="status-box"><b>Status:</b> {st.session_state.status}</div>', unsafe_allow_html=True)
    
    speech_detected = False
    silent_chunks_count = 0
    max_silence_chunks = int(1.2 / chunk_duration)
    start_time = time.time()
    
    with sd.InputStream(samplerate=sr, channels=1, callback=audio_callback, blocksize=frames_per_chunk, dtype='float32'):
        while (time.time() - start_time) < recording_duration:
            try:
                chunk = audio_queue.get(timeout=0.4)
                recorded_chunks.append(chunk)
                rms = np.sqrt(np.mean(chunk**2))
                
                if rms > silence_threshold:
                    speech_detected = True
                    silent_chunks_count = 0
                else:
                    if speech_detected:
                        silent_chunks_count += 1
                        
                if speech_detected and silent_chunks_count >= max_silence_chunks:
                    st.session_state.status = "?? Speech cadence pause caught. Running translation..."
                    break
            except queue.Empty:
                continue
                
        if not speech_detected and (time.time() - start_time) >= recording_duration:
            st.session_state.status = "?? Session timeout. No clear audio patterns registered."

    status_placeholder.markdown(f'<div class="status-box"><b>Status:</b> {st.session_state.status}</div>', unsafe_allow_html=True)

    if speech_detected and len(recorded_chunks) > 0:
        raw_audio = np.concatenate(recorded_chunks).flatten()
        max_val = np.max(np.abs(raw_audio))
        if max_val > 0:
            audio_signal = raw_audio / max_val
            result = model.transcribe(audio_signal, fp16=False, language="en")
            st.session_state.transcription = result["text"].strip()
            
            if st.session_state.transcription and "thank you" not in st.session_state.transcription.lower()[:10]:
                try:
                    translator = GoogleTranslator(source='en', target=selected_lang_data["code"])
                    st.session_state.translation = translator.translate(st.session_state.transcription)
                    st.session_state.status = "? Process completely generated."
                except Exception as ex:
                    st.session_state.translation = "Translation network failure."
                    st.session_state.status = f"? Error: {ex}"
            else:
                st.session_state.transcription = "[Empty or unreadable audio]"
                st.session_state.translation = ""
        else:
            st.session_state.status = "? Hardware signal zero value error."
            
    status_placeholder.markdown(f'<div class="status-box"><b>Status:</b> {st.session_state.status}</div>', unsafe_allow_html=True)

# --- Display Layout Results Panel ---
with right_col:
    st.subheader("Output Interface")
    st.text_area("Captured English Input Text:", value=st.session_state.transcription, height=80, disabled=True)
    st.text_area(f"Translated Text ({target_lang_label}):", value=st.session_state.translation, height=80, disabled=True)
    
    if st.session_state.translation:
        try:
            with st.spinner("Synthesizing fluent audio channels..."):
                tts = gTTS(text=st.session_state.translation, lang=selected_lang_data["code"].split('-'))
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                st.audio(fp, format="audio/mp3")
        except Exception as e:
            st.error(f"Voice generation failed: {e}")
