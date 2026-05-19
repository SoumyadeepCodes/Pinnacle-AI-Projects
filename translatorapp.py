import sounddevice as sd
import numpy as np
import whisper
import pyttsx3
import queue
import time
from deep_translator import GoogleTranslator

# --- Model & API Setup ---
print("⏳ Loading Whisper Base Model (Please wait)...")
model = whisper.load_model("base")
translator = GoogleTranslator(source='en', target='spanish')

# --- WARM-UP CYCLE ---
print("🔥 Warming up inference engines...")
fake_audio = np.zeros(int(16000 * 0.5), dtype=np.float32)
_ = model.transcribe(fake_audio, fp16=False, language="en") 
print("🚀 Engine ready.")

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """Continuously injects audio chunks into the processing queue."""
    audio_queue.put(indata.copy())

def speak_spanish(text):
    """Voice engine with cross-platform fallback."""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 145)
        voices = engine.getProperty('voices')
        
        for v in voices:
            if "spanish" in v.name.lower() or (hasattr(v, 'languages') and any("es" in str(lang).lower() for lang in v.languages)):
                engine.setProperty('voice', v.id)
                break
        
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"❌ Audio Output Error: {e}")

def calibrate_noise_floor(sr=16000, duration=0.5):
    """
    Listens to the room for 0.5 seconds before starting to establish a 
    dynamic noise threshold tailored exactly to the user's current environment.
    """
    print("🤫 Calibrating mic floor... Keep quiet for a split second...")
    # Flush queue
    while not audio_queue.empty():
        try: audio_queue.get_nowait()
        except queue.Empty: break
            
    chunks_to_read = int((sr * duration) / int(sr * 0.2)) # Calculate blocks
    ambient_energies = []
    
    with sd.InputStream(samplerate=sr, channels=1, callback=audio_callback, blocksize=int(sr * 0.2), dtype='float32'):
        for _ in range(max(2, chunks_to_read)):
            try:
                chunk = audio_queue.get(timeout=0.8)
                rms = np.sqrt(np.mean(chunk**2))
                ambient_energies.append(rms)
            except queue.Empty:
                pass
                
    if len(ambient_energies) == 0:
        return 0.015 # Safe fallback if hardware lags out
        
    mean_noise = np.mean(ambient_energies)
    # Dynamic threshold is 2.5x the average background hum or at least a baseline minimum of 0.003
    dynamic_threshold = max(mean_noise * 2.5, 0.003)
    return dynamic_threshold

def capture_phrase(sr=16000, max_silence_duration=1.2):
    """
    Listens streamingly using an adaptive noise gate.
    """
    # 1. Calibrate baseline dynamically every loop execution
    silence_threshold = calibrate_noise_floor(sr=sr)
    
    # 2. Clear out any calibration residue chunks
    while not audio_queue.empty():
        try: audio_queue.get_nowait()
        except queue.Empty: break

    print("🎤 Listening... (Speak now and pause to translate)")
    
    chunk_duration = 0.2  # 200ms blocks
    frames_per_chunk = int(sr * chunk_duration)
    recorded_chunks = []
    
    silent_chunks_count = 0
    speech_detected = False
    start_time = time.time()
    
    with sd.InputStream(samplerate=sr, channels=1, callback=audio_callback, blocksize=frames_per_chunk, dtype='float32'):
        while True:
            try:
                chunk = audio_queue.get(timeout=0.5)
                recorded_chunks.append(chunk)
                
                # Calculate energy volume
                rms = np.sqrt(np.mean(chunk**2))
                
                if rms > silence_threshold:
                    if not speech_detected:
                        speech_detected = True
                    silent_chunks_count = 0  # Voice active
                else:
                    if speech_detected:
                        silent_chunks_count += 1
                
                # Dynamic silence breaker
                if speech_detected and (silent_chunks_count * chunk_duration >= max_silence_duration):
                    print("⏹️ Speech ended. Processing...")
                    break
                    
                # Timeout safeguard 
                if (time.time() - start_time) > 10.0:
                    if speech_detected:
                        print("⏹️ Max limits reached. Processing speech captured so far...")
                    break
                    
            except queue.Empty:
                if (time.time() - start_time) > 10.0:
                    break
                continue

    if not speech_detected or len(recorded_chunks) == 0:
        return None
        
    return np.concatenate(recorded_chunks).flatten()

def run_session():
    fs = 16000
    raw_audio = capture_phrase(sr=fs)
    
    if raw_audio is None:
        print("❌ No speech detected. Please speak louder or check your input level configuration.")
        return

    # Peak Normalization
    max_val = np.max(np.abs(raw_audio))
    if max_val > 0:
        audio_signal = raw_audio / max_val
    else:
        return

    try:
        result = model.transcribe(audio_signal, fp16=False, language="en")
        en_text = result["text"].strip()
    except Exception as e:
        print(f"❌ Local Inference Failure: {e}")
        return

    if en_text and len(en_text) > 2 and "thank you" not in en_text.lower()[:10]:
        print(f"✅ You: {en_text}")
        print("🌐 Translating...")
        try:
            es_text = translator.translate(en_text)
            print(f"🇪🇸 Spanish: {es_text}")
            speak_spanish(es_text)
        except Exception as e:
            print(f"❌ Network Translation Failure: {e}")
    else:
        print("❓ Sound captured but text resolution failed standard filters.")

if __name__ == "__main__":
    print("🚀 Auto-Sensing Translation Engine Engaged. Press Ctrl+C to stop.")
    while True:
        try:
            run_session()
        except KeyboardInterrupt:
            print("\nSession wrapped up successfully.")
            break