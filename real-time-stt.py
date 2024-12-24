from openai import OpenAI
import pyaudio
import os
from colorama import Fore, Style
import wave
import time
import audioop
from multiprocessing import Process, Queue


# Constants
MODEL_SIZE = "whisper-1"
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
BUFFER_SIZE = 4096
DEVICE = "cpu"
COMPUTE_TYPE = "float32"
SILENCE_THRESHOLD = 500  # Adjust as needed
SILENCE_GAP_TIMEOUT = 3  # Timeout for complete silence to stop recording
ROLLING_WINDOW = 3  # Number of chunks to average for silence detection
MIN_SOUND_THRESHOLD = 1000  # Minimum total RMS to save file
OPENAI_API_KEY = "<openai-api-key>"

openai = OpenAI(api_key=OPENAI_API_KEY)

# Colors
NEON_GREEN = Fore.GREEN
RESET_COLOR = Style.RESET_ALL


# Record Audio Until Sustained Silence
def record_until_sustained_silence(p, stream, queue):
    frames = []
    silent_start = None
    rolling_window = []
    total_rms = 0

    while True:
        data = stream.read(BUFFER_SIZE)
        frames.append(data)

        rms = audioop.rms(data, 2)  # Calculate RMS (volume level)
        total_rms += rms
        rolling_window.append(rms)

        if len(rolling_window) > ROLLING_WINDOW:
            rolling_window.pop(0)

        avg_rms = sum(rolling_window) / len(rolling_window)

        if avg_rms < SILENCE_THRESHOLD:
            if silent_start is None:
                silent_start = time.time()
            elif time.time() - silent_start > SILENCE_GAP_TIMEOUT:
                break  # Stop after sustained silence
        else:
            silent_start = None  # Reset if speech resumes

    # Only save the chunk if total RMS exceeds the minimum sound threshold
    if total_rms > MIN_SOUND_THRESHOLD:
        chunk_file = f"chunk_{int(time.time())}.wav"
        with wave.open(chunk_file, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))

        queue.put(chunk_file)
        return True
    else:
        return False


# Transcribe Audio Chunk using OpenAI Whisper
def transcribe_chunk(queue):
    while True:
        chunk_file = queue.get()
        if chunk_file == "STOP":
            break

        with open(chunk_file, "rb") as audio_file:
            response = openai.audio.transcriptions.create(
                model=MODEL_SIZE, file=audio_file, language="en", response_format="text"
            )
            transcription = response
            print(NEON_GREEN + transcription + RESET_COLOR)
        os.remove(chunk_file)


# Real-Time STT Pipeline
def run_transcription():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=BUFFER_SIZE,
    )

    queue = Queue()

    transcription_process = Process(target=transcribe_chunk, args=(queue,))
    transcription_process.start()

    try:
        print("Recording...")
        while True:
            if not record_until_sustained_silence(p, stream, queue):
                break

    except KeyboardInterrupt:
        print("Stopping... Waiting for transcription to finish.")
        queue.put("STOP")
        transcription_process.join()

    finally:
        print("Recording Complete")
        stream.stop_stream()
        stream.close()
        p.terminate()


# Run the script if executed directly
if __name__ == "__main__":
    run_transcription()
