import json
import numpy as np
import pyaudio
import wave
from faster_whisper import WhisperModel

# Initialize Whisper model (optimized version)
model_size = "base.en"
model = WhisperModel(model_size, compute_type="int8", device="cpu")

# Audio recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 16kHz recommended for Whisper
CHUNK = 1024
RECORD_SECONDS = 10  # Adjust for longer recordings
OUTPUT_WAVE_FILE = "recorded_audio.wav"

# Initialize PyAudio
audio = pyaudio.PyAudio()

# JSON structure for periodontal chart
chart_data = {
    "patient_id": "12345",
    "exam_date": "2024-12-24",
    "periodontal_chart": []
}


def record_audio():
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    print("Recording... Speak Now")
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording Finished")

    stream.stop_stream()
    stream.close()

    # Save as WAV
    with wave.open(OUTPUT_WAVE_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))


def transcribe_audio():
    print("Transcribing...")
    segments, info = model.transcribe(OUTPUT_WAVE_FILE, beam_size=5)

    print("Detected language:", info.language)
    
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        fill_chart_data(segment.text)


def fill_chart_data(transcribed_text):
    """
    Fills the periodontal chart based on keywords in transcription.
    Example: "Pocket depth 4 millimeters on tooth 22"
    """
    keywords = ["pocket", "depth", "tooth", "gingival"]

    # Simple rule-based parser for the demo
    words = transcribed_text.lower().split()
    if "tooth" in words:
        tooth_index = words.index("tooth")
        tooth_number = words[tooth_index + 1] if tooth_index + 1 < len(words) else "unknown"

        pocket_depth = next((w for w in words if w.isdigit()), "N/A")

        chart_entry = {
            "tooth": tooth_number,
            "pocket_depth_mm": pocket_depth,
            "notes": transcribed_text
        }

        chart_data["periodontal_chart"].append(chart_entry)
        print(f"Updated Chart: {chart_entry}")

    # Save JSON dynamically
    with open("chart_data.json", "w") as f:
        json.dump(chart_data, f, indent=4)


if __name__ == "__main__":
    record_audio()
    transcribe_audio()

    print("Final Chart Data:")
    print(json.dumps(chart_data, indent=4))
