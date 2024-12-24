import json
import asyncio
import pyaudio
import numpy as np
import wave
import time
from faster_whisper import WhisperModel

# Initialize Whisper model
model_size = "base.en"
model = WhisperModel(model_size, compute_type="int8",device="cuda")

# Audio recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 1500  # ms to detect pause
MIN_VOLUME_THRESHOLD = 200  # Minimum audio volume to consider speech

audio = pyaudio.PyAudio()

chart_data = {
    "patient_id": "12345",
    "exam_date": "2024-12-24",
    "periodontal_chart": []
}


async def record_audio_segment():
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    frames = []
    silence_start = None
    processing_tasks = []

    print("Listening... Speak Now (Ctrl + C to exit)")

    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()

            # Detect silence and segment
            if volume < MIN_VOLUME_THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif (time.time() - silence_start) * 1000 > SILENCE_THRESHOLD:
                    if len(frames) > 20:  # Process only if there was speech
                        print("... Pause Detected, Processing ...")
                        task = asyncio.create_task(process_audio_segment(frames.copy()))
                        processing_tasks.append(task)
                    frames = []
                    silence_start = None
            else:
                silence_start = None

            # Cleanup finished tasks
            processing_tasks = [t for t in processing_tasks if not t.done()]

            await asyncio.sleep(0)  # Allow other tasks to run

    except KeyboardInterrupt:
        print("Recording stopped. Final processing...")
        if len(frames) > 20:
            await process_audio_segment(frames)
        stream.stop_stream()
        stream.close()
        print("Final Chart Data:")
        print(json.dumps(chart_data, indent=4))


async def process_audio_segment(frames):
    # Save the audio segment to a temp file
    with wave.open("temp_audio.wav", 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    # Transcribe audio
    segments, info = model.transcribe("temp_audio.wav", beam_size=5)
    print(f"Detected language: {info.language}")

    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        fill_chart_data(segment.text)


def fill_chart_data(transcribed_text):
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
        with open("chart_data.json", "w") as f:
            json.dump(chart_data, f, indent=4)
        print(f"Updated Chart: {chart_entry}")


if __name__ == "__main__":
    asyncio.run(record_audio_segment())
