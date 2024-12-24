import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI
import pyaudio
import os
from colorama import Fore, Style
import wave
import time
import audioop

from pydantic import BaseModel, Field

load_dotenv(override=True)

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Colors
NEON_GREEN = Fore.GREEN
RESET_COLOR = Style.RESET_ALL

openai = OpenAI(api_key=OPENAI_API_KEY)


# Define a model for each tooth
class Tooth(BaseModel):
    pocket_depths: List[int] = Field(
        default_factory=list, description="Pocket depth measurements in mm"
    )
    gingival_margin: List[int] = Field(
        default_factory=list, description="Gingival margin values in mm"
    )
    bleeding: Optional[str] = Field(
        None,
        description="Surface areas with bleeding (e.g., mesial, distal, buccal, all)",
    )
    mobility: Optional[int] = Field(None, description="Mobility score (0 to 3)")
    furcation_involvement: Optional[str] = Field(
        None, description="Furcation involvement grade (I, II, III) if present"
    )


# Periodontal Chart Schema
class PeriodontalChart(BaseModel):
    teeth: Dict[str, Tooth] = Field(
        default_factory=dict, description="Periodontal data for each tooth"
    )

    @staticmethod
    def initialize_chart():
        return PeriodontalChart(
            teeth={str(i): Tooth() for i in range(1, 33)}  # Initialize teeth 1-32
        )

current_chart = {}

def analyze_periodontal(transcription_text):
    print("Periodontal Analysis Result processing:")

    global current_chart

    # Prompt for GPT to analyze the transcription
    prompt = (
        "You are a dental assistant AI specialized in periodontal charting. "
        "I will provide dental examination transcriptions line by line. "
        "Your task is to analyze each line individually and progressively fill out a periodontal chart. \n"
        "Key instructions for each line of transcription: \n"
        "- Identify the relevant tooth number and update its chart entry. \n"
        "- If pocket depths are mentioned, record them as a list of integers. \n"
        "- If bleeding is noted, specify the affected surfaces (e.g., mesial, distal, buccal, lingual) or mark 'all' if applicable. \n"
        "- For gingival margin measurements, add them to the respective tooth's chart entry. \n"
        "- If mobility is mentioned, record it as an integer between 0 and 3. \n"
        "- If furcation involvement is described, record the grade (I, II, or III). \n"
        "Correct any transcription errors (e.g., 'three' to 3, 'one two one' to [1, 2, 1]). \n"
        "Return the updated periodontal chart in JSON format, adding the new measurements without overwriting previous values. \n"
        "Example output format: \n"
        "{\n"
        '  "teeth": {\n'
        '    "8": {\n'
        '      "pocket_depths": [3, 2, 3],\n'
        '      "gingival_margin": [3, 3, 3],\n'
        '      "bleeding": null,\n'
        '      "mobility": null,\n'
        '      "furcation_involvement": null\n'
        "    },\n"
        '    "1": {\n'
        '      "bleeding": "distal, mesial"\n'
        "    },\n"
        '    "4": {\n'
        '      "mobility": 2\n'
        "    },\n"
        '    "6": {\n'
        '      "furcation_involvement": "I"\n'
        "    }\n"
        "  }\n"
        "} \n"
        "If no data is available from the current transcription line, return {}. Only return the JSON format. No extra text or comments."
    )

    # Call GPT with the transcription text
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcription_text},
        ],
        temperature=0.2,
        max_tokens=500,
    )

    # Parse GPT Response
    try:
        parsed_analysis = json.loads(response.choices[0].message.content.strip("```json").strip("```"))
        print(json.dumps(parsed_analysis, indent=4))  # Display result

        # Merge new data into the existing chart
        for tooth, data in parsed_analysis.get("teeth", {}).items():
            if tooth in current_chart:
                # Update existing tooth entry without overwriting previous data
                for key, value in data.items():
                    if value:  # Append new values if they exist
                        if isinstance(value, list):
                            current_chart[tooth][key].extend(value)
                        else:
                            current_chart[tooth][key] = value
            else:
                # Add new tooth entry if not already in chart
                current_chart[tooth] = data

        print("Updated Periodontal Chart:")
        print(json.dumps(current_chart, indent=4))

    except Exception as e:
        print(f"Error processing GPT response: {e}")


def record_until_sustained_silence(p, stream):
    frames = []
    silent_start = None
    rolling_window = []
    total_rms = 0

    while True:
        data = stream.read(BUFFER_SIZE)
        frames.append(data)

        rms = audioop.rms(data, 2)
        total_rms += rms
        rolling_window.append(rms)

        if len(rolling_window) > ROLLING_WINDOW:
            rolling_window.pop(0)

        avg_rms = sum(rolling_window) / len(rolling_window)

        if avg_rms < SILENCE_THRESHOLD:
            if silent_start is None:
                silent_start = time.time()
            elif time.time() - silent_start > SILENCE_GAP_TIMEOUT:
                break
        else:
            silent_start = None

    if total_rms > MIN_SOUND_THRESHOLD:
        chunk_file = f"chunk_{int(time.time())}.wav"
        with wave.open(chunk_file, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
        return chunk_file
    return None


def transcribe_audio(audio_file):
    with open(audio_file, "rb") as f:
        response = openai.audio.transcriptions.create(
            model=MODEL_SIZE, file=f, language="en", response_format="text"
        )
    os.remove(audio_file)
    return response


def run_transcription():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=BUFFER_SIZE,
    )

    try:
        print("Recording...")
        while True:
            audio_file = record_until_sustained_silence(p, stream)
            if audio_file:
                print("Processing audio...")
                transcription = transcribe_audio(audio_file)
                print(NEON_GREEN + transcription + RESET_COLOR)
                analyze_periodontal(transcription)
            else:
                print("No significant audio detected")

    except KeyboardInterrupt:
        print("\nRecording stopped by user")

    finally:
        print("Recording Complete")
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    run_transcription()
