## 🦷 Real-Time Speech-to-Text Periodontal Charting  

A real-time dental assistant application that transcribes periodontal charting voice inputs and generates structured periodontal chart data using OpenAI GPT. This project utilizes PyAudio for live audio capture and OpenAI for chart analysis.  

---

### 📋 **Features**  
- **Real-time transcription** of periodontal charting procedures.  
- **Automatic chart generation** from voice commands.  
- **Error correction** for transcriptions (e.g., "three two three" → `[3, 2, 3]`).  
- **Progressive chart building** – Line-by-line input fills the chart without overwriting previous measurements.  
- **JSON output** formatted for easy integration with dental record systems.  

---

### 🚀 **Installation and Setup**  

#### **Prerequisites:**  
- Python 3.11+  
- [PDM (Python Dependency Manager)](https://pdm.fming.dev/latest/)  

#### **Install PDM:**  
```bash
pip install pdm
```

---

#### **Clone the Repository:**  
```bash
git clone https://github.com/rishabjn10/Real-Time-Speech-To-Text
cd Real-Time-Speech-To-Text
```

---

#### **Install Dependencies (Using PDM):**  
```bash
pdm install
```

---

### 🔑 **Environment Variables (.env):**  

Create a `.env` file in the project root and add the following:  
```plaintext
OPENAI_API_KEY=your_openai_api_key
```

---

### ▶️ **Run the Application:**  
```bash
pdm run python .\real-time-stt.py
```

---

### 🎙️ **How It Works:**  
1. The microphone captures audio for periodontal charting commands.  
2. The application transcribes the audio in real-time.  
3. Each line of transcription updates the periodontal chart progressively.  
4. The chart is generated in **JSON format** and printed to the console.  

---

### 📄 **Example Transcription:**  
```
Tooth number 8, pocket depths - three two three.  
Gingival margin - three three three.  
Tooth number 22, pocket depths - one two one.  
Tooth number 1, bleeding on the distal and mesial surfaces.  
```

---

### 🧾 **Example JSON Output:**  
```json
{
  "teeth": {
    "8": {
      "pocket_depths": [3, 2, 3],
      "gingival_margin": [3, 3, 3],
      "bleeding": null
    },
    "22": {
      "pocket_depths": [1, 2, 1]
    },
    "1": {
      "bleeding": "distal, mesial"
    }
  }
}
```

---

### 🛠️ **Customization:**  
- **Quadrant Handling**: Extend the logic to apply measurements across entire quadrants.  
- **Surface-specific Data**: Add buccal, lingual, mesial, and distal-specific depths.  
- **Mobility/Furcation**: Enhance the schema to capture advanced periodontal data like furcation involvement and tooth mobility.  

---

### 🤝 **Contributing:**  
Pull requests are welcome. For major changes, please open an issue to discuss what you would like to change.  

---

### 🧩 **Troubleshooting:**  
- **No Audio Detected** – Ensure your microphone is correctly set up and permissions are granted.  
- **GPT Errors** – Verify that your OpenAI API key is valid and has sufficient quota.  

---

### 📚 **Dependencies:**  
- `openai`  
- `pyaudio`  
- `pydantic`  
- `python-dotenv`  

---

### 📜 **License:**  
This project is licensed under the MIT License.  

---

### 👨‍💻 **Author:**  
**Rishabh Jain**  