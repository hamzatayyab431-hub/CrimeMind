# 🕵️‍♂️ CrimeMind
**AI Course Project - FAST-NUCES Lahore (Spring 2026) Track A: Application Development**

CrimeMind is an end-to-end, AI-powered detective agent that immerses users in interactive crime investigations. The application acts as a virtual detective, requiring users to deduce culprits through evidence, logical reasoning, and visual clue analysis.

### 👥 Team Members
*   Hamza Tayyab - 24L-0933
*   Ali Jawad - 24L-0531
*   Ibrahim Zaidi - 24L-0839
*   Raza Shahid - 24L-0789

---

## 🏗️ Architecture & Tech Stack

CrimeMind is built using a modern Python-based AI web stack designed for real-time multimodal interaction:
*   **Frontend / UI:** [Streamlit](https://streamlit.io/) handles the reactive UI, state management, and real-time DOM injections for background animations and custom CSS (Orbitron & Share Tech Mono fonts).
*   **Core LLM Engine:** Powered by [Groq's](https://groq.com/) blazing-fast API inference. 
    *   `llama-3.1-8b-instant` for core conversational reasoning and storytelling.
    *   `llama-3.2-11b-vision-preview` to process and deduce clues from user-uploaded images.
*   **Audio Pipeline:** 
    *   **Input:** Streamlit's native audio widget records user voice.
    *   **Transcription:** Processed instantly via Groq's `whisper-large-v3` model.
    *   **Output (TTS):** The `edge-tts` library generates dynamic voice replies which are seamlessly pushed to the UI via base64 encoded HTML injections for robust autoplay capabilities.
*   **Machine Learning (NLP):** Custom Python heuristics using `TextBlob` and `Scikit-Learn` perform real-time sentiment analysis, intent classification, and calculated severity tracking to adjust the AI's tone (e.g. flagging a suspect as "Defensive / Evasive").

---

## ✨ Core Features

*   **🎙️ Two-Way Audio Interrogations:** Speak directly to the AI using the native audio recording widget. The AI will transcribe your voice using Whisper, analyze your tone, and dynamically generate TTS voice replies that play automatically!
*   **👁️ Crime Scene Vision:** Upload actual images of crime scenes, suspects, or evidence. The app natively hot-swaps to the Vision model to visually analyze your images with Sherlock Holmes-level deduction.
*   **🧠 Machine Learning Analytics:** The sidebar tracks your every word. The ML engine calculates a dynamic **Confidence %**, flags the **Severity** of your statements (e.g. "🚨 MAJOR FELONY"), and actively detects your **Tone**. If you act suspiciously, the AI knows!
*   **🎨 Dynamic Cyberpunk Themes:** Fluidly swap between Light, Dark, and Noir themes. The application injects a persistent global canvas background with interactive particle physics and glitch effects that completely transform the UI aesthetic on the fly.
*   **🔒 Seamless State Management:** Your chat history, discovered clues, and audio widget state are perfectly preserved across theme changes without annoying page crashes or repeating audio loops.

---

## 🎮 Game Modes

CrimeMind offers 4 completely distinct ways to play:

### 1. 🕵️‍♂️ Detective Simulator (Sandbox)
*   **As the Detective:** The AI acts as your trusted buddy/partner, providing raw facts and dropping subtle hints.
*   **As the Victim:** The AI acts as a sympathetic Private Investigator you hired, building the case around your suspicions.
*   **As the Criminal:** You start by setting up the perfect crime. Once you commit the crime, the AI dynamically switches roles to interrogate *you*.
*   **The End State:** When the mystery is solved, the AI triggers a **Full Debrief**, grading your performance before explicitly stating *"STORY FINISHED"*.

### 2. 🧞‍♂️ Akinator Suspect Mode
*   Think of a famous historical or fictional criminal. The AI is restricted to asking you strictly *one* "Yes/No" question at a time to deduce who you are thinking of.

### 3. 😈 Witty Interrogator
*   Face off against a hostile interrogator. The AI uses your ML stats to attack you. If you sound nervous or evasive, it will mercilessly mock your intelligence and aggressively look for logical flaws in your alibi.

### 4. 🔍 Crime Scene Analyst
*   Upload images and provide clues to your personal forensic lab. The AI channels its inner Sherlock Holmes to logically deduce the sequence of events from raw data.

---

## 🚀 Installation & Setup Guide

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your computer.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
The application requires a Groq API key to power the Llama 3 models. Create a file named `.env` in the root folder and add your API key:
```env
GROQ_API_KEY=your_actual_api_key_here
```

### 4. Run the Application
Start the Streamlit server:
```bash
python -m streamlit run app.py
```
