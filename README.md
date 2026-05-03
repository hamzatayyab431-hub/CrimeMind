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
*   **Machine Learning (NLP):** 
    *   **SVM & Random Forest:** Baseline scikit-learn models for evidence classification and deception detection.
    *   **BiLSTM (TensorFlow/Keras):** A deep bidirectional LSTM neural network trained to categorize crime scene evidence.
    *   **DistilBERT (PyTorch/Transformers):** A fine-tuned Hugging Face transformer model acting as an advanced deception detector to classify suspect statements.
    *   **Real-time Analytics:** Custom heuristics using `TextBlob` and `Scikit-Learn` perform sentiment analysis, intent classification, and calculated severity tracking to adjust the AI's tone (e.g. flagging a suspect as "Defensive / Evasive").

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

### 🕵️‍♂️ 1. Detective Simulator (Sandbox)
This mode has the most complex logic, heavily shifting based on the story type and role you select. The AI is strictly programmed with a "Start -> Mid-Game -> End" flow.

**Story Options:**
*   **Premade Crime Story:** The AI dynamically generates a completely random, highly-detailed, and thrilling crime scenario behind the scenes so every playthrough is unique.
*   **Custom Story:** You can explicitly type out your own scenario (e.g. "A bank heist in 1920s Chicago") and the AI will build the entire game around your prompt.

**Role Dynamics:**
*   **As the Detective:** The AI acts as your trusted buddy/partner. It provides raw facts and occasionally drops subtle hints to point you in the right direction. 
*   **As the Victim:** The AI acts as a sympathetic Private Investigator you hired. It listens to your trauma and asks you who *you* think did it, building the case around your suspicions.
*   **As the Criminal:** This is the most unique route. The AI starts as a narrator helping you set up the perfect crime. Once you actually commit the crime, the AI dynamically switches roles to become the Detective interrogating *you*. **Win State:** If you lie smartly and cover your tracks, you can successfully evade the law and win. If you make logical errors, the AI catches you.
*   **As the Explorer:** The AI acts as a cinematic 3rd-party narrator, offering you limited narrative options to freely explore the unfolding story from an outside perspective.
*   **The End of the Story:** When the mystery is solved (or you get caught), the AI is programmed to trigger a **Full Debrief**. It will reveal who the real criminal was, explain the full timeline of the crime, grade you on how well you played your role, and explicitly print *"STORY FINISHED"*.

### 🧞‍♂️ 2. Akinator Suspect Mode
*   **How it Plays:** The AI is brutally restricted to asking you strictly *one* "Yes/No" question at a time. If you try to dodge the question via voice or text, the AI is programmed to briefly acknowledge what you said, but aggressively force you back into answering the Yes/No question.
*   **The End:** The game ends when the AI either successfully guesses your criminal (in which case it provides a cool historical breakdown of who they were) OR if it completely gives up, at which point it will roast you and admit defeat.

### 😈 3. Witty Interrogator
*   **How it Plays:** There is no "crime scene" here. The AI is programmed to act like an aggressive, hostile interrogator trying to figure out where you were today. It actively uses the new Machine Learning stats (Tone & Severity) to attack you. If you sound nervous or evasive, it will mercilessly mock your intelligence and life choices.
*   **The End:** This mode is an **open-ended psychological battle**. There is no formal "Game Over" screen. It ends when you either confess, break character, or the AI formally decides to "arrest" you based on catching a logical flaw in your alibi.

### 🔍 4. Crime Scene Analyst
*   **How it Plays:** The AI channels its inner Sherlock Holmes. It is programmed to show off by noticing tiny details. If you upload an image, it routes the image to the powerful `Llama 3.2 11B Vision` model to scan the pixels for clues. 
*   **The End:** This mode is also open-ended. It acts as your permanent forensic lab. You just keep throwing evidence, audio logs, and photos at it, and it will continuously update its working theory of who the killer is. 


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
