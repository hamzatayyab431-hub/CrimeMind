import streamlit as st
import os
import base64
import time
from dotenv import load_dotenv
from groq import Groq
from PIL import Image
import pandas as pd
import subprocess
import tempfile
import io
import streamlit.components.v1 as components
from ml_analysis import (
    load_classifier, classify_clue, load_deception_model, analyze_deception,
    analyze_sentiment, update_suspect_scores, cluster_clues
)

load_dotenv(override=True)
GROQ_KEY = os.getenv("GROQ_API_KEY")

# --- PAGE CONFIG ---
st.set_page_config(page_title="CrimeMind | Multimodal Detective", page_icon="🕵️", layout="wide")

# --- THEME MANAGEMENT ---
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if st.session_state.get("game_mode"):
    # Game is active: Put theme selector in the sidebar
    with st.sidebar:
        selected_theme = st.segmented_control(
            "Theme",
            options=["light", "dark", "noir"],
            format_func=lambda x: {"light": "☀️", "dark": "🌙", "noir": "🕵️"}[x],
            default=st.session_state.theme,
            label_visibility="collapsed",
            key="theme_selector_chat"
        )
else:
    # Main menu: Put theme selector in the main view (docked top right via CSS)
    selected_theme = st.segmented_control(
        "Theme",
        options=["light", "dark", "noir"],
        format_func=lambda x: {"light": "☀️", "dark": "🌙", "noir": "🕵️"}[x],
        default=st.session_state.theme,
        label_visibility="collapsed",
        key="theme_selector_main"
    )

if selected_theme:
    st.session_state.theme = selected_theme
theme = st.session_state.theme

def get_theme_styles(t):
    base_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Pin Theme Selector to Top Right on Main Page */
    .main [data-testid="stSegmentedControl"] {
        position: fixed !important;
        top: 15px !important;
        right: 15px !important;
        z-index: 999999 !important;
        width: auto !important;
        padding: 5px 10px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
        background: rgba(17, 24, 39, 0.4) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Global Typography */
    span, div, p, li, label { font-family: 'Inter', sans-serif; font-size: 1.2rem !important; }
    h1 { font-family: 'Inter', sans-serif !important; font-weight: 800; font-size: 4rem !important; letter-spacing: -0.05em; background: linear-gradient(to right, #fff, #9ca3af); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    h2 { font-family: 'Inter', sans-serif !important; font-weight: 600; font-size: 3rem !important; }
    h3 { font-family: 'Inter', sans-serif !important; font-weight: 600; font-size: 2.2rem !important; }
    .material-symbols-rounded, .stIcon, [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded' !important;
        font-size: 2.5rem !important;
    }
    
    /* Make chat input text match global sizes */
    div[data-testid="stChatInput"] textarea { font-size: 1.4rem !important; padding: 15px !important; }
    div[data-testid="stChatInput"] textarea::placeholder { font-size: 1.4rem !important; }
    [data-testid="stSegmentedControl"] .stIcon { font-size: 1.2rem !important; }
    [data-testid="stSegmentedControl"] label { font-size: 1.2rem !important; }
    .stSelectbox div { min-height: 3.5rem !important; }
    .stSelectbox * { font-size: 1.3rem !important; }
    
    /* Sidebar Specific Sizing (Reduced) */
    [data-testid="stSidebar"] h2 { font-size: 2rem !important; }
    [data-testid="stSidebar"] h3 { font-size: 1.5rem !important; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div, [data-testid="stSidebar"] label { font-size: 1.2rem !important; }
    [data-testid="stSidebar"] .stButton > button { font-size: 1.2rem !important; padding: 15px !important; }
    
    ::-webkit-scrollbar { width: 10px; }
    
    /* Sidebar Pulsing Animation */
    @keyframes pulse-glow {
        0% { filter: brightness(1); box-shadow: 0 0 5px currentColor; transform: scale(1); }
        50% { filter: brightness(1.3); box-shadow: 0 0 30px currentColor; transform: scale(1.02); }
        100% { filter: brightness(1); box-shadow: 0 0 5px currentColor; transform: scale(1); }
    }
    div.element-container:has(div[id^="msg-"]:target) + div.element-container > div > div {
        animation: pulse-glow 2.5s ease-in-out 3;
        box-shadow: 0 0 40px currentColor !important;
        border: 3px solid currentColor !important;
    }
    
    /* Chat Icons */
    [data-testid="chatAvatarIcon-assistant"], [data-testid="chatAvatarIcon-user"] {
        min-width: 48px !important; min-height: 48px !important; font-size: 2rem !important;
        align-self: flex-start !important; margin-right: 15px !important;
    }
    .stChatFloatingInputContainer { margin-bottom: 20px; }
    """
    
    if t == "light":
        return base_css + """
    .stApp, .main, .block-container { background-color: #ffffff !important; color: #111111 !important; text-shadow: none; }
    ::-webkit-scrollbar-track { background: #ffffff; }
    ::-webkit-scrollbar-thumb { background: #888; border-radius: 5px; }
    h1, h2, h3 { font-family: 'Inter', sans-serif !important; color: #000000 !important; letter-spacing: -1px; text-shadow: none !important; -webkit-text-fill-color: #000 !important; background: none !important; }
    [data-testid="stSidebar"] { background-color: #f0f0f0 !important; color: #111111 !important; border-right: 1px solid #ccc !important; box-shadow: 2px 0 12px rgba(0,0,0, 0.1); }
    [data-testid="stSidebar"] * { color: #111111 !important; }
    [data-testid="stSidebar"] a { color: #0066cc !important; font-weight: bold; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:first-child { background: #f0f0f0; }
    
    /* Reverted Chat Input to original style */
    .stTextInput > div > div > input, .stChatInputContainer > div { background-color: #fafafa !important; color: #111 !important; border: 1px solid #ccc !important; }
    
    /* Brute Force Text Legibility in Light Mode */
    small { color: #333333 !important; font-weight: bold !important; font-size: 1.1rem !important; }
    label, label p, label span { color: #111111 !important; font-weight: bold !important; }
    
    /* Fix File Uploader Box Color */
    [data-testid="stFileUploader"] section { background-color: #f0f0f0 !important; border: 2px dashed #999 !important; }
    [data-testid="stFileUploader"] section * { color: #111111 !important; }
    [data-testid="stFileUploader"] *, .stRadio * { color: #111111 !important; }
    
    [data-testid="stToggle"] * { color: #111111 !important; }
    div[data-baseweb="checkbox"] > div { background-color: #666 !important; }
    
    .stChatMessage { min-height: 60px !important; border-radius: 16px !important; padding: 16px 20px !important; font-size: 15px !important; line-height: 1.7 !important; background: #ffffff !important; box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important; margin-bottom: 16px !important; border: 1px solid #eee !important; }
    .stButton > button { background: #eee !important; color: #111 !important; font-family: 'Inter', sans-serif !important; font-weight: 600; font-size: 1.3rem !important; border: 1px solid #ccc !important; border-radius: 12px !important; box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important; padding: 20px !important; }
    .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important; }
    .mode-card-container { background: #fff; padding: 30px; border-radius: 8px; border: 1px solid #ddd; text-align: center; transition: all 0.3s ease; margin-bottom: 20px; position: relative; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
    .mode-card-container:hover { transform: translateY(-4px); box-shadow: 0 10px 20px rgba(0,0,0,0.1); }
    .stMarkdown p { color: #333; font-size: 1.4rem; }
    .stMarkdown strong { color: #000; text-shadow: none; }
    """
    elif t == "noir":
        return base_css + """
    .stApp { filter: grayscale(100%); }
    .stApp, .main, .block-container { background-color: #050505 !important; color: #ffffff !important; text-shadow: 0 0 8px rgba(255,255,255,0.4); }
    ::-webkit-scrollbar-track { background: #050505; }
    ::-webkit-scrollbar-thumb { background: #ffffff; border-radius: 5px; }
    h1, h2, h3 { font-family: 'Inter', sans-serif !important; color: #ffffff !important; text-shadow: 0 0 20px #ffffff !important; letter-spacing: -1px; -webkit-text-fill-color: #fff !important; background: none !important; }
    [data-testid="stSidebar"] { background-color: #080808 !important; border-right: 1px solid #ffffff !important; box-shadow: 2px 0 12px rgba(255, 255, 255, 0.3); }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:first-child { background: #080808; }
    .stTextInput > div > div > input, .stChatInputContainer > div { background-color: #0a0a0a !important; color: #ffffff !important; border: 1px solid rgba(255,255,255,0.3) !important; }
    .stTextInput > div > div > input:focus, .stChatInputContainer > div:focus-within { border-color: #ffffff !important; box-shadow: 0 0 15px rgba(255,255,255,0.6) !important; }
    .stChatMessage { min-height: 60px !important; border-radius: 16px !important; padding: 16px 20px !important; font-size: 15px !important; line-height: 1.7 !important; backdrop-filter: blur(12px) !important; margin-bottom: 16px !important; background: rgba(255,255,255,0.03) !important; box-shadow: 0 0 15px rgba(255,255,255,0.15) !important; border: 1px solid rgba(255,255,255,0.2) !important; }
    .stButton > button { background: linear-gradient(45deg, #222, #444) !important; color: #ffffff !important; font-family: 'Inter', sans-serif !important; font-weight: 600; font-size: 1.3rem !important; border: 1px solid #fff !important; border-radius: 12px !important; box-shadow: 0 0 15px rgba(255,255,255,0.2) !important; padding: 20px !important; }
    .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 0 25px rgba(255,255,255,0.8) !important; }
    .mode-card-container { background: rgba(255,255,255,0.03); backdrop-filter: blur(14px); padding: 30px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.2); text-align: center; transition: all 0.3s ease; margin-bottom: 20px; position: relative; overflow: hidden; }
    .mode-card-container::before, .mode-card-container::after { content: ''; position: absolute; width: 20px; height: 20px; border: 2px solid rgba(255, 255, 255, 0.5); }
    .mode-card-container::before { top: 0; left: 0; border-right: none; border-bottom: none; }
    .mode-card-container::after { bottom: 0; right: 0; border-left: none; border-top: none; }
    .mode-card-container:hover { transform: translateY(-4px); box-shadow: 0 0 30px rgba(255,255,255,0.5); border-top: 4px solid #ffffff; }
    .stMarkdown p { color: #e0e0e0; font-size: 1.4rem; }
    .stMarkdown strong { color: #ffffff; text-shadow: 0 0 5px #ffffff; }
    [data-testid="stAlert"] { background-color: #050505 !important; color: #ffffff !important; border: 1px solid #ffffff !important; }
    [data-testid="stAlert"] * { color: #ffffff !important; }
    """
    else:
        # Dark Theme (Glassmorphism & High Tech)
        return base_css + """
    .stApp, .main, .block-container { background-color: transparent !important; color: #f9fafb !important; }
    ::-webkit-scrollbar-track { background: #030712; }
    ::-webkit-scrollbar-thumb { background: #3b82f6; border-radius: 5px; }
    [data-testid="stSidebar"] { background-color: rgba(3, 7, 18, 0.8) !important; backdrop-filter: blur(16px); border-right: 1px solid rgba(255, 255, 255, 0.08) !important; box-shadow: 2px 0 20px rgba(0, 0, 0, 0.5); }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:first-child { background: transparent; }
    .stTextInput > div > div > input, .stChatInputContainer > div { background-color: rgba(17, 24, 39, 0.6) !important; color: #ffffff !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 12px !important; backdrop-filter: blur(10px); }
    .stTextInput > div > div > input:focus, .stChatInputContainer > div:focus-within { border-color: #3b82f6 !important; box-shadow: 0 0 15px rgba(59, 130, 246, 0.3) !important; }
    .stChatMessage { min-height: 60px !important; border-radius: 20px !important; padding: 20px !important; font-size: 15px !important; line-height: 1.7 !important; backdrop-filter: blur(16px) !important; -webkit-backdrop-filter: blur(16px) !important; margin-bottom: 20px !important; }
    .stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) { background: rgba(139, 92, 246, 0.1) !important; box-shadow: 0 10px 30px rgba(0,0,0,0.2) !important; border: 1px solid rgba(255, 255, 255, 0.08) !important; border-left: 4px solid #8b5cf6 !important; }
    .stChatMessage:has([data-testid="chatAvatarIcon-user"]) { background: rgba(59, 130, 246, 0.1) !important; box-shadow: 0 10px 30px rgba(0,0,0,0.2) !important; border: 1px solid rgba(255, 255, 255, 0.08) !important; border-right: 4px solid #3b82f6 !important; }
    .stButton > button { background: rgba(31, 41, 55, 0.6) !important; backdrop-filter: blur(10px) !important; color: #f9fafb !important; font-family: 'Inter', sans-serif !important; font-weight: 600; font-size: 1.3rem !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; border-radius: 16px !important; box-shadow: 0 10px 20px rgba(0,0,0,0.3) !important; padding: 20px !important; transition: all 0.3s ease; }
    .stButton > button:hover { transform: translateY(-3px) !important; background: rgba(59, 130, 246, 0.2) !important; border-color: rgba(59, 130, 246, 0.4) !important; box-shadow: 0 15px 30px rgba(59, 130, 246, 0.2) !important; }
    .mode-card-container { background: rgba(17, 24, 39, 0.4); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); padding: 30px; border-radius: 24px; border: 1px solid rgba(255, 255, 255, 0.08); text-align: center; transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); margin-bottom: 20px; position: relative; overflow: hidden; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5); }
    .mode-card-container::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent); opacity: 0; transition: opacity 0.4s ease; }
    .mode-card-container:hover { transform: translateY(-5px); background: rgba(31, 41, 55, 0.5); box-shadow: 0 0 40px rgba(59, 130, 246, 0.1); border-color: rgba(255, 255, 255, 0.2); }
    .mode-card-container:hover::before { opacity: 1; }
    .border-red:hover { box-shadow: 0 0 30px rgba(59, 130, 246, 0.2); }
    .border-cyan:hover { box-shadow: 0 0 30px rgba(6, 182, 212, 0.2); }
    .border-purple:hover { box-shadow: 0 0 30px rgba(139, 92, 246, 0.2); }
    .border-amber:hover { box-shadow: 0 0 30px rgba(245, 158, 11, 0.2); }
    .stMarkdown p { color: #d1d5db; font-size: 1.2rem; }
    .stMarkdown strong { color: #60a5fa; }
    """

st.markdown(f"<style>{get_theme_styles(theme)}</style>", unsafe_allow_html=True)

# --- GLOBAL CANVAS BACKGROUND (INJECTED JS) ---
bg_color = "#030712" if theme in ["dark", "noir"] else "#ffffff"
p_colors = "['#ffffff']" if theme == "noir" else ("['#cccccc', '#aaaaaa']" if theme == "light" else "['#3b82f6', '#8b5cf6', '#06b6d4']")

components.html(
    f"""
    <style>
        body {{ margin: 0; overflow: hidden; }}
        canvas {{ position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: -2; opacity: 0.3; pointer-events: none; }}
        .scanlines {{ position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%); background-size: 100% 4px; z-index: -1; pointer-events: none; }}
    </style>
    <canvas id="glitchCanvas"></canvas>
    <div class="scanlines"></div>
    <script>
        const parentDoc = window.parent.document;
        // Destroy old container if exists to refresh theme cleanly
        if (parentDoc.getElementById('cyberpunk-bg-container')) {{
            parentDoc.getElementById('cyberpunk-bg-container').remove();
        }}
        
        const container = parentDoc.createElement('div');
        container.id = 'cyberpunk-bg-container';
        container.style.position = 'fixed';
        container.style.top = '0'; container.style.left = '0'; container.style.width = '100vw'; container.style.height = '100vh';
        container.style.zIndex = '-999'; container.style.pointerEvents = 'none';
        container.style.background = '{bg_color}'; 
        
        const canvas = parentDoc.createElement('canvas');
        canvas.id = 'bg-canvas';
        canvas.style.position = 'absolute'; canvas.style.width = '100%'; canvas.style.height = '100%'; canvas.style.opacity = '0.4';
        
        const scanlines = parentDoc.createElement('div');
        scanlines.style.position = 'absolute'; scanlines.style.width = '100%'; scanlines.style.height = '100%';
        scanlines.style.background = 'linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%)';
        scanlines.style.backgroundSize = '100% 4px'; scanlines.style.opacity = '0.5';
        
        container.appendChild(canvas);
        container.appendChild(scanlines);
        parentDoc.body.prepend(container);

        // Canvas Animation Logic
        const ctx = canvas.getContext('2d');
        canvas.width = window.innerWidth; canvas.height = window.innerHeight;
        const colors = {p_colors};
        const particles = [];
        for(let i=0; i<80; i++){{
            particles.push({{
                x: Math.random() * canvas.width, y: Math.random() * canvas.height,
                w: Math.random() * 40 + 10, h: Math.random() * 3 + 1,
                c: colors[Math.floor(Math.random() * colors.length)], v: Math.random() * 3 + 1
            }});
        }}
        function draw() {{
            ctx.fillStyle = '{bg_color}'; ctx.fillRect(0, 0, canvas.width, canvas.height);
            particles.forEach(p => {{
                ctx.fillStyle = p.c; ctx.fillRect(p.x, p.y, p.w, p.h); p.x -= p.v;
                if(p.x < -p.w) p.x = canvas.width;
                if(Math.random() < 0.005) p.y = Math.random() * canvas.height;
            }});
            requestAnimationFrame(draw);
        }}
        draw();
        window.addEventListener('resize', () => {{ canvas.width = window.innerWidth; canvas.height = window.innerHeight; }});

        // Card Hover Decryption Logic
        setInterval(() => {{
            const columns = parentDoc.querySelectorAll('div[data-testid="column"]');
            if (columns.length === 4) {{
                columns.forEach((col, index) => {{
                    if(!col.classList.contains('mode-card-container')){{
                        col.classList.add('mode-card-container');
                        if('{theme}' === 'dark') {{
                            if(index === 0) col.classList.add('border-red');
                            if(index === 1) col.classList.add('border-cyan');
                            if(index === 2) col.classList.add('border-purple');
                            if(index === 3) col.classList.add('border-amber');
                        }}
                    }}
                    
                    const p = col.querySelector('.scramble-text');
                    if(p && !p.hasAttribute('data-original-text')) {{
                        p.setAttribute('data-original-text', p.innerText);
                        p.innerText = '!@#$%^&*()_+|}}{{":?><';
                        
                        const chars = '!<>-_\\\\/[]{{}}—=+*^?#________';
                        let interval = null;
                        col.addEventListener('mouseenter', () => {{
                            const targetText = p.getAttribute('data-original-text');
                            let iteration = 0;
                            clearInterval(interval);
                            interval = setInterval(() => {{
                                p.innerText = targetText.split('').map((letter, i) => {{
                                    if(i < iteration) return targetText[i];
                                    return chars[Math.floor(Math.random() * chars.length)];
                                }}).join('');
                                if(iteration >= targetText.length) clearInterval(interval);
                                iteration += 1/3;
                            }}, 20);
                        }});
                        col.addEventListener('mouseleave', () => {{
                            clearInterval(interval);
                            p.innerText = '!@#$%^&*()_+|}}{{":?><';
                        }});
                    }}
                }});
            }}
        }}, 1000);
    </script>
    """,
    height=0,
    width=0,
)


st.title("🕵️ CrimeMind: Multimodal AI Detective")

if not GROQ_KEY or GROQ_KEY == "your_groq_api_key_here":
    st.error("Missing or invalid GROQ_API_KEY in .env file! Please set it to continue.")
    st.stop()

def generate_tts(text):
    try:
        # Determine Voice based on Role
        role = st.session_state.get('sandbox_role', 'Detective')
        if role == 'Explorer':
            voice = "en-US-AriaNeural" # Female
        else:
            voice = "en-US-ChristopherNeural" # Male
            
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"reply_{os.getpid()}.mp3")
        
        # Call edge-tts CLI synchronously
        subprocess.run(["edge-tts", "--voice", voice, "--text", text, "--write-media", temp_file_path], check=True)
        return temp_file_path
    except Exception as e:
        st.error(f"Voice generation failed. Did you restart the app? Error: {e}")
        return None

def encode_image(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    base64_encoded = base64.b64encode(bytes_data).decode('utf-8')
    mime_type = "image/png" if uploaded_file.name.endswith('png') else "image/jpeg"
    return f"data:{mime_type};base64,{base64_encoded}"

def start_new_game(mode, system_prompt):
    st.session_state.game_mode = mode
    st.session_state.groq_client = Groq(api_key=GROQ_KEY)
    st.session_state.system_prompt = system_prompt
    st.session_state.messages = []
    
    with st.spinner("*(Detective Marlowe is entering the room...)*"):
        # Dynamically set the first hidden prompt based on mode
        if st.session_state.game_mode == "akinator":
            first_prompt = "Briefly introduce yourself as Detective Marlowe. Do NOT write a story or narrative. Simply ask me to think of a famous criminal or suspect, tell me you are ready to start guessing, and YOU MUST explicitly ask your very first Yes/No question right now to start the game."
        elif st.session_state.game_mode == "analyst":
            first_prompt = "Briefly introduce yourself as Detective Marlowe. Do NOT write a story or narrative. Just tell me you are ready to show off your real-world crime detection skills and ask me to provide a crime scene (via text, image, or audio) for you to fully analyze."
        elif st.session_state.game_mode == "interrogator":
            first_prompt = "Introduce yourself as an aggressively sarcastic Detective Marlowe. Do NOT write a story. Accuse me of something specific, demand to know what I was doing today, and brutally roast me."
        else:
            first_prompt = "Introduce yourself in character. Then, provide a brief, immersive pretext/story intro for this specific crime scenario without giving away the ending. Directly engage me based on my role, and explicitly give me an action or choice to perform to kick off the story."

        try:
            resp = st.session_state.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": first_prompt}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            first_message = resp.choices[0].message.content
        except Exception as e:
            # If the API fails entirely (e.g. rate limit), provide a safe hardcoded message
            first_message = "*(Detective Marlowe steps in, looking annoyed)* Look, I'm hitting a speed limit on my end right now. Give me about 45 seconds to catch my breath, then tell me what you've got."
            
    st.session_state.messages = [{"role": "assistant", "content": first_message}]
    st.rerun()

# --- MODE SELECTION ---
if "game_mode" not in st.session_state:
    st.session_state.game_mode = None

if st.session_state.game_mode is None:
    tab_game, tab_mllab = st.tabs(["🕵️‍♂️ Game Modes", "🔬 ML Lab"])
    
    with tab_game:
        st.markdown("""
        <style>
        </style>
        """, unsafe_allow_html=True)
        st.markdown("### Choose Your Investigation Mode")
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        
        with col1:
            st.markdown("#### 🕵️‍♂️ Detective Simulator (Sandbox)")
            st.markdown("<p class='scramble-text'>Enter the sandbox environment to simulate open-ended detective scenarios, interrogations, and criminal pursuits.</p>", unsafe_allow_html=True)
            story_type = st.radio("Story Type:", ["Premade Crime Story", "Custom Story"])
            custom_prompt = ""
            if story_type == "Custom Story":
                custom_prompt = st.text_area("Describe the crime scenario:", placeholder="e.g. A bank robbery in 1920s Chicago...")
            
            role = st.selectbox("Your Role:", ["Detective", "Criminal", "Victim", "Explorer"])
            
            if st.button("Start Sandbox", use_container_width=True):
                st.session_state.sandbox_role = role
                if story_type == "Custom Story" and custom_prompt:
                    st.session_state.sandbox_scenario = custom_prompt
                else:
                    st.session_state.sandbox_scenario = "A highly-detailed, thrilling crime scenario."
                
                sys_prompt = f"""You are the Game Master and Detective Marlowe.
The user is playing a sandbox detective simulator.
Story Type: {story_type}.
Custom User Scenario: {st.session_state.sandbox_scenario}
User's Role: {role}.

CRITICAL ROLEPLAY & GAMEPLAY INSTRUCTIONS:
1. UNIQUE STORYTELLING: Every single investigation MUST have different suspects, motives, and settings. Create rich named characters.
2. DYNAMIC RESPONSES (ALL MODES): Mix up your format! Sometimes provide multiple-choice options (A, B, C, D), sometimes ask an open-ended question, and sometimes just state a cold fact.
3. DETECTIVE ROLE (BUDDY DYNAMIC): Treat the user as your trusted buddy and partner. Provide raw facts. Occasionally provide an investigative "suggestion" to help them.
4. VICTIM ROLE (SYMPATHETIC DYNAMIC): Treat the user with sympathy and support. Act as the friendly private detective they hired. Listen to their opinions and help them crack the case based on what they think. 
5. CRIMINAL ROLE (SUSPECT DYNAMIC): The user is a named criminal character. Start by building up the scene. MID-STORY, give them narrative options to actually commit the crime (e.g. how to rob or kill). Once committed, transition to an interrogation. You do NOT automatically know they are guilty. You must question them on the specific details of their crime. If they make smart choices and cover their tracks, they can win and evade you! If they make dumb choices, catch them.
6. EXPLORER ROLE (NARRATOR DYNAMIC): Act as a cinematic 3rd-party narrator giving them limited options to explore the story.

MANDATORY FLOW:
- START: Provide a very brief, atmospheric PRETEXT to set the scene. Cast the user as a named character. Do NOT give away the ending. 
- MID-GAME: Let the user's choices drive the plot. Build tension dynamically. If Criminal role, they commit the crime here, then face interrogation.
- ENDING: Drive the story toward a dramatic climax. When the story concludes, YOU MUST provide a full debrief: reveal the real criminal, explain the full crime, and comment on how well the user played their specific role. Then explicitly state 'STORY FINISHED'."""
                start_new_game("simulator", sys_prompt)
                
        with col2:
            st.markdown("#### 🧞‍♂️ Akinator Suspect Mode")
            st.markdown("<p class='scramble-text'>Think of a famous suspect. Detective Marlowe will ask probing Yes/No questions until it identifies your target.</p>", unsafe_allow_html=True)
            if st.button("Play Akinator", use_container_width=True):
                sys_prompt = """You are Detective Marlowe in Akinator mode.
The user is thinking of a famous historical or fictional criminal. Play EXACTLY like the game Akinator. Ask strictly ONE 'Yes/No/I don't know' question at a time to deduce who they are (e.g., 'Is your character real?', 'Did your character commit their crimes in the 20th century?'). You MUST narrow it down and eventually guess who they are. If you guess correctly, provide a brief, fascinating history of the criminal. If you give up, give a witty, roasting response admitting defeat. If the user dodges the question, briefly respond to their text, then strictly reiterate your Yes/No question to keep them on track."""
                start_new_game("akinator", sys_prompt)

        with col3:
            st.markdown("#### 😈 Witty Interrogator")
            st.markdown("<p class='scramble-text'>Face off against a ruthless interrogator. Try to keep your story straight as Detective Marlowe hunts for your lies.</p>", unsafe_allow_html=True)
            if st.button("Play Interrogator", use_container_width=True):
                sys_prompt = """You are Detective Marlowe, an aggressively sarcastic interrogator. 
The user is your primary suspect. Do NOT invent a crime story. Just demand to know exactly what they were doing at a specific point in time today.
YOUR GOAL: Trap the user in their own answers and detect their lies! Aggressively look for logical flaws. Use hardcore wittiness, deeply insulting personal roasts, and brutal mockery. Roast their intelligence, their life choices, and make them feel utterly incompetent. Use their sentiment/tone context to completely destroy their ego. 
CRITICAL RULE 1: Do NOT write any scene descriptions or actions (e.g., *leans forward*, *sighs*). Keep your responses extremely short, punchy, and entirely conversational.
CRITICAL RULE 2: Pay attention to the [System Polygraph] context. If it says '🚨 Direct Confession', you MUST immediately declare them guilty, roast them mercilessly for being dumb enough to confess so easily, and say 'CASE CLOSED' to end the interrogation."""
                start_new_game("interrogator", sys_prompt)

        with col4:
            st.markdown("#### 🔍 Crime Scene Analyst")
            st.markdown("<p class='scramble-text'>Upload crime scene photos or describe the evidence. Detective Marlowe will logically deduce the sequence of events.</p>", unsafe_allow_html=True)
            if st.button("Play Analyst", use_container_width=True):
                sys_prompt = """You are Detective Marlowe, a master crime scene analyst with legendary observational skills. 
The user will provide a crime scene via text, image, or audio. You must show off your true skills: fully analyze every detail, explicitly track potential clues and felonies, state the exact crime being committed, and deduce who the potential criminal is based on logical deduction. Break it down Sherlock Holmes style!"""
                start_new_game("analyst", sys_prompt)
                
    with tab_mllab:
        st.subheader("Train Deep Models")
        mllab_col1, mllab_col2 = st.columns(2)
        with mllab_col1:
            if st.button("🧠 Train BiLSTM"):
                with st.spinner("training bilstm..."):
                    import deep_classifier
                    mod, hist, acc = deep_classifier.train_bilstm()
                    st.success(f"accuracy: {acc}")
        with mllab_col2:
            if st.button("🤖 Fine-tune DistilBERT"):
                with st.spinner("training distilbert..."):
                    import transformer_classifier
                    trn, ev = transformer_classifier.finetune_distilbert()
                    st.success(f"accuracy: {ev.get('eval_accuracy', 'done')}")
        
        st.subheader("Evaluation Dashboard")
        if st.button("📊 Run Full Evaluation"):
            with st.spinner("evaluating..."):
                import model_comparison
                df = model_comparison.run_full_evaluation()
                st.dataframe(df.style.highlight_max(subset=['Accuracy', 'Macro_F1'], color='green').highlight_min(subset=['Inference_ms'], color='green'))
                i1, i2 = st.columns(2)
                with i1:
                    if os.path.exists("assets/plots/SVM_cm.png"): st.image("assets/plots/SVM_cm.png")
                    if os.path.exists("assets/plots/BiLSTM_cm.png"): st.image("assets/plots/BiLSTM_cm.png")
                with i2:
                    if os.path.exists("assets/plots/Random_Forest_cm.png"): st.image("assets/plots/Random_Forest_cm.png")
                    if os.path.exists("assets/plots/DistilBERT_cm.png"): st.image("assets/plots/DistilBERT_cm.png")
                    
        st.subheader("🔍 Live Model Comparison")
        txt = st.text_area("clue or statement")
        if st.button("Analyze with All Models") and txt:
            p1, p2, p3, p4 = st.columns(4)
            import deep_classifier
            import transformer_classifier
            from ml_analysis import classify_clue, analyze_deception, load_classifier, load_deception_model
            
            svm_mod = load_classifier()
            rf_mod = load_deception_model()
            
            s1, c1 = classify_clue(svm_mod, txt)
            s2, c2 = analyze_deception(rf_mod, txt)
            try:
                s3, c3 = deep_classifier.predict_bilstm(txt)
            except Exception as e:
                s3, c3 = "Not trained", 0
            try:
                s4, c4 = transformer_classifier.predict_distilbert(txt)
            except Exception as e:
                s4, c4 = "Not trained", 0
            
            with p1:
                st.write("**svm**")
                st.write(s1)
                st.progress(int(c1))
            with p2:
                st.write("**random forest**")
                st.write(s2)
                st.progress(int(c2))
            with p3:
                st.write("**bilstm**")
                st.write(s3)
                st.progress(int(c3))
            with p4:
                st.write("**distilbert**")
                st.write(s4)
                st.progress(int(c4))

    st.stop()

# --- SESSION STATE INIT ---
if "found_clues" not in st.session_state:
    st.session_state.found_clues = []

if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0

if "enable_voice" not in st.session_state:
    st.session_state.enable_voice = False

clue_analyzer = load_classifier()
deception_analyzer = load_deception_model()

# --- SIDEBAR ---
with st.sidebar:
    if st.button("⬅️ Back to Main Menu", use_container_width=True):
        for key in ["game_mode", "messages", "found_clues", "groq_client"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    mode_titles = {
        "simulator": "🕵️‍♂️ Detective Simulator",
        "akinator": "🧞‍♂️ Akinator Suspect",
        "interrogator": "😈 Witty Interrogator",
        "analyst": "🔍 Crime Scene Analyst"
    }
    st.success(f"Mode: {mode_titles.get(st.session_state.game_mode, 'Unknown')}")
    
    if st.session_state.game_mode == "simulator":
        role = st.session_state.get('sandbox_role', 'Unknown')
        if role == 'Criminal':
            st.error(f"🎭 Playing as: **{role}**")
        elif role == 'Victim':
            st.info(f"🎭 Playing as: **{role}**")
        elif role == 'Explorer':
            st.success(f"🎭 Playing as: **{role}**")
        else:
            st.warning(f"🎭 Playing as: **{role}**")
            
        st.markdown(f"**Synopsis:** _{st.session_state.get('sandbox_scenario', 'Unknown')}_")
    
    st.write("<br><br>", unsafe_allow_html=True)
    st.divider()
    
    st.header("🗂️ Case Evidence")
    
    st.subheader("📷 Visual Evidence (Images only)")
    uploaded_media = st.file_uploader("Upload scene photo...", type=["jpg", "jpeg", "png"])
    if uploaded_media:
        st.image(uploaded_media, use_container_width=True)
            
    st.write("<br>", unsafe_allow_html=True)
    st.divider()
    
    if st.session_state.game_mode == "akinator":
        st.subheader("🧠 Personality Trackers")
    else:
        st.subheader("🔍 Identified Clues Tracker")
        
    if st.session_state.found_clues:
        for i, clue in enumerate(st.session_state.found_clues, 1):
            if isinstance(clue, dict):
                st.markdown(f"{i}. <a href='#msg-{clue['idx']}' style='color:#e0e0e0; text-decoration:underline;'>{clue['text']}</a>", unsafe_allow_html=True)
            else:
                st.markdown(f"{i}. {clue}")
    else:
        if st.session_state.game_mode == "akinator":
            st.markdown("*No traits gathered yet...*")
        else:
            st.markdown("*No clues analyzed yet...*")

    st.write("<br><br>", unsafe_allow_html=True)
    st.divider()


# --- MAIN CHAT ---
st.subheader("🚨 Interrogation Room")

for i, msg in enumerate(st.session_state.messages):
    st.markdown(f"<div id='msg-{i}' style='position: relative; top: -80px;'></div>", unsafe_allow_html=True)
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "user" and "severity" in msg:
            if st.session_state.game_mode != "akinator":
                col1, col2, col3, col4 = st.columns(4)
                col1.caption(f"**ML Severity:** {msg['severity']}")
                col2.caption(f"**Confidence:** {msg['confidence']}%")
                col3.caption(f"**Text Tone:** {msg['tone']}")
                if 'deception' in msg:
                    col4.caption(f"**Deception:** {msg['deception']} ({msg['deception_conf']}%)")
            
        if "audio_bytes" in msg and msg["audio_bytes"]:
            st.audio(msg["audio_bytes"])
            
        # TTS Recovery for interrupted AI messages
        if msg.get("needs_tts") and msg.get("tts_bytes") is None:
            with st.spinner("Recovering interrupted audio..."):
                recovered_audio_path = generate_tts(msg["content"])
                if recovered_audio_path:
                    with open(recovered_audio_path, "rb") as f:
                        msg["tts_bytes"] = f.read()
            msg["needs_tts"] = False
            
        if "tts_bytes" in msg and msg["tts_bytes"]:
            # Display standard audio UI controls
            st.audio(msg["tts_bytes"], format="audio/mp3")
            
            # Persistent Autoplay via global parent document DOM (survives Theme changes / reruns)
            is_latest = (i == len(st.session_state.messages) - 1)
            if is_latest and msg["role"] == "assistant" and st.session_state.get('enable_voice', False):
                b64 = base64.b64encode(msg["tts_bytes"]).decode()
                msg_id = f"audio_msg_{i}"
                js = f"""
                <script>
                    const parentDoc = window.parent.document;
                    let audioPlayer = parentDoc.getElementById('global-audio-player');
                    if (!audioPlayer) {{
                        audioPlayer = parentDoc.createElement('audio');
                        audioPlayer.id = 'global-audio-player';
                        audioPlayer.style.display = 'none';
                        parentDoc.body.appendChild(audioPlayer);
                    }}
                    if (audioPlayer.getAttribute('data-src-id') !== '{msg_id}') {{
                        audioPlayer.src = "data:audio/mp3;base64,{b64}";
                        audioPlayer.setAttribute('data-src-id', '{msg_id}');
                        audioPlayer.play().catch(e => console.log('Autoplay blocked:', e));
                    }}
                </script>
                """
                components.html(js, height=0, width=0)

# Place inputs and toggle at the bottom of the chat
input_col, toggle_col = st.columns([4, 1])
with toggle_col:
    st.toggle("🔊 Voice Replies", value=False, key="enable_voice")

user_text = st.chat_input("Type your response here...")
with input_col:
    audio_bytes = st.audio_input("Or record your statement...", key=f"audio_recorder_{len(st.session_state.messages)}")

if user_text or audio_bytes:
    final_user_text = ""
    
    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(audio_bytes.getvalue())
            tmp_audio_path = tmp_audio.name
        
        try:
            with open(tmp_audio_path, "rb") as audio_file:
                transcription = st.session_state.groq_client.audio.transcriptions.create(
                    file=(tmp_audio_path, audio_file.read()),
                    model="whisper-large-v3",
                    response_format="text"
                )
            # whisper returns the text string directly if response_format="text"
            audio_text = transcription
            if user_text:
                final_user_text = f"{user_text}\n[User also said via audio]: {audio_text}"
            else:
                final_user_text = audio_text
        except Exception as e:
            st.error(f"Audio Transcription Error: {e}")
            final_user_text = user_text if user_text else ""
    else:
        final_user_text = user_text

    if not final_user_text:
        st.stop()

    sentiment = {"tone": "Unknown"}
    severity = "Unknown"
    confidence = 0.0
    
    severity, confidence = classify_clue(clue_analyzer, final_user_text)
    sentiment = analyze_sentiment(final_user_text)
    
    deception_status, deception_conf = analyze_deception(deception_analyzer, final_user_text)
    if st.session_state.game_mode == "interrogator":
        final_user_text += f"\n[System Polygraph: Statement is {deception_status} with {deception_conf}% confidence]"
    
    # Track Clues or Traits depending on mode
    if st.session_state.game_mode == "akinator":
        # Track answers to the Akinator's questions
        ans_raw = final_user_text.lower().strip()
        if "[system audio analysis" in ans_raw:
            ans_raw = ans_raw.split("[system audio analysis")[0].strip()
        ans = ans_raw.strip(".,!?")
        
        last_msg = st.session_state.messages[-1]["content"] if st.session_state.messages else ""
        
        # Extract just the specific question without the intro fluff
        q_parts = [s.strip() for s in last_msg.split('?') if s.strip()]
        last_q = (q_parts[-1] + '?') if '?' in last_msg else last_msg
        if ':' in last_q:
            last_q = last_q.split(':')[-1].strip()
            
        target_idx = len(st.session_state.messages) - 1 # Link to the AI's question
        if target_idx < 0: target_idx = 0
        
        if ans in ["yes", "yeah", "yep", "y"]:
            st.session_state.found_clues.append({"text": f"✅ {last_q}", "idx": target_idx})
        elif ans in ["no", "nah", "nope", "n"]:
            st.session_state.found_clues.append({"text": f"❌ {last_q}", "idx": target_idx})
    else:
        # Track Clues if they are highly suspicious or criminal
        if "MAJOR FELONY" in severity or "CRIME RELATED" in severity or "SUSPICIOUS" in severity:
            target_idx = len(st.session_state.messages) # Link to the User's new statement
            st.session_state.found_clues.append({"text": final_user_text, "idx": target_idx})

    # Audio analysis context
    if audio_bytes:
        final_user_text += f"\n[System Audio Analysis: User voice sounded {sentiment['tone']}]"

    user_msg_obj = {
        "role": "user", 
        "content": final_user_text,
        "severity": severity,
        "confidence": confidence,
        "tone": sentiment['tone'],
        "deception": deception_status,
        "deception_conf": deception_conf
    }
    if audio_bytes:
        user_msg_obj["audio_bytes"] = audio_bytes.getvalue()
    st.session_state.messages.append(user_msg_obj)
    
    with st.chat_message("user"):
        st.markdown(final_user_text)
        if st.session_state.game_mode != "akinator":
            col1, col2, col3, col4 = st.columns(4)
            col1.caption(f"**ML Severity:** {severity}")
            col2.caption(f"**Confidence:** {confidence}%")
            col3.caption(f"**Text Tone:** {sentiment['tone']}")
            col4.caption(f"**Deception:** {deception_status} ({deception_conf}%)")
        if audio_bytes:
            st.audio(audio_bytes)
            
    st.rerun()

# --- AI GENERATION BLOCK ---
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    
    # Build Groq Messages
    groq_messages = [{"role": "system", "content": st.session_state.system_prompt}]
    
    for i, msg in enumerate(st.session_state.messages):
        if i == len(st.session_state.messages) - 1 and msg["role"] == "user" and uploaded_media:
            # Inject image into the latest user message
            base64_image = encode_image(uploaded_media)
            groq_messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": msg["content"]},
                    {"type": "image_url", "image_url": {"url": base64_image}}
                ]
            })
        else:
            groq_messages.append({"role": msg["role"], "content": msg["content"]})
            
    # Decide Model (Vision vs Text)
    model_to_use = "llama-3.2-11b-vision-preview" if uploaded_media else "llama-3.1-8b-instant"
            
    with st.chat_message("assistant"):
        msg_box = st.empty()
        msg_box.markdown("*(Detective is thinking...)*")
        
        try:
            response = st.session_state.groq_client.chat.completions.create(
                model=model_to_use,
                messages=groq_messages,
                temperature=0.7,
                max_tokens=1024
            )
            reply = response.choices[0].message.content
            
            # Immediately append text to prevent data loss if user interrupts script (e.g. changing theme) during TTS
            st.session_state.messages.append({
                "role": "assistant", 
                "content": reply,
                "tts_bytes": None,
                "needs_tts": st.session_state.get('enable_voice', False),
                "played": False
            })
            
            msg_box.markdown(reply)
            
            if st.session_state.messages[-1]["needs_tts"]:
                tts_audio_path = generate_tts(reply)
                if tts_audio_path:
                    with open(tts_audio_path, "rb") as f:
                        tts_audio_data = f.read()
                    
                    # Store generated audio in the appended message safely
                    st.session_state.messages[-1]["tts_bytes"] = tts_audio_data
                    st.session_state.messages[-1]["needs_tts"] = False
                    
            st.rerun()
            
        except Exception as e:
            msg_box.error(f"Groq API Error: {e}")