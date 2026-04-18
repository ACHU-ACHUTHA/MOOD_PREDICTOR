"""
Mood Predictor — College AI Project
Detects mood from text/voice in English, Telugu, Hindi, and Mizo.
Tech: Streamlit · scikit-learn · langdetect · SpeechRecognition · urllib (built-in)
"""

import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import urllib.request
import urllib.parse
import json

from database import init_db, save_prediction, load_predictions, clear_predictions

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


def _google_translate(text: str, target: str) -> str:
    """Translate text using Google Translate public endpoint — no API key needed."""
    params = urllib.parse.urlencode({
        "client": "gtx", "sl": "auto", "tl": target, "dt": "t", "q": text,
    })
    url = f"https://translate.googleapis.com/translate_a/single?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return "".join(part[0] for part in data[0] if part[0])

# ---------------------------------------------------------------------------
# Training Data
# ---------------------------------------------------------------------------

TRAINING_DATA = {
    "happy": [
        "I am so happy today", "This is amazing", "I feel wonderful",
        "Everything is going great", "I love this", "I feel fantastic",
        "Today is the best day", "I am joyful and excited",
        "Great news just arrived", "I am thrilled about this",
        "Feeling blessed and grateful", "Life is beautiful",
        "నేను చాలా సంతోషంగా ఉన్నాను", "ఈరోజు చాలా బాగుంది",
        "నాకు చాలా ఆనందంగా ఉంది", "అద్భుతంగా అనిపిస్తోంది",
        "मैं बहुत खुश हूँ", "आज का दिन बहुत अच्छा है",
        "मुझे बहुत अच्छा लग रहा है", "जिंदगी खूबसूरत है",
        "Ka lawmawm hle", "Ni tha tak a ni", "Ka hlim lutuk",
    ],
    "sad": [
        "I feel so sad", "Everything is hopeless", "I am crying",
        "I feel empty inside", "Nobody cares about me", "I am heartbroken",
        "Life feels meaningless", "I miss them so much",
        "I am feeling depressed", "Nothing makes me happy anymore",
        "I feel lonely all the time", "My heart is broken",
        "నాకు చాలా దుఃఖంగా ఉంది", "నేను ఒంటరిగా ఉన్నాను",
        "జీవితం నిస్సారంగా అనిపిస్తోంది", "నాకు ఏదీ నచ్చడం లేదు",
        "मुझे बहुत दुख हो रहा है", "मैं बहुत उदास हूँ",
        "जिंदगी बेकार लग रही है", "मैं अकेला हूँ",
        "Ka lungawi lo", "Ka hrechhuak hle", "Ka nuam lo",
    ],
    "angry": [
        "I am so angry", "This makes me furious", "I hate this situation",
        "I want to scream", "Everything is unfair", "I am filled with rage",
        "This is outrageous", "I cannot tolerate this anymore",
        "They make me so mad", "I am really frustrated with this",
        "Why does this keep happening", "I am absolutely livid",
        "నాకు చాలా కోపంగా ఉంది", "ఇది చాలా అన్యాయం",
        "నేను సహించలేకపోతున్నాను", "ఇది నన్ను కోపంగా చేస్తోంది",
        "मुझे बहुत गुस्सा आ रहा है", "यह बहुत अनुचित है",
        "मैं बहुत क्रोधित हूँ", "यह सहन नहीं होता",
        "Ka roh hle", "Ka innghat lo", "Ka lungngaih hle",
    ],
    "anxious": [
        "I am so worried", "I feel very nervous", "I cannot stop panicking",
        "I am scared about the future", "My heart is racing",
        "I feel anxious all the time", "I cannot sleep due to worry",
        "Everything makes me feel stressed", "I am overthinking everything",
        "I feel like something bad will happen", "I am trembling with fear",
        "The anxiety is overwhelming",
        "నాకు చాలా ఆందోళనగా ఉంది", "నేను భయపడుతున్నాను",
        "నాకు నిద్ర పట్టడం లేదు", "నాకు చాలా టెన్షన్ గా ఉంది",
        "मुझे बहुत चिंता हो रही है", "मैं बहुत घबराया हुआ हूँ",
        "मुझे डर लग रहा है", "मेरी नींद नहीं आ रही",
        "Ka lung a awi lo", "Ka hlau hle", "Ka ngaihtuah nasa hle",
    ],
    "neutral": [
        "Today was okay", "I feel fine", "Nothing special happened",
        "It was an average day", "I am doing alright",
        "Not good not bad", "Things are normal",
        "Just another regular day", "I feel okay I guess",
        "Everything is as usual",
        "ఈరోజు సాధారణంగా ఉంది", "నేను బాగానే ఉన్నాను",
        "ప్రత్యేకంగా ఏమీ జరగలేదు",
        "आज का दिन ठीक था", "मैं ठीक हूँ", "कुछ खास नहीं हुआ",
        "Ka tha", "A tlangpui a ni", "A dik tawh",
    ],
}

# ---------------------------------------------------------------------------
# Static Config
# ---------------------------------------------------------------------------

MOOD_TRANSLATIONS = {
    "te": {
        "happy": "సంతోషం", "sad": "దుఃఖం", "angry": "కోపం",
        "anxious": "ఆందోళన", "neutral": "సాధారణం",
    },
    "hi": {
        "happy": "खुश", "sad": "दुखी", "angry": "गुस्सा",
        "anxious": "चिंतित", "neutral": "सामान्य",
    },
    "mzo": {
        "happy": "Lawmawm", "sad": "Lungawi lo", "angry": "Roh",
        "anxious": "Hlau", "neutral": "Tlangpui",
    },
}

MOOD_CONFIG = {
    "happy": {
        "emoji": "😊", "severity": "Low", "color": "#2ecc71",
        "suggestions": [
            "🎉 Keep spreading your positivity!",
            "📝 Journal about what made you happy today.",
            "🤝 Share your joy with someone you love.",
            "🎵 Play your favourite upbeat music.",
            "🌿 Take a gratitude walk outside.",
        ],
        "breathing": None,
    },
    "sad": {
        "emoji": "😢", "severity": "Medium", "color": "#3498db",
        "suggestions": [
            "💬 Talk to a trusted friend or family member.",
            "🎵 Listen to calming, uplifting music.",
            "📖 Read an inspiring book or watch a feel-good movie.",
            "🚶 Go for a short walk in fresh air.",
            "🛁 Take care of yourself — eat well and rest.",
        ],
        "breathing": "4-7-8 Breathing: Inhale 4 sec → Hold 7 sec → Exhale 8 sec. Repeat 4×.",
    },
    "angry": {
        "emoji": "😠", "severity": "Medium", "color": "#e74c3c",
        "suggestions": [
            "🧘 Pause and take 5 slow deep breaths before reacting.",
            "🚶 Walk away from the situation for 10 minutes.",
            "💦 Splash cold water on your face.",
            "✍️ Write down what's frustrating you — it helps!",
            "🎶 Listen to calming music to cool down.",
        ],
        "breathing": "Box Breathing: Inhale 4 sec → Hold 4 sec → Exhale 4 sec → Hold 4 sec. Repeat.",
    },
    "anxious": {
        "emoji": "😰", "severity": "High", "color": "#f39c12",
        "suggestions": [
            "🆘 If severe, please reach iCall helpline: 9152987821",
            "🧘 Try 5-4-3-2-1 grounding: name 5 things you see, 4 you hear…",
            "📵 Reduce screen time and social media for a while.",
            "🛌 Maintain a regular sleep schedule.",
            "🗣️ Consider speaking with a counsellor.",
        ],
        "breathing": "4-4-4 Breathing: Inhale 4 sec → Hold 4 sec → Exhale 4 sec. Repeat 6×.",
    },
    "neutral": {
        "emoji": "😐", "severity": "Low", "color": "#95a5a6",
        "suggestions": [
            "🌱 Try something new today to spark some excitement.",
            "📚 Learn a skill you've been putting off.",
            "☕ Enjoy a quiet moment with tea/coffee.",
            "🤝 Reach out to an old friend.",
            "🎨 Try a creative hobby — draw, write, cook.",
        ],
        "breathing": None,
    },
}

SEVERITY_MESSAGES = {
    "Low":    ("✅ Mild — You're doing well! Small tips below to keep it up.", "#2ecc71"),
    "Medium": ("⚠️ Moderate — Consider talking to someone you trust.",        "#f39c12"),
    "High":   ("🚨 High — Please seek support. You are not alone!",            "#e74c3c"),
}

LANG_NAMES = {
    "en": "English",
    "te": "Telugu / తెలుగు",
    "hi": "Hindi / हिंदी",
    "mzo": "Mizo",
    "unknown": "Unknown",
}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@st.cache_resource
def train_model():
    """Build and train a TF-IDF + Logistic Regression pipeline."""
    texts, labels = [], []
    for mood, sentences in TRAINING_DATA.items():
        for sentence in sentences:
            texts.append(sentence)
            labels.append(mood)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=5000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5,
            solver="lbfgs",
            multi_class="auto",
        )),
    ])
    model.fit(texts, labels)
    return model

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    if not LANGDETECT_AVAILABLE:
        return "en"
    try:
        return detect(text)
    except Exception:
        return "en"


def translate_to_english(text: str) -> str:
    try:
        return _google_translate(text, "en")
    except Exception:
        return text


def translate_mood_label(mood: str, lang: str) -> str:
    if lang in MOOD_TRANSLATIONS and mood in MOOD_TRANSLATIONS[lang]:
        return MOOD_TRANSLATIONS[lang][mood]
    if lang not in ("en", "unknown"):
        try:
            return _google_translate(mood, lang)
        except Exception:
            pass
    return mood.capitalize()


def translate_text(text: str, dest_lang: str) -> str:
    if dest_lang in ("en", "unknown"):
        return text
    try:
        return _google_translate(text, dest_lang)
    except Exception:
        return text


def predict_mood(text: str, model) -> tuple[str, float]:
    probs = model.predict_proba([text])[0]
    idx = np.argmax(probs)
    return model.classes_[idx], float(probs[idx])


def record_voice() -> str:
    if not SPEECH_AVAILABLE:
        return ""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening… speak now!")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=7, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            st.warning("No speech detected. Please try again.")
            return ""
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        st.warning("Could not understand audio. Please speak clearly.")
        return ""
    except sr.RequestError:
        st.error("Speech service unavailable. Check your internet connection.")
        return ""

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    init_db()

    st.set_page_config(page_title="Mood Predictor 🎭", page_icon="🎭", layout="centered")

    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
        .main-title {
            text-align: center; font-size: 2.6rem; font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .subtitle { text-align: center; color: #888; font-size: 1rem; margin-bottom: 2rem; }
        .mood-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 16px; padding: 24px 28px; margin-top: 1.5rem;
            border-left: 6px solid; color: #f0f0f0;
        }
        .mood-title { font-size: 1.8rem; font-weight: 800; margin-bottom: 4px; }
        .severity-badge {
            display: inline-block; padding: 4px 14px; border-radius: 20px;
            font-weight: 700; font-size: 0.85rem; margin-bottom: 14px;
        }
        .suggestion-item {
            background: rgba(255,255,255,0.07); border-radius: 10px;
            padding: 8px 14px; margin: 6px 0; font-size: 0.95rem;
        }
        .breathing-box {
            background: rgba(102,126,234,0.15); border-radius: 10px;
            padding: 12px 16px; margin-top: 12px;
            border: 1px solid rgba(102,126,234,0.4); font-size: 0.9rem;
        }
        .confidence-bar-bg {
            background: rgba(255,255,255,0.1); border-radius: 20px;
            height: 10px; margin: 8px 0 14px; overflow: hidden;
        }
        .confidence-bar-fill { height: 10px; border-radius: 20px; transition: width 0.5s ease; }
        .lang-badge {
            display: inline-block; background: rgba(255,255,255,0.12);
            border-radius: 12px; padding: 3px 12px; font-size: 0.8rem; margin-bottom: 10px;
        }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none; color: white; font-weight: 700;
        }
        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #5a6fd6 0%, #6a4292 100%);
            border: none; color: white;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🎭 Mood Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Detect your mood in English · తెలుగు · हिंदी · Mizo</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading AI model…"):
        model = train_model()

    st.markdown("### 💬 How are you feeling?")

    user_input = st.text_area(
        label="Type in any language:",
        placeholder="e.g. I feel amazing today!  /  నాకు చాలా సంతోషంగా ఉంది  /  मुझे बहुत चिंता हो रही है",
        height=120,
        key="text_input",
    )

    _, mid_col, _ = st.columns([1, 2, 1])
    with mid_col:
        predict_btn = st.button("🔍 Predict My Mood", use_container_width=True, type="primary")

    if predict_btn or (user_input and st.session_state.get("auto_predict")):
        if not user_input.strip():
            st.warning("Please type something or use voice input first.")
        else:
            lang_code = detect_language(user_input)
            lang_display = LANG_NAMES.get(lang_code, f"Detected: {lang_code}")

            translated_input = translate_to_english(user_input)
            mood, confidence = predict_mood(translated_input, model)
            config = MOOD_CONFIG[mood]

            mood_in_lang = translate_mood_label(mood, lang_code)
            severity = config["severity"]
            sev_msg, sev_color = SEVERITY_MESSAGES[severity]
            suggestions = config["suggestions"]
            border_color = config["color"]
            emoji = config["emoji"]

            card_html = f"""
            <div class="mood-card" style="border-color:{border_color};">
                <div class="lang-badge">🌐 {lang_display}</div>
                <div class="mood-title">{emoji} {mood.capitalize()} &nbsp;<span style="opacity:0.6;font-size:1.2rem">/ {mood_in_lang}</span></div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width:{confidence*100:.0f}%;background:{border_color};"></div>
                </div>
                <div style="font-size:0.82rem;opacity:0.6;margin-bottom:12px;">Confidence: {confidence*100:.1f}%</div>
                <div class="severity-badge" style="background:{sev_color}22;color:{sev_color};border:1px solid {sev_color};">
                    {severity} Severity
                </div>
                <div style="margin-bottom:14px;font-size:0.95rem;">{sev_msg}</div>
                <div style="font-weight:700;margin-bottom:6px;">💡 Suggestions:</div>
                {"".join(f'<div class="suggestion-item">{s}</div>' for s in suggestions)}
                {f'<div class="breathing-box">🌬️ <b>Breathing Exercise:</b><br>{config["breathing"]}</div>' if config["breathing"] else ""}
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            # Save to database
            save_prediction(
                user_text=user_input,
                language=lang_display,
                mood=mood,
                confidence=confidence,
                severity=severity,
            )

            if lang_code not in ("en", "unknown"):
                with st.expander(f"📝 Suggestions in your language ({lang_display})"):
                    for s in suggestions:
                        st.markdown(f"- {translate_text(s, lang_code)}")

    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown("""
        
        **Supported Languages:**
        | Language | Script |
        |----------|--------|
        | English  | Latin  |
        | Telugu   | తెలుగు |
        | Hindi    | हिंदी  |
        | Mizo     | Latin  |

        **Moods Detected:**
        😊 Happy · 😢 Sad · 😠 Angry · 😰 Anxious · 😐 Neutral

        
        """)

        st.markdown("---")
        st.markdown("### 📞 Emergency Help")
        st.error("**Call (India):** 9152987821\n\n**helpline:** 1860-2662-345")

        st.markdown("---")
        st.markdown("### 🕓 Mood History")

        history = load_predictions(limit=50)
        if not history:
            st.info("No predictions yet. Try predicting your mood!")
        else:
            st.caption(f"{len(history)} recent entries")
            MOOD_EMOJI = {"happy": "😊", "sad": "😢", "angry": "😠", "anxious": "😰", "neutral": "😐"}
            SEV_COLOR  = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            for row in history:
                emoji = MOOD_EMOJI.get(row["mood"], "🎭")
                dot   = SEV_COLOR.get(row["severity"], "⚪")
                with st.expander(f"{emoji} {row['mood'].capitalize()}  {dot}  — {row['timestamp']}"):
                    st.write(f"**Input:** {row['user_text']}")
                    st.write(f"**Language:** {row['language']}")
                    st.write(f"**Confidence:** {row['confidence']*100:.1f}%")
                    st.write(f"**Severity:** {row['severity']}")

            st.markdown(" ")
            if st.button("🗑️ Clear History", use_container_width=True):
                clear_predictions()
                st.success("History cleared!")
                st.rerun()


if __name__ == "__main__":
    main()
