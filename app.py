# app.py
import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from transformers import pipeline

# ==========================
# Load dataset
# ==========================
df = pd.read_csv('D:\\mizo_mood_data.csv', quotechar='"')

# Tokenize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['mood']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# ==========================
# Replies by mood
# ==========================
suggestions = {
    'happy': [
        "Ka hria ve, ka lawm e! 😄 - Oh! I'm glad that you’re happy",
        "Ka nuam tak e! 😄 - Yay! That makes me happy too"
    ],
    'sad': [
        "Ka hria hle che 😔 - Oh! I’m sorry to hear that",
        "Ka thinlung a luang hle 😢 - I hope things get better soon"
    ],
    'angry': [
        "Ka lo hria e 😤 - Take a deep breath, stay calm",
        "Ka thinlung a boh hle 😡 - I understand, try to relax"
    ],
    'neutral': [
        "Ka hria che 🙂 - Alright, noted",
        "Ka lo hmu tawh a ni 🙂 - Okay, got it"
    ]
}

meaning_enhanced = {
    'happy': [
        "You sound really joyful and bright today! 😊",
        "That’s such positive energy — keep smiling! 😄",
        "I can feel your happiness through your words! 🌞"
    ],
    'sad': [
        "It sounds like your heart is heavy. Things will get better. 💙",
        "I can sense sadness — take it easy on yourself. 🌧️",
        "It’s okay to feel low sometimes. You’re not alone. 🤍"
    ],
    'angry': [
        "I sense frustration — maybe take a deep breath first. 😤",
        "Anger can be hard to handle. Try calming your thoughts. 🧘",
        "It’s okay to be mad — just don’t let it control you. 💢"
    ],
    'neutral': [
        "You seem calm and balanced. 🙂",
        "Alright, noted — sounds neutral. 👍",
        "Okay, I understand. Thanks for sharing. 🙂"
    ]
}

# ==========================
# Translation model
# ==========================
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

# ==========================
# Caching
# ==========================
translation_cache = {}
prediction_cache = {}

def get_translation(text):
    if text in translation_cache:
        return translation_cache[text]
    translation = translator(text)[0]['translation_text']


# ==========================
# Initialize session state
# ==========================
if "users" not in st.session_state:
    # Dummy user database: username -> password
    st.session_state.users = {"achu": "password123"}  

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "page" not in st.session_state:
    st.session_state.page = "login"  # login or signup or chatbot

# ==========================
# Functions
# ==========================
def signup(username, password):
    if username in st.session_state.users:
        st.error("Username already exists!")
    else:
        st.session_state.users[username] = password
        st.success("Account created! You can now login.")
        st.session_state.page = "login"

def login(username, password):
    if username in st.session_state.users and st.session_state.users[username] == password:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.page = "chatbot"
        st.success(f"Welcome, {username}!")
    else:
        st.error("Invalid username or password")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.page = "login"

# ==========================
# Page Navigation
# ==========================
if st.session_state.page == "signup":
    st.title("📝 Sign Up")
    new_username = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    if st.button("Create Account"):
        signup(new_username, new_password)
    if st.button("Go to Login"):
        st.session_state.page = "login"
    st.stop()

elif st.session_state.page == "login":
    st.title("🔒 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        login(username, password)
    if st.button("Sign Up"):
        st.session_state.page = "signup"
    st.stop()

# ==========================
# Load dataset and train model
# ==========================
df = pd.read_csv('D:\\mizo_mood_data.csv', quotechar='"')

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['mood']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Replies by mood
suggestions = {
    'happy': ["Ka hria ve, ka lawm e! 😄", "Ka nuam tak e! 😄"],
    'sad': ["Ka hria hle che 😔", "Ka thinlung a luang hle 😢"],
    'angry': ["Ka lo hria e 😤", "Ka thinlung a boh hle 😡"],
    'neutral': ["Ka hria che 🙂", "Ka lo hmu tawh a ni 🙂"]
}

meaning_enhanced = {
    'happy': ["You sound really joyful today! 😊", "Keep smiling! 😄"],
    'sad': ["It sounds like your heart is heavy 💙", "Take it easy 🌧️"],
    'angry': ["I sense frustration 😤", "Try calming your thoughts 🧘"],
    'neutral': ["You seem calm and balanced 🙂", "Alright, noted 👍"]
}

# Translation model
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

# Caching
translation_cache = {}
prediction_cache = {}

def get_translation(text):
    if text in translation_cache:
        return translation_cache[text]
    translation = translator(text)[0]['translation_text']
    translation_cache[text] = translation
    return translation

def get_prediction(text):
    if text in prediction_cache:
        return prediction_cache[text]
    vector = vectorizer.transform([text])
    mood = model.predict(vector)[0]
    prediction_cache[text] = mood
    return mood

# ==========================
# Chatbot Interface
# ==========================
st.title(f"🟣 Mizo Mood Chatbot )")
st.markdown(f"<h1 style='font-size:28px;'>Mizo Mood Chatbot</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='font-size:14px; color:gray;'>Logged in as {st.session_state.username}</p>", unsafe_allow_html=True)


if st.button("Logout"):
    logout()
    st.experimental_rerun()

st.write("Type a message in Mizo and see mood prediction + English translation!")

user_input = st.text_input("You (Mizo):")

if user_input:
    predicted_mood = get_prediction(user_input)
    translation = get_translation(user_input)
    reply_mizo = random.choice(suggestions[predicted_mood])
    reply_english = random.choice(meaning_enhanced[predicted_mood])

    st.markdown(f"**🧠 Predicted Mood:** {predicted_mood}")
    st.markdown(f"**🗣️ English Translation:** {translation}")
    st.markdown(f"**🤖 Bot Reply (Mizo):** {reply_mizo}")
    st.markdown(f"**💬 Suggestion in English:** {reply_english}")

