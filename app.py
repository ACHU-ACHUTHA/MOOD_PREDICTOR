import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from transformers import pipeline

# ==========================
# Initialize session state
# ==========================
if "users" not in st.session_state:
    st.session_state.users = {
        "achu": "password123",  # regular user
        "admin": "admin123"     # ✅ admin user
    }

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "page" not in st.session_state:
    st.session_state.page = "login"  # login / signup / chatbot
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}  # Dictionary to store chat history per user

# ==========================
# Authentication functions
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
        if username not in st.session_state.chat_histories:
            st.session_state.chat_histories[username] = []  # Initialize history for user
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
try:
    df = pd.read_csv('D:\\mizo_mood_data.csv', quotechar='"')
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['mood']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# ==========================
# Replies by mood
# ==========================
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

# ==========================
# Translation model
# ==========================
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
st.title("🟣 Mizo Mood Chatbot")
st.markdown(f"<p style='font-size:14px; color:gray;'>Logged in as {st.session_state.username}</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Logout"):
        logout()
        st.experimental_rerun()
with col2:
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_histories[st.session_state.username] = []

st.write("Type a message in Mizo and see the mood prediction + English translation!")

# Input box
user_input = st.text_input("You (Mizo):")

# Handle user input
if user_input:
    predicted_mood = get_prediction(user_input)
    translation = get_translation(user_input)
    reply_mizo = random.choice(suggestions.get(predicted_mood, ["Ka hria lo e 🙂"]))
    reply_english = random.choice(meaning_enhanced.get(predicted_mood, ["I'm not sure how to respond 🙂"]))

    # Save chat to current user's history
    st.session_state.chat_histories[st.session_state.username].append({
        "user": user_input,
        "translation": translation,
        "mood": predicted_mood,
        "bot_mizo": reply_mizo,
        "bot_english": reply_english
    })

# ==========================
# Display Chat History (Current User)
# ==========================
current_user_history = st.session_state.chat_histories.get(st.session_state.username, [])
if current_user_history:
    st.subheader("💬 Your Chat History")
    for chat in reversed(current_user_history):
        st.markdown(f"**🧍‍♂️ You (Mizo):** {chat['user']}")
        st.markdown(f"**🗣️ English Translation:** {chat['translation']}")
        st.markdown(f"**🧠 Predicted Mood:** {chat['mood']}")
        st.markdown(f"**🤖 Bot (Mizo):** {chat['bot_mizo']}")
        st.markdown(f"**💬 Suggestion (English):** {chat['bot_english']}")
        st.markdown("---")

# ==========================
# Admin View: See All User Histories
# ==========================
if st.session_state.username == "admin":
    st.subheader("📁 View Chat History by User")
    selected_user = st.selectbox("Select a user:", options=list(st.session_state.chat_histories.keys()))

    if selected_user:
        st.markdown(f"### 🧑‍💻 Chat History of `{selected_user}`")
        for chat in reversed(st.session_state.chat_histories[selected_user]):
            st.markdown(f"**🧍‍♂️ You (Mizo):** {chat['user']}")
            st.markdown(f"**🗣️ English Translation:** {chat['translation']}")
            st.markdown(f"**🧠 Predicted Mood:** {chat['mood']}")
            st.markdown(f"**🤖 Bot (Mizo):** {chat['bot_mizo']}")
            st.markdown(f"**💬 Suggestion (English):** {chat['bot_english']}")
            st.markdown("---")
