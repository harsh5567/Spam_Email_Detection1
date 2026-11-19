

import streamlit as st
import joblib

# Page configuration
st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("spam_model.joblib")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1120, #111b32);
    color: white;
}
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}
.hero {
    background: linear-gradient(120deg, #202e52, #49347c);
    padding: 30px;
    border-radius: 20px;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 38px;
}
.hero p {
    color: #d5d9ed;
}
.result {
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    margin-top: 20px;
}
.safe {
    background: #123c35;
    border: 1px solid #20b486;
}
.danger {
    background: #49232f;
    border: 1px solid #ef6478;
}
div.stButton > button {
    background: linear-gradient(90deg, #6554d9, #9671ee);
    color: white;
    border: none;
    border-radius: 10px;
    height: 48px;
    font-weight: bold;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🛡️ SpamGuard AI")
    st.caption("Intelligent Email Security")
    st.divider()

    st.subheader("About Project")
    st.write("Machine Learning based email spam detection.")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Text Processing:** TF-IDF")

    st.divider()
    st.caption("Developed by Harsh Tiwari | Harsh Pandey")

# Header
st.markdown("""
<div class="hero">
    <h1>🛡️ Spam Email Detection</h1>
    <p>Analyze your email and identify whether it is Spam or Not Spam using Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

# Sample email data
sample_data = {
    "Spam Email Example": (
        "offers@gmail.com",
        "You won a free iPhone!",
        "Congratulations! You have won a free iPhone. Click here to claim your prize now. Limited time offer!"
    ),

    "Not Spam Email Example": (
        "teacher@college.edu",
        "Assignment submission confirmation",
        "Hello, your assignment has been received successfully. Thank you for submitting it before the deadline."
    )
}

# Function to load sample email
def load_sample():
    selected = st.session_state.sample

    if selected in sample_data:
        sender, subject, body = sample_data[selected]

        st.session_state.sender = sender
        st.session_state.subject = subject
        st.session_state.email_text = body

# Email Analyzer
st.subheader("📧 Email Analyzer")

st.selectbox(
    "Try an example email",
    [
        "Choose an example",
        "Spam Email Example",
        "Not Spam Email Example"
    ],
    key="sample",
    on_change=load_sample
)

# Email input form
with st.form("email_form"):

    col1, col2 = st.columns(2)

    with col1:
        sender = st.text_input(
            "👤 Sender Email",
            key="sender"
        )

    with col2:
        subject = st.text_input(
            "📝 Email Subject",
            key="subject"
        )

    email_text = st.text_area(
        "✉️ Email Content",
        key="email_text",
        height=180
    )

    submitted = st.form_submit_button("🔍 Analyze Email")

# Prediction
if submitted:

    if not sender.strip() or not subject.strip() or not email_text.strip():
        st.warning("Please fill all three fields.")

    else:
        # Combine all inputs
        new_email = sender + " " + subject + " " + email_text

        # Predict email
        prediction = model.predict([new_email])[0]

        st.divider()
        st.subheader("📊 Analysis Result")

        if str(prediction).strip().lower() == "spam":

            st.markdown("""
            <div class="result danger">
                <h1>🚨 SPAM EMAIL</h1>
                <p>This email has been classified as Spam.</p>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result safe">
                <h1>✅ NOT SPAM</h1>
                <p>This email has been classified as Not Spam.</p>
            </div>
            """, unsafe_allow_html=True)

        # Prediction probability
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba([new_email])[0]
            classes = list(model.classes_)

            index = classes.index(prediction)
            probability = float(probabilities[index])

            st.metric(
                "Prediction Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(probability)

        st.caption(
            "Note: Predictions depend on the trained dataset. "
            "Always verify suspicious emails."
        )

# Footer
st.divider()

st.caption(
    "SpamGuard AI | Python | Streamlit | TF-IDF | Logistic Regression"
)
