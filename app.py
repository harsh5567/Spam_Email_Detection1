
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
    font-size: 16px;
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
div.stButton > button:hover {
    border: 1px solid white;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🛡️ SpamGuard AI")
    st.caption("Intelligent Email Security")
    st.divider()
    st.subheader("About Project")
    st.write("Machine Learning based email spam detection system.")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Text Processing:** TF-IDF")
    st.write("**Model:** Trained Spam Detection Pipeline")
    st.divider()
    st.caption("Developed by Harsh Tiwari | Harsh Pandey")

# Header
st.markdown("""
<div class="hero">
    <h1>🛡️ Spam Email Detection</h1>
    <p>Analyze your email and identify whether it is Spam or Not Spam using Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

# Sample emails
st.subheader("📧 Email Analyzer")
sample = st.selectbox(
    "Try an example email",
    [
        "Choose an example",
        "Spam Email Example",
        "Not Spam Email Example"
    ]
)

sample_data = {
    "Spam Email Example": (
        "offers@gmail.com",
        "You won a free iPhone!",
        "Congratulations! Click here to claim your free prize now."
    ),
    "Not Spam Email Example": (
        "teacher@college.edu",
        "Assignment submission",
        "Your assignment has been received successfully. Thank you."
    )
}

default_sender, default_subject, default_body = sample_data.get(
    sample, ("", "", "")
)

# Input section
with st.form("email_form"):
    col1, col2 = st.columns(2)

    with col1:
        sender = st.text_input(
            "👤 Sender Email",
            value=default_sender,
            placeholder="example@gmail.com"
        )

    with col2:
        subject = st.text_input(
            "📝 Email Subject",
            value=default_subject,
            placeholder="Enter email subject"
        )

    email_text = st.text_area(
        "✉️ Email Content",
        value=default_body,
        placeholder="Paste your complete email content here...",
        height=180
    )

    submitted = st.form_submit_button("🔍 Analyze Email")

# Prediction
if submitted:

    if not sender.strip() or not subject.strip() or not email_text.strip():
        st.warning("Please fill all three fields.")

    else:
        # Combine inputs in the same order used during training
        new_email = sender + " " + subject + " " + email_text

        prediction = model.predict([new_email])[0]
        prediction_label = str(prediction).strip().lower()

        st.divider()
        st.subheader("📊 Analysis Result")

        if prediction_label == "spam":
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

        # Show model probability if available
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([new_email])[0]
            classes = list(model.classes_)
            predicted_index = classes.index(prediction)
            probability = float(probabilities[predicted_index])

            st.metric(
                "Model Prediction Probability",
                f"{probability * 100:.2f}%"
            )
            st.progress(probability)

        st.caption(
            "Note: Predictions are based on the trained dataset. "
            "Always verify suspicious emails before clicking links."
        )

# Footer
st.divider()
st.caption(
    "SpamGuard AI | Built with Python, Streamlit, TF-IDF and Logistic Regression"
)

