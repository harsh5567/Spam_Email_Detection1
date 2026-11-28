"""Spam Email Detector - Streamlit app. Run: streamlit run app.py"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "spam_model_enron.joblib"
TEXT_COLUMNS = ["email_text", "message", "text"]

st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="wide")


STYLE = """
<style>
/* Background */
.stApp {background: radial-gradient(circle at top, #17213d 0%, #070a12 60%); color: #f2f7ff;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"] {background: #0b1120; border-right: 1px solid #00d9ff55;}

/* All normal text light */
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp li, .stApp small,
[data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"], [data-testid="stMetricValue"],
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {color: #f2f7ff !important;}

/* Heading */
.hero {text-align: center; font-size: 2.6rem; font-weight: 800; color: #fff;
       text-shadow: 0 0 8px #00d9ff, 0 0 24px #00d9ff; margin-bottom: .2rem;}
.tagline {text-align: center; color: #a9d4ff; margin-bottom: 1.2rem;}

/* Tabs */
button[data-baseweb="tab"] p {color: #a9bcd6 !important; font-weight: 700;}
button[data-baseweb="tab"][aria-selected="true"] p {color: #00d9ff !important;}

/* Inputs */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"] {
       background: #111a2e !important; border-color: #00d9ff88 !important;}
.stTextInput input, .stTextArea textarea {background: #111a2e !important;
       color: #ffffff !important; -webkit-text-fill-color: #ffffff; caret-color: #00d9ff;}
[data-testid="InputInstructions"] {color: #a9bcd6 !important;}

/* Buttons */
.stButton button, .stFormSubmitButton button, .stDownloadButton button,
[data-testid="stFileUploader"] button {background: #0d1424 !important;
       border: 2px solid #00d9ff !important; border-radius: 12px;
       box-shadow: 0 0 12px #00d9ff88;}
.stApp button:not([data-baseweb="tab"]), .stApp button:not([data-baseweb="tab"]) p {
       color: #00e5ff !important; font-weight: 700;}

/* File upload box */
[data-testid="stFileUploaderDropzone"] {background: #111a2e !important;
       border: 1px dashed #00d9ff;}
[data-testid="stFileUploader"] span, [data-testid="stFileUploader"] small,
[data-testid="stFileUploaderDropzone"] div {color: #f2f7ff !important;}

/* Message boxes */
[data-testid="stAlert"] {background: #14233f; border-radius: 12px;}

/* Blinking result box */
.result {--neon: #39ff14; border: 4px solid var(--neon); border-radius: 18px;
       padding: 28px 16px; margin: 18px 0; text-align: center; background: #05070d;
       animation: blink 1s ease-in-out infinite;}
.result.spam {--neon: #ff073a;}
.result .icon {font-size: 3.5rem;}
.result .title {font-size: 2.4rem; font-weight: 900; color: var(--neon) !important;
       text-shadow: 0 0 10px var(--neon), 0 0 30px var(--neon);}
.result .score {color: #f2f7ff; font-size: 1.1rem;}
@keyframes blink {
  0%, 100% {box-shadow: 0 0 10px var(--neon), 0 0 35px var(--neon),
            0 0 70px var(--neon), inset 0 0 25px var(--neon); opacity: 1;}
  50% {box-shadow: 0 0 3px var(--neon); opacity: .6;}
}
/* Sidebar info box */
[data-testid="stAlert"], [data-testid="stAlert"] > div,
[data-testid="stAlertContainer"], [data-baseweb="notification"] {
       background: #14233f !important; color: #f2f7ff !important;}
[data-testid="stAlert"] p {color: #f2f7ff !important;}

/* Help tooltip, popup menu and dialog */
[data-baseweb="tooltip"], [data-baseweb="tooltip"] div,
[data-testid="stTooltipContent"],
[data-baseweb="popover"] > div, [data-baseweb="popover"] ul,
[role="dialog"] {background: #111a2e !important; color: #f2f7ff !important;}
[data-baseweb="tooltip"] p, [data-testid="stTooltipContent"] p,
[data-baseweb="popover"] li, [data-baseweb="popover"] span,
[role="dialog"] p, [role="dialog"] label, [role="dialog"] span {color: #f2f7ff !important;}
</style>
"""


def result_card(probability, is_spam):
    """Blinking neon box: red for spam, green for not spam."""
    kind, icon, title = (("spam", "🚨", "SPAM DETECTED") if is_spam
                         else ("safe", "🛡️", "NOT SPAM"))
    return (f'<div class="result {kind}"><div class="icon">{icon}</div>'
            f'<div class="title">{title}</div>'
            f'<div class="score">Spam probability: {probability:.1%}</div></div>')


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def spam_probability(model, texts):
    spam_index = list(model.classes_).index("spam")
    return model.predict_proba(texts)[:, spam_index]


def top_words(model, text, n=8):
    """Words that pushed this email towards spam or not spam."""
    vectorizer, classifier = model.steps[0][1], model.steps[-1][1]
    row = vectorizer.transform([text])
    # positive coefficient favours classes_[1]
    sign = 1 if classifier.classes_[1] == "spam" else -1
    scores = row.data * classifier.coef_[0][row.indices] * sign
    words = vectorizer.get_feature_names_out()[row.indices]
    table = pd.DataFrame({"word": words, "score": scores})
    spam = table[table.score > 0].nlargest(n, "score")
    safe = table[table.score < 0].nsmallest(n, "score")
    return spam, safe


def safe_cell(value):
    """Stop spreadsheet apps from running text as a formula."""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


st.markdown(STYLE, unsafe_allow_html=True)
st.markdown('<div class="hero">📧 Spam Email Detector</div>'
            '<div class="tagline">TF-IDF + Logistic Regression</div>',
            unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error(f"Model file not found. Put '{MODEL_PATH.name}' in the same folder as app.py.")
    st.stop()

model = load_model()
st.session_state.setdefault("history", [])

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Spam threshold", 0.10, 0.90, 0.50, 0.05,
                          help="Higher value: fewer emails are marked as spam.")
    st.info("This model can make mistakes, especially on short or new kinds of emails.")

single_tab, batch_tab, history_tab = st.tabs(["Check email", "Check CSV", "History"])

with single_tab:
    with st.form("email_form"):
        subject = st.text_input("Subject")
        body = st.text_area("Message", height=200)
        submitted = st.form_submit_button("Check", type="primary")

    if submitted:
        text = f"{subject} {body}".strip()
        if not text:
            st.warning("Please enter a subject or message.")
        else:
            probability = float(spam_probability(model, [text])[0])
            is_spam = probability >= threshold
            st.markdown(result_card(probability, is_spam), unsafe_allow_html=True)
            st.progress(probability)

            spam_words, safe_words = top_words(model, text)
            if spam_words.empty and safe_words.empty:
                st.warning("The model does not know any word in this email, "
                           "so do not trust this result.")
            else:
                left, right = st.columns(2)
                left.subheader("Words pointing to spam")
                left.dataframe(spam_words, hide_index=True)
                right.subheader("Words pointing to not spam")
                right.dataframe(safe_words, hide_index=True)

            st.session_state.history.append({
                "subject": subject,
                "result": "spam" if is_spam else "not spam",
                "spam_probability": round(probability, 4),
            })

with batch_tab:
    st.write("Upload a CSV with a message column (`email_text`, `Message` or `text`). "
             "A `subject` column is optional.")
    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded:
        try:
            data = pd.read_csv(uploaded)
        except (ValueError, UnicodeDecodeError) as error:
            st.error(f"Could not read this CSV: {error}")
        else:
            columns = {str(c).strip().lower(): c for c in data.columns}
            text_column = next((columns[c] for c in TEXT_COLUMNS if c in columns), None)
            if text_column is None:
                st.error("No message column found.")
            elif data.empty:
                st.warning("The CSV has no rows.")
            else:
                texts = data[text_column].fillna("").astype(str)
                if "subject" in columns:
                    texts = data[columns["subject"]].fillna("").astype(str) + " " + texts
                probabilities = spam_probability(model, texts)
                data["spam_probability"] = probabilities.round(4)
                data["prediction"] = ["spam" if p >= threshold else "not spam"
                                      for p in probabilities]

                total = len(data)
                spam_count = int((data.prediction == "spam").sum())
                a, b, c = st.columns(3)
                a.metric("Total", total)
                b.metric("Spam", spam_count)
                c.metric("Not spam", total - spam_count)
                st.dataframe(data, hide_index=True)

                export = data.apply(lambda column: column.map(safe_cell))
                st.download_button("Download results",
                                   export.to_csv(index=False).encode("utf-8"),
                                   "predictions.csv", "text/csv")

with history_tab:
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.write("No emails checked yet.")


