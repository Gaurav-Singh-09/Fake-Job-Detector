import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------
# Page Setup & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="JobGuard AI | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .badge-legit {
        background-color: #dcfce7;
        color: #166534;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 16px;
        display: inline-block;
    }
    .badge-fake {
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 16px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

SEQ_LEN = 50

# ---------------------------------------------------------
# Load Cached Assets
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading detection engine...")
def load_assets():
    model = load_model("fake_job_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

try:
    model, tokenizer = load_assets()
except Exception as e:
    st.error(f"⚠️ Error loading model/tokenizer files: {e}")
    st.stop()

# ---------------------------------------------------------
# Sidebar: Metadata & Sample Loader
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("JobGuard AI")
    st.caption("Deep Learning Fake Posting Classifier (RNN)")
    st.markdown("---")

    st.subheader("Model Evaluation Specs")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric("Test Accuracy", "99.55%")
        st.metric("Fake Recall (TPR)", "100.0%")
    with col_sb2:
        st.metric("Vocabulary", "10,000")
        st.metric("False Positive", "0.47%")

    st.markdown("---")
    st.subheader("Load Demo Presets")
    preset = st.radio(
        "Auto-fill fields with test cases:",
        ("Blank Form", "Typical Legitimate Job", "Suspicious/Fraudulent Job"),
        index=0
    )

# Preset Templates
presets = {
    "Blank Form": {
        "title": "", "company_profile": "", "salary_range": "",
        "requirements": "", "benefits": "", "location": "",
        "description": "", "industry": ""
    },
    "Typical Legitimate Job": {
        "title": "Marketing Intern",
        "company_profile": "We are Food52, a fast-growing digital publisher and lifestyle brand.",
        "salary_range": "30000-45000",
        "requirements": "Experience with CMS, social platforms, and strong written English.",
        "benefits": "Health coverage, 401(k), hybrid flexibility, and mentorship.",
        "location": "US, NY, New York",
        "description": "Looking for an energetic marketing intern to support editorial and social campaigns.",
        "industry": "Marketing / Online Media"
    },
    "Suspicious/Fraudulent Job": {
        "title": "Data Entry Representative - Immediate Start",
        "company_profile": "Global consultancy company providing flexible remote assignments.",
        "salary_range": "100000-120000",
        "requirements": "No prior experience needed. Basic typing skills and high-speed internet.",
        "benefits": "Full remote equipment check delivered upon fee transfer confirmation.",
        "location": "US, TX, Houston",
        "description": "Earn $50/hour working from home immediately. Wire processing fee for onboarding.",
        "industry": "Data Entry"
    }
}

active_preset = presets[preset]

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.markdown("## 🔍 Fake Job Posting Analysis")
st.markdown(
    "Analyze job listings using a recurrent neural network model trained on natural language features."
)

with st.form("job_input_form"):
    st.markdown("### Job Post Details")

    col1, col2 = st.columns([2, 1])
    with col1:
        title_in = st.text_input("Job Title", value=active_preset["title"], placeholder="e.g. Senior Software Engineer")
    with col2:
        salary_in = st.text_input("Salary Range", value=active_preset["salary_range"], placeholder="e.g. 60000-80000")

    col3, col4 = st.columns(2)
    with col3:
        location_in = st.text_input("Location", value=active_preset["location"], placeholder="e.g. US, CA, San Francisco")
    with col4:
        industry_in = st.text_input("Industry", value=active_preset["industry"], placeholder="e.g. Information Technology")

    st.markdown("---")
    company_profile_in = st.text_area("Company Profile", value=active_preset["company_profile"], height=90, placeholder="Brief background of the hiring entity...")
    description_in = st.text_area("Job Description", value=active_preset["description"], height=120, placeholder="Role details, scope, day-to-day responsibilities...")

    col5, col6 = st.columns(2)
    with col5:
        requirements_in = st.text_area("Requirements / Qualifications", value=active_preset["requirements"], height=90)
    with col6:
        benefits_in = st.text_area("Benefits & Perks", value=active_preset["benefits"], height=90)

    submitted = st.form_submit_button("🚀 Run Risk Analysis", use_container_width=True)

# ---------------------------------------------------------
# Inference & Results Dashboard
# ---------------------------------------------------------
if submitted:
    # Concatenate identical to training pipeline:
    # company_profile + " " + salary_range + " " + requirements + " " + benefits + " " + location + " " + description + " " + industry
    combined_text = (
        f"{company_profile_in}  {salary_in}  {requirements_in}  "
        f"{benefits_in}  {location_in}  {description_in}  {industry_in}"
    ).strip()

    if len(combined_text) == 0:
        st.warning("⚠️ Please provide text in at least one field before analyzing.")
    else:
        # Preprocessing
        sequences = tokenizer.texts_to_sequences([combined_text])
        padded = pad_sequences(sequences, maxlen=SEQ_LEN)

        # Inference
        probs = model.predict(padded)[0]
        legit_prob = float(probs[0])
        fake_prob = float(probs[1])
        prediction_class = int(np.argmax(probs))

        st.markdown("### 📊 Diagnostic Results")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if prediction_class == 1:
                st.markdown('<span class="badge-fake">FLAGGED AS FRAUDULENT</span>', unsafe_allow_html=True)
                st.subheader("🚨 High Risk Posting")
                st.write("This job description matches linguistic and formatting patterns strongly correlated with fraudulent listings.")
            else:
                st.markdown('<span class="badge-legit">VERIFIED AS LEGITIMATE</span>', unsafe_allow_html=True)
                st.subheader("✅ Likely Authentic")
                st.write("This job listing exhibits common markers of legitimate corporate employment opportunities.")
            st.markdown('</div>', unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.write("**Model Confidence Breakdown**")

            st.write(f"Legitimate Confidence: **{legit_prob * 100:.2f}%**")
            st.progress(min(legit_prob, 1.0))

            st.write(f"Fraud Probability: **{fake_prob * 100:.2f}%**")
            st.progress(min(fake_prob, 1.0))
            st.markdown('</div>', unsafe_allow_html=True)

        # Token Sequence Preview
        with st.expander("🛠️ Inspect Preprocessed Sequence & Raw Features"):
            st.write("**Concatenated Feature String:**")
            st.code(combined_text)
            st.write(f"**Padded Sequence (Length {SEQ_LEN}):**")
            st.code(str(padded.tolist()))
