# 🛡️ JobGuard AI — Fake Job Posting Classifier

An interactive **Streamlit** application powered by a **TensorFlow / Keras Recurrent Neural Network (RNN)** that analyzes natural language patterns across job listings to detect fraudulent recruitment posts in real time.

---

## ✨ Key Features

- **Multi-Field Job Listing Analysis:** Evaluates structured and unstructured job post fields—including *Company Profile*, *Salary Range*, *Requirements*, *Benefits*, *Location*, *Job Description*, and *Industry*.
- **High-Recall Fraud Detection:** Powered by a trained Keras RNN model achieving **99.55% test accuracy**, **100.0% fake recall (TPR)**, and a **0.47% false positive rate** across a 10,000-word vocabulary.
- **Instant Demo Presets:** Test model predictions immediately using sidebar radio presets (*Blank Form*, *Typical Legitimate Job*, and *Suspicious/Fraudulent Job*).
- **Confidence Breakdown & Transparency:** Displays clear status badges (`VERIFIED AS LEGITIMATE` vs. `FLAGGED AS FRAUDULENT`), dual confidence progress bars, and an expandable inspector showing the exact concatenated feature string and padded token sequence.

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/) (`>=1.30.0`)
- **Deep Learning:** [TensorFlow](https://www.tensorflow.org/) / Keras (`tensorflow-cpu>=2.15.0`)
- **Numerical Processing:** [NumPy](https://numpy.org/) (`>=1.24.0`)
- **Serialization:** Python `pickle` & Keras native format (`.keras`)

---

## 📁 Project Structure

```text
├── app.py                   # Main Streamlit application
├── fake_job_model.keras     # Trained Keras RNN classification model
├── tokenizer.pkl            # Fitted Keras Tokenizer (10,000-word vocabulary)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
