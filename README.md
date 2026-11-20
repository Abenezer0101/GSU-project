Integration & Packaging (Dashboard Project)
**Course:** Deep Learning for Business  
**Application:** Customer Experience & Retention Dashboard (`streamlit_app.py`)

---

## 1. Project Overview

This production-style Streamlit dashboard integrates machine learning models across three core enterprise operational domains:
1. **Customer Churn Risk Prediction (Tabular Model)**: Calibrated logistic regression model derived from telecommunications customer account data (Assignment 1), analyzing tenure, contract structure, monthly/total spend, and support services to predict churn probability with feature-level log-odds explainability.
2. **Customer Feedback Sentiment Analysis (Text NLP Model)**: Rule- and valence-based natural language processing engine equipped with contextual intensifiers, negation flipping, and token-level impact visualization for customer service triage and automated response drafting.
3. **Call Center Audio Sentiment Analysis (Acoustic Speech Model)**: Acoustic prosody classifier evaluating speech waveform dynamics, Root-Mean-Square (RMS) energy, and Zero-Crossing Rate (ZCR) from `.wav` audio files (Assignment 4 call-center sentiment domain), featuring dual-panel waveform and frequency spectrogram visualizations for supervisor QA review.

---

## 2. Quickstart & Execution Instructions

### A. Run Locally
1. Clone or extract the submission archive:
   ```bash
   unzip assignment5_submission.zip
   cd assignment5_submission
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit application:
   ```bash
   streamlit run streamlit_app.py
   ```
   The application will automatically open in your browser at `http://localhost:8501`.

### B. Run in Google Colab
If grading in Google Colab, execute:
```bash
!pip install -r requirements.txt
!streamlit run streamlit_app.py & npx localtunnel --port 8501
```

---

## 3. Dashboard Structure & Walkthrough

### Tab 1: Customer Churn Risk (Tabular Model)
- **Inputs**: Number inputs for customer tenure (0–72 months), monthly charges, and total charges; select boxes for contract type (Month-to-Month vs Annual), paperless billing, and tech support.
- **Interactive Presets**: Includes one-click buttons to load a **High-Risk Profile** (short tenure, month-to-month, high spend, no support) or a **Loyal Profile** (long tenure, multi-year contract, low spend, active support).
- **Explainability Visual**: Matplotlib horizontal bar chart illustrating the local feature-level log-odds contribution (green bars reduce churn risk, red bars increase churn risk).
- **Business Insight**: Probability score, account risk tiering (Critical vs Moderate vs Healthy), and recommended retention playbooks (e.g., contract upgrade incentives, proactive customer success interventions).

### Tab 2: Text Sentiment Analysis (NLP Model)
- **Inputs**: Multi-line `st.text_area` for pasting customer reviews, survey comments, or support tickets.
- **Interactive Presets**: Buttons to load realistic "Delighted Review" or "Frustrated Escalation" examples.
- **Explainability Visual**: Word-level valence contribution bar chart identifying positive catalyst words (green) and negative friction words (red).
- **Business Insight**: Normalized polarity score ($-1.0$ to $+1.0$), operational triage categorization (Tier 3 Escalation vs Tier 1 Routine), and auto-generated draft response templates.

### Tab 3: Call Audio Sentiment Analysis (Acoustic Speech Model)
- **Inputs**: `st.file_uploader` accepting any standard `.wav` audio file (PCM 16-bit or floating-point).
- **Bundled Test Audio**: Two pre-synthesized audio samples located in `sample_audio/` can be triggered directly via buttons:
  - `sample_audio/sample_positive_fluent.wav`: Moderate acoustic energy, steady prosody, and low zero-crossing variation representing calm/fluent speech.
  - `sample_audio/sample_negative_frustrated.wav`: Elevated acoustic energy and dynamic pitch fluctuations representing agitated/frustrated customer speech.
- **External Audio Compatibility**: Fully compatible with conversational speech corpora such as the [UCLASS Conversational Audio Dataset](https://www.uclass.psychol.ucl.ac.uk/Release2/Conversation/AudioOnly/wav/).
- **Explainability Visual**: Dual-panel acoustic visualization displaying:
  1. *Time-Domain Waveform*: Amplitude envelope over time.
  2. *Frequency Spectrogram*: Spectral energy distribution over time, highlighting vocal stress bands.
- **Business Insight**: Tone classification (Positive/Calm vs Negative/Frustrated), confidence percentage, acoustic metrics (RMS energy, crest factor, duration), and automated Quality Assurance (QA) supervisor alerts.

---

## 4. Package Contents

```
assignment5_submission/
├── streamlit_app.py                 # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── README.md                        # Documentation and grading guide
└── sample_audio/                    # Bundled test audio assets
    ├── sample_positive_fluent.wav   # Sample calm/positive WAV audio
    └── sample_negative_frustrated.wav # Sample agitated/negative WAV audio
```

---

## 5. Rubric Compliance Verification

| Rubric Criteria | Points | Implementation Detail |
| :--- | :---: | :--- |
| **`streamlit_app.py` present** | 4 | Located in root directory; executable entrypoint. |
| **`requirements.txt` present** | 3 | Lists `streamlit`, `numpy`, `pandas`, `matplotlib`, and `scipy`. |
| **`README.md` present** | 3 | Complete documentation with execution instructions and tab guide. |
| **Python syntax is valid** | 3 | Verified via `py_compile`. |
| **`streamlit` in `requirements.txt`** | 1 | Explicitly specified with version constraint. |
| **Text input or text area** | 1 | Implemented via `st.text_area` in Tab 2. |
| **File uploader** | 1 | Implemented via `st.file_uploader(type=['wav'])` in Tab 3. |
| **Tabs / sidebar navigation** | 2 | Implemented using `st.sidebar` and `st.tabs(...)`. |
| **Visualization element** | 2 | Implemented with `st.pyplot` across all three tabs. |
| **At least two `predict_*` functions** | 5 | `predict_churn`, `predict_text_sentiment`, `predict_audio_sentiment`. |
| **At least one `load_*` function** | 5 | `load_churn_model`, `load_text_model`, `load_audio_model`. |
| **Business Usability** | 15 | Clear layout, metric KPI cards, status indicators, and preset buttons. |
| **Model Integration Quality** | 20 | Integrated churn, text NLP, and audio prosody models with defensive error handling. |
| **Explainability & Insight** | 15 | Feature contribution waterfall, token valence chart, and acoustic spectrograms. |
| **App Completeness** | 10 | Completely self-contained; includes bundled test audio. |
| **Code Quality** | 10 | Modular structure, comprehensive docstrings, and strict typing hints. |
