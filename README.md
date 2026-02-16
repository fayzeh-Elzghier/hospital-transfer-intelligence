# Hospital Transfer Intelligence (Prototype)

AI-powered prototype for a clinical transfer decision support system.  
The system analyzes synthetic medical reports using lightweight NLP to infer case severity and required specialty, then ranks hospitals based on capacity, load, distance, and specialty match.

## Why this project
During high-pressure periods (crises, peak load), transfer decisions may be delayed or inconsistent.  
This prototype demonstrates a feasibility workflow for supporting faster, more consistent transfer recommendations **without using real patient data**.

## What it does
- Generates **synthetic** hospitals data (specialties, beds/ICU, load, distance)
- Generates **synthetic** transfer cases (report text + labels)
- Trains a simple NLP model (TF-IDF + Logistic Regression) to predict:
  - Required specialty
  - Severity level
- Ranks hospitals using an explainable scoring method:
  - Specialty match (required)
  - Capacity (beds/ICU)
  - Distance
  - Current load
- Provides a Streamlit UI for demo and explanation

## Repository files
- `app.py` Streamlit demo app
- `generator.py` synthetic data generator
- `analyzer.py` NLP report analyzer (training + prediction)
- `decision.py` hospital ranking logic
- `requirements.txt` dependencies
- `docs/` GitHub Pages website (static)

## How to run locally
```bash
python -m venv .venv
source .venv/bin/activate   # mac/linux
# .venv\Scripts\activate    # windows
pip install -r requirements.txt
streamlit run app.py
