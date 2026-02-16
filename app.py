import streamlit as st
import pandas as pd

from generator import generate_hospitals, generate_transfers
from analyzer import ReportAnalyzer
from decision import rank_hospitals

st.set_page_config(page_title="Medical Transfer DSS Prototype", layout="wide")

st.title("Medical Transfer Decision Support Prototype")
st.caption("Synthetic demo: report analysis + hospital ranking (no real patient data).")

# Sidebar controls
st.sidebar.header("Demo Controls")
seed_h = st.sidebar.number_input("Hospitals seed", min_value=1, max_value=9999, value=7)
seed_t = st.sidebar.number_input("Transfers seed", min_value=1, max_value=9999, value=13)
n_h = st.sidebar.slider("Number of hospitals", 5, 15, 8)
n_t = st.sidebar.slider("Synthetic training cases", 50, 400, 120)

# Generate synthetic data
hospitals = generate_hospitals(n=n_h, seed=int(seed_h))
transfers = generate_transfers(n=n_t, seed=int(seed_t))

# Fit analyzer
analyzer = ReportAnalyzer()
analyzer.fit(transfers)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Hospitals (Synthetic)")
    st.dataframe(hospitals, use_container_width=True)

with col2:
    st.subheader("Sample Training Transfers (Synthetic)")
    st.dataframe(transfers.sample(min(8, len(transfers))), use_container_width=True)

st.divider()
st.subheader("Try a new transfer request")

default_report = "Patient has chest pain and shortness of breath. Suspected myocardial infarction. Needs Cardiology."
report = st.text_area("Medical report text", value=default_report, height=120)

if st.button("Analyze + Recommend", type="primary"):
    required_spec, severity, spec_probs, sev_probs = analyzer.predict(report)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Predicted specialty", required_spec)
    with c2:
        st.metric("Predicted severity", severity)
    with c3:
        st.write("Explainability (top probabilities)")

    probs_spec_sorted = sorted(spec_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    probs_sev_sorted = sorted(sev_probs.items(), key=lambda x: x[1], reverse=True)

    st.write("Specialty probabilities:")
    st.table(pd.DataFrame(probs_spec_sorted, columns=["specialty", "probability"]))

    st.write("Severity probabilities:")
    st.table(pd.DataFrame(probs_sev_sorted, columns=["severity", "probability"]))

    # --- Hackathon-style outputs ---
    st.divider()
    st.subheader("Decision Output (Hackathon Style)")

    ranked = rank_hospitals(hospitals, required_spec, severity, top_k=5)
    best = ranked.iloc[0]

    st.success(f"Recommended Hospital: {best['hospital']}")

    # KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Decision Score", f"{float(best['score']):.4f}")
    with k2:
        st.metric("Distance (km)", f"{float(best['distance_km']):.1f}")
    with k3:
        st.metric("Beds Free", f"{int(best['beds_free'])}/{int(best['beds_total'])}")
    with k4:
        st.metric("ICU Free", f"{int(best['icu_free'])}/{int(best['icu_total'])}")
    with k5:
        st.metric("Current Load", f"{float(best['load']):.2f}")

    st.write("Why this hospital")
    reasons = str(best["reason"]).split(",")
    for r in reasons:
        st.write(f"- {r.strip()}")

    st.divider()
    st.subheader("Ranked Hospitals (Top 5)")
    st.dataframe(
        ranked[["hospital","specialties","beds_free","icu_free","load","distance_km","score","reason"]],
        use_container_width=True
    )

    st.subheader("Score Comparison")
    st.bar_chart(ranked.set_index("hospital")["score"])

    st.subheader("Hospitals Map (Synthetic)")
    map_df = hospitals.rename(columns={"lat": "latitude", "lon": "longitude"})
    st.map(map_df[["latitude", "longitude"]])
