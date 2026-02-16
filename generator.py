import numpy as np
import pandas as pd
import random

SPECIALTIES = ["Emergency", "Surgery", "Cardiology", "Neurology", "Orthopedics", "ICU", "Pediatrics"]
SEVERITY_LEVELS = ["stable", "urgent", "critical"]

TEMPLATES = [
    "Patient has {symptom1} and {symptom2}. Suspected {cond}. Needs {spec}.",
    "Report indicates {cond} with {symptom1}. Risk is {risk}. Suggested {spec}.",
    "Severe {symptom1}, {symptom2}. Possible {cond}. Transfer to {spec}."
]

SYMPTOMS = ["chest pain", "shortness of breath", "bleeding", "high fever", "head trauma", "seizure", "low BP", "vomiting"]
CONDITIONS = [
    ("myocardial infarction", "Cardiology", "critical"),
    ("stroke", "Neurology", "critical"),
    ("internal bleeding", "Surgery", "critical"),
    ("fracture", "Orthopedics", "urgent"),
    ("sepsis", "ICU", "critical"),
    ("appendicitis", "Surgery", "urgent"),
    ("asthma attack", "Emergency", "urgent"),
    ("pediatric dehydration", "Pediatrics", "urgent"),
]
RISK_WORDS = ["low", "moderate", "high"]

def generate_hospitals(n=8, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        name = f"Hospital_{i+1}"
        # each hospital has 2-4 specialties
        specs = rng.choice(SPECIALTIES, size=int(rng.integers(2,5)), replace=False).tolist()
        beds_total = int(rng.integers(40, 220))
        beds_free = int(rng.integers(0, max(1, beds_total//4)))
        icu_total = int(rng.integers(4, 25))
        icu_free = int(rng.integers(0, max(1, icu_total//3)))
        load = float(np.clip(1 - (beds_free / max(1, beds_total)), 0, 1))  # 0 low load, 1 high load
        distance_km = float(rng.integers(2, 60))
        rows.append({
            "hospital": name,
            "specialties": ", ".join(specs),
            "beds_total": beds_total,
            "beds_free": beds_free,
            "icu_total": icu_total,
            "icu_free": icu_free,
            "load": round(load, 3),
            "distance_km": distance_km
        })
    return pd.DataFrame(rows)

def generate_transfers(n=120, seed=13):
    random.seed(seed)
    rows = []
    for i in range(n):
        symptom1, symptom2 = random.sample(SYMPTOMS, 2)
        cond, spec, sev = random.choice(CONDITIONS)
        risk = random.choice(RISK_WORDS)
        template = random.choice(TEMPLATES)
        report = template.format(symptom1=symptom1, symptom2=symptom2, cond=cond, risk=risk, spec=spec)
        rows.append({
            "case_id": f"C{i+1:04d}",
            "report_text": report,
            "true_specialty": spec,
            "true_severity": sev
        })
    return pd.DataFrame(rows)
