import pandas as pd

def specialty_match(hospital_specs: str, required_spec: str) -> float:
    specs = [s.strip() for s in hospital_specs.split(",")]
    return 1.0 if required_spec in specs else 0.0

def compute_score(row, required_spec: str, severity: str):
    # weights tuned for demo (explainable)
    w_spec = 0.45
    w_capacity = 0.25
    w_distance = 0.15
    w_load = 0.15

    # specialty match is hard gate; if no match, big penalty
    spec = specialty_match(row["specialties"], required_spec)
    if spec < 0.5:
        return -1e9, "No specialty match"

    # capacity signals: if critical, prioritize ICU
    if severity == "critical":
        cap = min(1.0, row["icu_free"] / max(1, row["icu_total"]))
        cap_reason = f"ICU free {row['icu_free']}/{row['icu_total']}"
    else:
        cap = min(1.0, row["beds_free"] / max(1, row["beds_total"]))
        cap_reason = f"Beds free {row['beds_free']}/{row['beds_total']}"

    # distance normalized (smaller is better)
    dist = float(row["distance_km"])
    dist_score = 1.0 / (1.0 + dist / 10.0)

    # load: smaller is better
    load_score = 1.0 - float(row["load"])

    score = (w_spec * spec) + (w_capacity * cap) + (w_distance * dist_score) + (w_load * load_score)

    reason = f"Match={required_spec}, {cap_reason}, Distance={dist}km, Load={row['load']}"
    return score, reason

def rank_hospitals(hospitals_df: pd.DataFrame, required_spec: str, severity: str, top_k=3):
    rows = []
    for _, r in hospitals_df.iterrows():
        score, reason = compute_score(r, required_spec, severity)
        rows.append({**r.to_dict(), "score": score, "reason": reason})
    out = pd.DataFrame(rows)
    out = out.sort_values("score", ascending=False).head(top_k)
    return out
