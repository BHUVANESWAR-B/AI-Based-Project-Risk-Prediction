import streamlit as st
import requests
import numpy as np
import pandas as pd

API_BASE = "http://127.0.0.1:8001"   # change if your API uses another port

st.set_page_config(page_title="AI Project Risk Predictor", layout="wide", page_icon="🚨")

st.title("🚨 AI Project Risk Predictor")
st.caption("Keep the FastAPI server running:  python notebooks/06_deployment_api.py")

# ---------- Helpers ----------
def get_api_features():
    r = requests.get(f"{API_BASE}/", timeout=5)
    r.raise_for_status()
    return r.json()

def safe_post_predict(features_list):
    r = requests.post(
        f"{API_BASE}/predict",
        json={"features": features_list},
        timeout=10
    )
    # If API returns an error HTML/text, don't call r.json() blindly
    if r.status_code != 200:
        return None, f"API Error {r.status_code}: {r.text}"
    try:
        return r.json(), None
    except ValueError:
        return None, f"API returned non‑JSON response: {r.text}"

def risk_level_from_score(score: float):
    if score > 0.7:
        return "HIGH"
    if score > 0.4:
        return "MEDIUM"
    return "LOW"

def reasons_and_actions(feature_names, feature_values):
    """
    Simple rule-based explanations based on inputs.
    (This is not SHAP, but gives clear, human-friendly reasons + actions.)
    """
    v = dict(zip(feature_names, feature_values))

    reasons = []
    actions = []

    # Common drivers (tune thresholds as you like)
    if v.get("Complexity_Score", 0) >= 3.5:
        reasons.append("High complexity increases integration and testing risk.")
        actions.append("Reduce complexity: split into smaller modules, freeze core requirements early.")

    if v.get("Change_Request_Frequency", 0) >= 0.6:
        reasons.append("Frequent change requests cause scope creep and rework.")
        actions.append("Introduce change control: approval workflow, limit changes per sprint, prioritize backlog.")

    if v.get("Team_Turnover_Rate", 0) >= 0.3:
        reasons.append("High team turnover disrupts continuity and slows delivery.")
        actions.append("Stabilize team: knowledge transfer, documentation, pair programming, retention plan.")

    if v.get("Vendor_Reliability_Score", 1) <= 0.5:
        reasons.append("Low vendor reliability increases dependency delays.")
        actions.append("Add backup vendor/plan, set SLAs, track vendor tasks weekly.")

    if v.get("External_Dependencies_Count", 0) >= 3:
        reasons.append("Many external dependencies increase the chance of delays.")
        actions.append("Reduce dependencies or add buffers; track dependencies with owners and deadlines.")

    if v.get("Communication_Frequency", 10) <= 2:
        reasons.append("Low communication frequency can hide issues until late.")
        actions.append("Daily/weekly syncs, clear reporting, single source of truth (Jira/Sheets).")

    if v.get("Schedule_Pressure", 0) >= 0.7:
        reasons.append("High schedule pressure leads to shortcuts and quality issues.")
        actions.append("Replan with realistic milestones; add buffer for testing and review.")

    if v.get("Budget_Utilization_Rate", 0) >= 0.85:
        reasons.append("Budget already heavily utilized increases overrun risk.")
        actions.append("Add contingency budget; cut non-essential scope; renegotiate resources.")

    if v.get("Resource_Availability", 1) <= 0.5:
        reasons.append("Low resource availability slows progress.")
        actions.append("Add resources, reduce scope, or extend timeline; remove blockers quickly.")

    if v.get("Technical_Debt_Level", 0) >= 0.7:
        reasons.append("High technical debt increases bugs and rework.")
        actions.append("Allocate refactoring time; enforce code reviews; improve test coverage.")

    # If we didn’t catch anything, still show generic guidance
    if not reasons:
        reasons.append("Risk is influenced by combined patterns across multiple features.")
        actions.append("Track risks weekly, review scope, monitor budget burn rate, and update plan continuously.")

    return reasons[:5], actions[:6]  # keep concise


# ---------- Check API ----------
st.sidebar.subheader("🔌 API Status")
try:
    api_info = get_api_features()
    feature_names = api_info["features"]
    n_features = api_info["n_features"]
    st.sidebar.success(f"API Live ✅  (features: {n_features})")
except Exception as e:
    st.sidebar.error("API not reachable ❌")
    st.error(
        "Your FastAPI server is not reachable. Start it in a separate terminal:\n\n"
        "`python notebooks/06_deployment_api.py`\n\n"
        f"Details: {e}"
    )
    st.stop()

# ---------- Inputs UI ----------
st.sidebar.subheader("🧾 Project Inputs")

# Choose good defaults + ranges for a clean demo
# For any feature where you don’t know real scale, keep 0–1 sliders.
inputs = {}

# Numeric fields / sliders (adjust as per your dataset meaning)
inputs["Team_Size"] = st.sidebar.number_input("Team_Size", min_value=1, max_value=100, value=5)
inputs["Project_Budget_USD"] = st.sidebar.number_input("Project_Budget_USD", min_value=1000.0, max_value=100000000.0, value=250000.0, step=1000.0)
inputs["Estimated_Timeline_Months"] = st.sidebar.number_input("Estimated_Timeline_Months", min_value=1, max_value=60, value=6)
inputs["Complexity_Score"] = st.sidebar.slider("Complexity_Score", 0.0, 5.0, 2.5)
inputs["Stakeholder_Count"] = st.sidebar.number_input("Stakeholder_Count", min_value=1, max_value=50, value=3)
inputs["Past_Similar_Projects"] = st.sidebar.number_input("Past_Similar_Projects", min_value=0, max_value=50, value=2)
inputs["External_Dependencies_Count"] = st.sidebar.number_input("External_Dependencies_Count", min_value=0, max_value=20, value=1)
inputs["Change_Request_Frequency"] = st.sidebar.slider("Change_Request_Frequency (0-1)", 0.0, 1.0, 0.3)
inputs["Team_Turnover_Rate"] = st.sidebar.slider("Team_Turnover_Rate (0-1)", 0.0, 1.0, 0.1)
inputs["Vendor_Reliability_Score"] = st.sidebar.slider("Vendor_Reliability_Score (0-1)", 0.0, 1.0, 0.8)
inputs["Historical_Risk_Incidents"] = st.sidebar.number_input("Historical_Risk_Incidents", min_value=0, max_value=50, value=1)
inputs["Communication_Frequency"] = st.sidebar.number_input("Communication_Frequency (per week)", min_value=0, max_value=14, value=3)
inputs["Geographical_Distribution"] = st.sidebar.slider("Geographical_Distribution (0=co-located, 1=distributed)", 0.0, 1.0, 0.2)
inputs["Schedule_Pressure"] = st.sidebar.slider("Schedule_Pressure (0-1)", 0.0, 1.0, 0.4)
inputs["Budget_Utilization_Rate"] = st.sidebar.slider("Budget_Utilization_Rate (0-1)", 0.0, 1.0, 0.5)
inputs["Market_Volatility"] = st.sidebar.slider("Market_Volatility (0-1)", 0.0, 1.0, 0.3)
inputs["Integration_Complexity"] = st.sidebar.slider("Integration_Complexity (0-1)", 0.0, 1.0, 0.4)
inputs["Resource_Availability"] = st.sidebar.slider("Resource_Availability (0-1)", 0.0, 1.0, 0.8)
inputs["Organizational_Change_Frequency"] = st.sidebar.slider("Organizational_Change_Frequency (0-1)", 0.0, 1.0, 0.2)
inputs["Cross_Functional_Dependencies"] = st.sidebar.slider("Cross_Functional_Dependencies (0-1)", 0.0, 1.0, 0.3)
inputs["Previous_Delivery_Success_Rate"] = st.sidebar.slider("Previous_Delivery_Success_Rate (0-1)", 0.0, 1.0, 0.7)
inputs["Technical_Debt_Level"] = st.sidebar.slider("Technical_Debt_Level (0-1)", 0.0, 1.0, 0.3)
inputs["Project_Start_Month"] = st.sidebar.number_input("Project_Start_Month (1-12)", min_value=1, max_value=12, value=2)
inputs["Current_Phase_Duration_Months"] = st.sidebar.number_input("Current_Phase_Duration_Months", min_value=0, max_value=24, value=1)

# Build feature vector in correct order required by API/model
features_vector = [float(inputs[name]) for name in feature_names]

# ---------- Main action ----------
left, right = st.columns([1, 1])

with left:
    st.subheader("📌 Current Input Summary")
    st.dataframe(pd.DataFrame({"Feature": feature_names, "Value": features_vector}), height=420, use_container_width=True)

with right:
    st.subheader("🔮 Prediction")

    if st.button("Predict Risk", type="primary", use_container_width=True):
        result, err = safe_post_predict(features_vector)

        if err:
            st.error(err)
            st.info("Tip: Keep FastAPI running in a separate terminal on port 8001.")
            st.stop()

        risk_score = float(result.get("risk_score", 0.0))
        # if API returns risk_level use it; else compute
        risk_level = result.get("risk_level") or risk_level_from_score(risk_score)

        col1, col2 = st.columns(2)
        col1.metric("Risk Score", f"{risk_score:.3f}")
        col2.metric("Risk Level", risk_level)

        st.markdown("---")
        st.subheader("🤔 Why is it risky?")
        reasons, actions = reasons_and_actions(feature_names, features_vector)

        for r in reasons:
            st.write(f"- {r}")

        st.subheader("✅ How to decrease the risk")
        for a in actions:
            st.write(f"- {a}")

        st.markdown("---")
        st.caption("Note: Explanations are rule-based for clarity. You can later replace with SHAP-based explanations.")

