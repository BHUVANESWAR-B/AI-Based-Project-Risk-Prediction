from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

app = FastAPI(title="Project Risk Predictor API")

# Load saved artifacts from notebook 05
model = joblib.load("models/production_risk_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

print("✅ Loaded:", len(feature_names), "features")

class RiskInput(BaseModel):
    features: list[float]

@app.post("/predict")
async def predict_risk(input_data: RiskInput):
    X_input = np.array(input_data.features).reshape(1, -1)
    X_scaled = scaler.transform(X_input)
    
    risk_score = model.predict(X_scaled)[0]
    
    return {
        "risk_score": float(risk_score),
        "risk_level": "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.4 else "LOW",
        "features_used": feature_names[:len(input_data.features)]
    }

@app.get("/")
async def home():
    return {"message": "API ready", "n_features": len(feature_names), "features": feature_names}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
