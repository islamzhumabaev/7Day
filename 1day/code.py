from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.joblib")

class ModelInput(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(input: ModelInput):
    X = np.array(input.features).reshape(1, -1)
    prediction = model.predict(X)
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("code:app", host="127.0.0.1", port=8000, reload=True)
