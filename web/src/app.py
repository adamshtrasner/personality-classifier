import gradio as gr
import pickle
import numpy as np
import os

# Load model and scaler
MODEL_PATH = os.path.join("web", "src", "model.pkl")
SCALER_PATH = os.path.join("web", "src", "scaler.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Prediction function
def predict_personality(
    time_alone: int,
    stage_fear: str,
    social_event_attendance: int,
    going_outside: int,
    drained_after_socializing: str,
    friends_circle_size: int,
    post_frequency: int
) -> str:
    # Encode binary features
    sf = 1 if stage_fear == "Yes" else 0
    dr = 1 if drained_after_socializing == "Yes" else 0

    # Build feature array and scale
    features = np.array([[time_alone, sf, social_event_attendance,
                          going_outside, dr, friends_circle_size,
                          post_frequency]])
    features_scaled = scaler.transform(features)

    # Predict
    proba = model.predict_proba(features_scaled)[0]
    pred = model.classes_[np.argmax(proba)]
    label = "Extrovert" if pred == 0 else "Introvert"
    confidence = round(np.max(proba) * 100, 2)

    return f"{label} ({confidence}%)"

# Build the Gradio interface
iface = gr.Interface(
    fn=predict_personality,
    inputs=[
        gr.Slider(0, 11, step=1, label="Time spent alone (0–11 hours)"),
        gr.Radio(["Yes", "No"], label="Stage fear"),
        gr.Slider(0, 10, step=1, label="Social event attendance (0–10)"),
        gr.Slider(0, 7, step=1, label="Going outside (0–7)"),
        gr.Radio(["Yes", "No"], label="Drained after socializing"),
        gr.Slider(0, 15, step=1, label="Friends circle size (0–15)"),
        gr.Slider(0, 10, step=1, label="Post frequency (0–10)")
    ],
    outputs="text",
    title="🧠 Personality Classifier",
    description="Answer the questions below to predict whether someone is an Introvert or Extrovert."
)

if __name__ == "__main__":
    iface.launch()