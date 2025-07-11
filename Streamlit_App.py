import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_REPO = "GeorgiosKoutroumanos/NLP-Roberta-HP-detection"

@st.cache_resource(show_spinner=False)
def load_model():
    st.write("Loading model from Hugging Face Hub...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
    model.eval()
    st.write("Model loaded successfully!")
    return tokenizer, model

def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    predicted_class_id = torch.argmax(probs).item()
    confidence = probs[0, predicted_class_id].item()
    return predicted_class_id, confidence

def main():
    st.title("Hate Speech Detector 🔍")

    tokenizer, model = load_model()

    user_input = st.text_area("Enter text to classify", height=150)

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some text to classify.")
        else:
            pred_class, conf = predict(user_input, tokenizer, model)
            st.write(f"**Predicted class:** {pred_class}")
            st.write(f"**Confidence:** {conf:.2f}")

if __name__ == "__main__":
    main()