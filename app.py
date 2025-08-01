import streamlit as st
import torch
import shap
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, TextClassificationPipeline
import streamlit.components.v1 as components

CLASS_NAMES = [
    "Company", "Educational Institution", "Artist", "Athlete", "Office Holder",
    "Mean of Transportation", "Building", "Natural Place", "Village", "Animal",
    "Plant", "Album", "Film", "Written Work"
]

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = BertForSequenceClassification.from_pretrained("Sandhya385/dbpedia-bert-model")
    tokenizer = BertTokenizer.from_pretrained("Sandhya385/dbpedia-bert-model")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# For normal predictions
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

# Custom wrapper just for SHAP
def predict_proba(texts):
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**encodings).logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.numpy()

# UI
st.set_page_config(page_title="DBPedia BERT Classifier", layout="wide")
st.title("📚 DBPedia Text Classifier")

user_input = st.text_area("Enter the text for classification", "Apple is looking at buying U.K. startup for $1 billion")

if st.button("Classify") and user_input.strip():
    results = pipe(user_input)[0]
    probs = np.array([r['score'] for r in results])
    pred_index = int(np.argmax(probs))
    confidence = probs[pred_index]

    st.subheader("🔍 Prediction")
    st.write(f"**Predicted Class:** {CLASS_NAMES[pred_index]}")
    st.write(f"**Confidence:** {confidence:.4f}")
    st.bar_chart(dict(zip(CLASS_NAMES, probs)))

    # SHAP Explanation (finally fixed)
    st.subheader("💡 Token Importance (SHAP)")
    with st.spinner("Generating SHAP Explanation..."):
        masker = shap.maskers.Text(tokenizer)
        explainer = shap.Explainer(pipe, masker)

        shap_values = explainer([user_input])

        # Get predicted index
        predicted_index = np.argmax([r['score'] for r in pipe(user_input)[0]])

        # Generate HTML from SHAP
        shap_html = shap.plots.text(shap_values[:, :, predicted_index], display=False)

        # Render HTML in Streamlit
        components.html(shap_html, height=400, scrolling=True)
    
    #Explanation below the plot
    st.markdown("""
    **ℹ️ Plot Annotations:**
    - **base value** (left): baseline model output with no input.
    - **f(label)** (right): final prediction score for the chosen class.
    - Words with **positive SHAP values** increase prediction score (🔴), and those with **negative** values decrease it (🔵).
    - Hover over any word to see its SHAP value.

    ⚠️ Only the first 128 tokens are used (longer texts are truncated).
       """)

