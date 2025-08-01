[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bert-multi-class-text-classifier-dwnpdkqnzu7vnihl8wbdwt.streamlit.app/)

# 🧠 BERT Multi-Class Text Classifier (DBpedia 14)

This project implements an end-to-end **multi-class text classification system** using a fine-tuned transformer model (BERT) to categorize short entity descriptions into one of **14 predefined classes** from the DBpedia ontology.

---

## 🎯 Objective

To develop a **high-accuracy**, **interpretable**, and **scalable** text classification system for use in **Knowledge Graphs** and other NLP pipelines.

---

## 📦 Dataset

- **Source**: [DBpedia 14](https://huggingface.co/datasets/dbpedia_14) via Hugging Face  
- **Original size**: 560,000 training samples, 70,000 test samples  
- **Used size**: Stratified subset of 20,000 training samples due to hardware constraints  
- **Features**: `label`, `title`, `content` (short Wikipedia abstracts)

---

## 📁 Classes

`Album, Animal, Artist, Athlete, Building, Company, Educational Institution, Film, Means of Transportation, Natural Place, Office Holder, Plant, Village, 
Written  Work`

---

## 🛠️ Preprocessing

- Removed duplicates  
- Mapped class labels to integers (0 to 13)  
- Analyzed text length for `max_length` truncation  
- Applied Hugging Face tokenizer with `padding=max_length`, `truncation=True`  
- Balanced training set using stratified sampling  

---

## 🧠 Model & Training

- **Model**: `bert-base-uncased` via `AutoModelForSequenceClassification`  
- **Num labels**: 14  
- **Training**: via Hugging Face `Trainer` API  
- **Custom metrics**: Precision, Recall, F1  
- **TrainingArguments**: Configurable batch size, weight decay, evaluation strategy, logging frequency

✅ Saved model and tokenizer for reuse in app deployment

---

## 📊 Evaluation

- Predictions on test set
- Generated:
  - **Confusion Matrix**
  - **Classification Report (per-class precision, recall, F1)**
- Interpretability:
  - `visualize_attention()` → Shows attention heatmap for single input
  - **SHAP** → Shows token contribution via `shap.plots.text`

---

## 🌐 Web App (Streamlit)

A Streamlit web interface is provided with the following features:

- Input text box for short entity description  
- "Classify" button → shows:
  - Predicted **class name + probability**
  - **Bar chart** of all class probabilities
  - **SHAP explanation plot** (text visualization of word importance)

---

## 📁 Directory Structure

├── app/ # Streamlit app code
├── sampletexts.txt # Sample test cases
├── requirements.txt # Dependencies
├── Text_classifier.ipynb # Full training notebook
└── README.md

📁 The trained model files (`dbpedia_bert_model_20k_cpu/`) are excluded from GitHub due to file size limits.  
If needed, they can be shared upon request or hosted via Google Drive.

