import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

#gi@st.cache_data.clear()
def get_model():
    model_name = "DeepChem/ChemBERTa-77M-MTR"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained('ramyanjn/ChemBERTaFTTox')
    return tokenizer,model

tokenizer, model = get_model()

st.title("Toxicity Predictor")
user_input = st.text_area('Enter SMILES')
button =st.button('Predict Toxicity')

d = {1:'Toxic', 0:'Not Toxic'}

if user_input and button:
    user_input_encoded = tokenizer([user_input], padding=True, truncation=True, max_length=300, return_tensors='pt')
    output = model(**user_input_encoded)
    st.write('Logits:', output.logits)
    probabilities = torch.softmax(output.logits, dim=1)
    y_pred = np.argmax(probabilities.detach().numpy(), axis=1)
    st.write('Prediction:', d[y_pred[0]])