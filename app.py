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

st.markdown(
    """
    <style>
    /* Style for the button */
    div.stButton > button {
        font-size: 20px !important; 
        color: white !important; 
        background-color: green !important; 
        border: none; 
        padding: 10px 20px; 
        border-radius: 10px; 
        cursor: pointer; 
    }

    /* Style for the text input label */
    div.stTextInput > label {
        font-size: 24px ;
        color: black ;
        font-family: Arial, sans-serif;
    }

    /* Style for the text input field */
    div.stTextInput > div > input {
        font-size: 24px ;
        color: black;
        border: 2px solid black;
        border-radius: 15px ;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
user_input = st.text_area(r"$\textsf{\large Enter a SMILES string}$")
button =st.button(r"$\textsf{\normalsize Predict Toxicity}$")

d = {1:'Toxic', 0:'Not Toxic'}

if user_input and button:
    user_input_encoded = tokenizer([user_input], padding=True, truncation=True, max_length=300, return_tensors='pt')
    output = model(**user_input_encoded) 
    st.markdown('##### ' + 'Logits:' + ' ' + str(output.logits)) 
    probabilities = torch.softmax(output.logits, dim=1) 
    y_pred = np.argmax(probabilities.detach().numpy(), axis=1) 
    st.markdown('##### ' + 'Prediction:' + ' ' + d[y_pred[0]]) 