import streamlit as st

import json

import tensorflow as tf
from keras.models import load_model
from PIL import Image

def load_model_():
    model = load_model("./models/document_classification_model")
    with open('assets/doc_classes.json', 'r') as f:
        classes = json.load(f)
    return model, classes
with st.spinner('Resources are being loaded...'):
  model, classes=load_model_()

def preprocess(image):
    """
        Accepts path for the image to be classified.
        
        Returns X.
    """
    try:
        im = image.resize((224,224)).convert("RGB")
    except:
        return "The format of the image isn't suitable."
    X = tf.keras.utils.img_to_array(im)
    return tf.reshape(X,[X.shape[0]//224,224,224,3])
 
st.write("""
         # Documents Classifier
         """)

file = st.file_uploader("Upload the image to be classified", type=["jpg", "jpeg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)

def upload_predict(image, model):
    
    return classes[str(model.predict(preprocess(image)).argmax())]
    
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predicted_class = upload_predict(image, model)
    st.write("The image is classified as",predicted_class)