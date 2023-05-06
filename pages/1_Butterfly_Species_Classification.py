import streamlit as st

import json

import tensorflow as tf
from keras.models import load_model
from PIL import Image


def load_model_():
    model = load_model("./models/butterfly_classification_model")
    with open('assets/butterfly_classes.json', 'r') as f:
        classes = json.load(f)
    return model, classes
with st.spinner('Resources are being loaded...'):
  model, classes=load_model_()
 
st.write("""
         # Butterfly Species Classification
         """)

file = st.file_uploader("Upload the image to be classified", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)

def upload_predict(image, model):
    
    img_to_tensor = tf.convert_to_tensor(image)
    idx = model.predict(tf.image.resize(tf.expand_dims(img_to_tensor, axis=0),[100,100])).argmax()
    for key in classes.keys():
        if classes[key] == idx:
            return key
    
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predicted_class = upload_predict(image, model)
    st.write("The image is classified as",predicted_class)


