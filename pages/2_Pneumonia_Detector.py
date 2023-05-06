import streamlit as st

import json

import tensorflow as tf
from keras.models import load_model
from PIL import Image


def load_model_():
    model = load_model("./models/pneumonia_classification_model")
    return model
with st.spinner('Resources are being loaded...'):
  model=load_model_()
 
st.write("""
         # Pneumonia Detector
         """)

file = st.file_uploader("Upload an X-Ray to detect pneumonia", type=["jpg", "jpeg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)

def upload_predict(image, model):
    
    img_to_tensor = tf.expand_dims(tf.cast(tf.convert_to_tensor(image), dtype="float32")/255.0, axis=0)
    idx = model.predict(img_to_tensor).argmax()

    return idx
    
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file).resize((100,100))
    st.image(image, use_column_width=True)
    predicted_class = upload_predict(image, model)
    if predicted_class == 1:
        st.warning("The image shows presence of pneumonia.")
    else:
        st.success("The image doesn't show any presence of pneumonia.")
