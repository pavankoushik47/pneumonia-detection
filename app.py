import tensorflow as tf 
import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config("AI Pneumonia Detection","🫁",layout="centered")
 
st.title("AI powered Pneumonia detection")
st.caption("chest X-ray analysis using Deep Learning")
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pneumonia_vgg16_model.h5")
model=load_model()
file=st.file_uploader("upload your chest x-ray (JPG/PNG)",["jpg","jpeg","png"])
if file:
    img=Image.open(file).convert("RGB")
    st.image(img)
    if st.button("Analyze the X-ray"):
        img_resized=img.resize((224,224))
        arr=image.img_to_array(img_resized)/255.0
        arr=np.expand_dims(arr,0)
        prob=model.predict(arr)[0][0]
        label= "PNEUMONIA IS DETECTED" if prob > 0.5 else "NORMAL LUNGS"
        st.subheader("Diagnosis Result::")
        st.markdown(
            f"<div class='result {'bad' if label=='PNEUMONIA IS DETECTED' else 'NORMAL LUNGS'}'>"
            f"{label}<br>Confidence: {prob*100:.2f}%</div>",
            unsafe_allow_html=True
        )
        
st.warning("this is only for educational use. Not a tool for real-time doctors")