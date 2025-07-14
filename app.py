import streamlit as st 
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
@st.cache_resource
def carrega_modelo():
    url = 'https://drive.google.com/uc?id=1GpWix8dp6FeFAs6g0etbnw_avu9Aflfp'
    gdown.down(url, 'modelo_quantizado16bits.tflite')
    interpreter = tf.lite.interpreter(model_path='modelo_quantizado16bits.tflite')
    interpreter.allocate_tensors()

    return interpreter

def carrega_imagem():
    uploaded_file = st.file_uploader('Arraste e solte uma imagem, ou clique para selecionar uma', type=['png','jpg','jpeg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))
        st.image(image)
        st.success('Imagem foi carregada com sucesso!')

        image = np.array(image, dtype=np.float32)

        image = image/255.0
        image = np.expand_dims(image, axis=0)
        return image

def main():
    st.set_page_config(
        page_title = "Classifica folhas de videiras!"
    )
    st.write("# Classifica folhas de videiras!")

    #carrega o modelo
    interpreter = carrega_modelo()
    #carrega a imagem
    image = carrega_imagem()
    #classifica a imagem

if __name__ == "__main__":
    main()