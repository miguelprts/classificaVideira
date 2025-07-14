import streamlit as st 
import gdown
import tensorflow as tf

def carrega_modelo():
    url = 'https://drive.google.com/uc?id=1GpWix8dp6FeFAs6g0etbnw_avu9Aflfp'
    gdown.down(url, 'modelo_quantizado16bits.tflite')
    interpreter = tf.lite.interpreter(model_path='modelo_quantizado16bits.tflite')
    interpreter.allocate_tensors()

    return interpreter

def main():
    st.set_page_config(
        page_title = "Classifica folhas de videiras!"
    )
    st.write("# Classifica folhas de videiras!")

    #carrega o modelo

    #carrega a imagem

    #classifica a imagem

if __name__ == "__main__":
    main()