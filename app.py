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
    gdown.download(url, 'modelo_quantizado16bits.tflite')
    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
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

def previsao(interpreter, image):
    # Obtém os detalhes da entrada do modelo (por exemplo: shape, índice do tensor)
    input_details = interpreter.get_input_details()
    
    # Obtém os detalhes da saída do modelo
    output_details = interpreter.get_output_details()
    
    # Define a imagem (pré-processada) como entrada no modelo
    interpreter.set_tensor(input_details[0]['index'], image)
    
    # Executa a inferência (processo de predição) com o modelo TFLite
    interpreter.invoke()
    
    # Obtém a saída do modelo — geralmente, as probabilidades para cada classe
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Lista com os nomes das classes, correspondentes à ordem da saída do modelo
    classes = ['BlackMeasles', 'BlackRot', 'HealthyGrapes', 'LeafBlight']
    
    # Cria um DataFrame para exibir as classes e suas probabilidades
    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100 * output_data[0]  # Converte para porcentagem
    
    # Cria um gráfico de barras horizontal com Plotly para exibir as probabilidades
    fig = px.bar(
        df,
        y='classes',
        x='probabilidades (%)',
        orientation='h',
        text='probabilidades (%)',
        title='Probabilidade de Classes de Doenças em Uvas'
    )
    
    # Exibe o gráfico interativo na interface do Streamlit
    st.plotly_chart(fig)

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
    if image is not None:
        previsao(interpreter, image)

if __name__ == "__main__":
    main()