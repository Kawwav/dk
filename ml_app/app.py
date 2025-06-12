import streamlit as st
import pandas as pd
import numpy as np

from pycaret.classification import load_model, predict_model, setup, pull, compare_models, finalize_model, save_model
from sklearn.datasets import load_iris
from transformers import pipeline 

import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(__file__)

default_model_path = os.path.join(script_dir, "model")

iris = load_iris(as_frame=True)
default_iris_df = iris.frame
default_iris_df['target'] = iris.target 
default_feature_names = iris.feature_names
default_target_name = 'target' 


if 'current_model' not in st.session_state:
    try:
        st.session_state['current_model'] = load_model(default_model_path)
    except Exception as e:
        st.warning(f"Não foi possível carregar o modelo padrão 'model.pkl'. Por favor, faça upload de um CSV para treinar um novo modelo ou verifique o arquivo. Erro: {e}")
        st.session_state['current_model'] = None

if 'comparison_df' not in st.session_state:
    try:
        comparison_path = os.path.join(script_dir, "model_comparison.csv")
        st.session_state['comparison_df'] = pd.read_csv(comparison_path)
    except FileNotFoundError:
        st.session_state['comparison_df'] = None

if 'current_model_features' not in st.session_state:
    st.session_state['current_model_features'] = default_feature_names
if 'current_model_target' not in st.session_state:
    st.session_state['current_model_target'] = default_target_name

if 'uploaded_custom_df' not in st.session_state:
    st.session_state['uploaded_custom_df'] = None 

st.sidebar.title("🧪 Classificador de Íris/Dados Customizados")
page = st.sidebar.radio("Escolha uma página", ["🧬 Prever", "📊 Comparar Modelos", "➕ Adicionar Novo Banco de Dados", "💬 Análise de Sentimento"])


if page == "➕ Adicionar Novo Banco de Dados":
    st.title("➕ Treinar um Novo Modelo com Dados Customizados")
    st.write("Faça o upload de um arquivo CSV para treinar um novo modelo de classificação.")

    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    if uploaded_file is not None:
        st.info("Arquivo carregado. Por favor, especifique a coluna alvo para classificação.")

        custom_df = pd.read_csv(uploaded_file)

        st.session_state['uploaded_custom_df'] = custom_df

        st.subheader("Prévia dos Dados Carregados:")
        st.dataframe(custom_df.head())

        available_columns = custom_df.columns.tolist()
        target_column = st.selectbox("Selecione a coluna alvo (target) para classificação:", available_columns)

        if st.button("Treinar Novo Modelo"):
            if target_column:
                with st.spinner("Treinando e comparando modelos... Isso pode levar alguns minutos."):
                    try:
                       
                        exp_setup = setup(data=custom_df, target=target_column, session_id=456, verbose=False, html=False)

                       
                        best_custom_model = compare_models(sort='Accuracy')

                        
                        custom_comparison_df = pull()
                        custom_comparison_df.to_csv(os.path.join(script_dir, "custom_model_comparison.csv"), index=False)

                        
                        final_custom_model = finalize_model(best_custom_model)
                        save_model(final_custom_model, os.path.join(script_dir, "custom_model")) 

                        st.success("Novo modelo treinado e salvo com sucesso!")
                        st.write("Você pode agora ir para a página 'Prever' ou 'Comparar Modelos' para usar/ver o novo modelo.")
                        st.session_state['current_model'] = final_custom_model
                        st.session_state['comparison_df'] = custom_comparison_df
                        st.session_state['current_model_features'] = [col for col in custom_df.columns if col != target_column]
                        st.session_state['current_model_target'] = target_column

                    except Exception as e:
                        st.error(f"Ocorreu um erro durante o treinamento do modelo: {e}")
                        st.info("Verifique se os dados estão limpos e no formato correto para classificação.")
            else:
                st.warning("Por favor, selecione a coluna alvo antes de treinar o modelo.")


elif page == "🧬 Prever":
    st.title("🌸 Previsão da Espécie de Íris ou Dados Customizados")

   
    current_model = st.session_state.get('current_model')
    current_feature_names = st.session_state.get('current_model_features')
    current_target_name = st.session_state.get('current_model_target')

    if current_model is None:
        st.warning("Nenhum modelo carregado ou treinado. Por favor, adicione um novo banco de dados para treinar um modelo.")
    else:
        st.subheader("Insira os valores para a previsão:")

        data_for_sliders = st.session_state.get('uploaded_custom_df')
        if data_for_sliders is None:
            data_for_sliders = default_iris_df 

        input_values = {}
        for feature in current_feature_names:
            if feature in data_for_sliders.columns and \
               (data_for_sliders[feature].dtype == 'float64' or data_for_sliders[feature].dtype == 'int64'):
                min_val = float(data_for_sliders[feature].min())
                max_val = float(data_for_sliders[feature].max())
                default_val = float(data_for_sliders[feature].mean())
                input_values[feature] = st.slider(f"{feature}", min_val, max_val, default_val)
            else: 
                input_values[feature] = st.text_input(f"{feature}", "")


        input_data = pd.DataFrame([input_values])
        
        input_data = input_data[current_feature_names]

        if st.button("Prever"):
            prediction = predict_model(current_model, data=input_data)

            pred_label = prediction["prediction_label"][0]

           
            if current_target_name == default_target_name and isinstance(pred_label, (int, float)):
                
                st.success(f"Espécie prevista: **{iris.target_names[int(pred_label)]}**")
            else:
                
                st.success(f"Classe prevista: **{pred_label}**")

            if 'prediction_score' in prediction:
                st.write(f"Confiança da previsão: {prediction['prediction_score'][0]:.2f}")



elif page == "📊 Comparar Modelos":
    st.title("📊 Comparação de Modelos (Acurácia)")

    
    current_comparison_df = st.session_state.get('comparison_df')

    if current_comparison_df is None or current_comparison_df.empty:
        st.warning("Nenhum dado de comparação disponível. Treine um modelo para ver a comparação.")
    else:
        st.dataframe(current_comparison_df)
        st.bar_chart(current_comparison_df.set_index("Model")["Accuracy"])


elif page == "💬 Análise de Sentimento":
    st.title("💬 Análise de Sentimento com 🤗 Transformers")
    st.write("Insira um texto ou uma lista de textos para analisar o sentimento.")

    
    @st.cache_resource 
    def load_sentiment_pipeline():
        st.info("Carregando modelo de Análise de Sentimento... (Pode levar alguns segundos na primeira vez)")
        return pipeline("sentiment-analysis")

    sentiment_pipeline = load_sentiment_pipeline()

    text_input = st.text_area("Insira o texto (ou textos separados por nova linha):", height=150)

    if st.button("Analisar Sentimento"):
        if text_input:
            
            data_for_sentiment = [t.strip() for t in text_input.split('\n') if t.strip()]

            if data_for_sentiment:
                with st.spinner("Analisando sentimento..."):
                    results = sentiment_pipeline(data_for_sentiment)

                st.subheader("Resultados da Análise de Sentimento:")
                for i, res in enumerate(results):
                    st.write(f"**Texto:** `{data_for_sentiment[i]}`")
                    st.write(f"**Sentimento:** {res['label']} (Confiança: {res['score']:.2f})")
                    st.write("---")
            else:
                st.warning("Por favor, insira algum texto para análise.")
        else:
            st.warning("Por favor, insira algum texto para análise.")

            [{'label': 'POSITIVE', 'score': 0.9998},
 {'label': 'NEGATIVE', 'score': 0.9991}]

specific_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
specific_model(data)


#cd "C:\Users\kawav\OneDrive\Área de Trabalho\dk-main\dk-main\ml_app"
#python train_model.py
#streamlit run app.py