import streamlit as st
import pandas as pd
import numpy as np
# Certifique-se de que todas as funções pycaret necessárias estão importadas
from pycaret.classification import load_model, predict_model, setup, pull, compare_models, finalize_model, save_model
from sklearn.datasets import load_iris
from transformers import pipeline # Já importado, ótimo!

import matplotlib.pyplot as plt
import os

# --- Configurações Iniciais ---
# Determine o diretório do script atual
script_dir = os.path.dirname(__file__)
# Construa o caminho completo para o arquivo do modelo padrão
default_model_path = os.path.join(script_dir, "model")

# Carrega o dataset Iris para referência de features e target_names
iris = load_iris(as_frame=True)
default_iris_df = iris.frame
default_iris_df['target'] = iris.target # Adiciona a coluna target ao DataFrame do Iris
default_feature_names = iris.feature_names
default_target_name = 'target' # Nome padrão para a coluna target no dataframe Iris

# --- Inicializa st.session_state ---
# O Streamlit re-executa o script inteiro a cada interação.
# st.session_state é a forma correta de persistir dados entre re-execuções.

# Inicializa o modelo atual
if 'current_model' not in st.session_state:
    try:
        st.session_state['current_model'] = load_model(default_model_path)
    except Exception as e:
        st.warning(f"Não foi possível carregar o modelo padrão 'model.pkl'. Por favor, faça upload de um CSV para treinar um novo modelo ou verifique o arquivo. Erro: {e}")
        st.session_state['current_model'] = None

# Inicializa o dataframe de comparação atual
if 'comparison_df' not in st.session_state:
    try:
        comparison_path = os.path.join(script_dir, "model_comparison.csv")
        st.session_state['comparison_df'] = pd.read_csv(comparison_path)
    except FileNotFoundError:
        st.session_state['comparison_df'] = None

# Inicializa as features e o target_name do modelo atual
if 'current_model_features' not in st.session_state:
    st.session_state['current_model_features'] = default_feature_names
if 'current_model_target' not in st.session_state:
    st.session_state['current_model_target'] = default_target_name

# Inicializa o DataFrame customizado carregado (para sliders na página de previsão)
if 'uploaded_custom_df' not in st.session_state:
    st.session_state['uploaded_custom_df'] = None # Inicialmente None

# --- Layout do Streamlit ---
st.sidebar.title("🧪 Classificador de Íris/Dados Customizados")
# Adicione a nova opção "Análise de Sentimento" ao radio button
page = st.sidebar.radio("Escolha uma página", ["🧬 Prever", "📊 Comparar Modelos", "➕ Adicionar Novo Banco de Dados", "💬 Análise de Sentimento"])

# --- Página: Adicionar Novo Banco de Dados ---
if page == "➕ Adicionar Novo Banco de Dados":
    st.title("➕ Treinar um Novo Modelo com Dados Customizados")
    st.write("Faça o upload de um arquivo CSV para treinar um novo modelo de classificação.")

    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    if uploaded_file is not None:
        st.info("Arquivo carregado. Por favor, especifique a coluna alvo para classificação.")

        custom_df = pd.read_csv(uploaded_file)
        # Armazena o DataFrame customizado no session_state imediatamente após o upload
        st.session_state['uploaded_custom_df'] = custom_df

        st.subheader("Prévia dos Dados Carregados:")
        st.dataframe(custom_df.head())

        available_columns = custom_df.columns.tolist()
        target_column = st.selectbox("Selecione a coluna alvo (target) para classificação:", available_columns)

        if st.button("Treinar Novo Modelo"):
            if target_column:
                with st.spinner("Treinando e comparando modelos... Isso pode levar alguns minutos."):
                    try:
                        # Configura o PyCaret com os dados customizados
                        exp_setup = setup(data=custom_df, target=target_column, session_id=456, verbose=False, html=False)

                        # Compara os modelos
                        best_custom_model = compare_models(sort='Accuracy')

                        # Salva a tabela de comparação no disco
                        custom_comparison_df = pull()
                        custom_comparison_df.to_csv(os.path.join(script_dir, "custom_model_comparison.csv"), index=False)

                        # Finaliza e salva o melhor modelo no disco
                        final_custom_model = finalize_model(best_custom_model)
                        save_model(final_custom_model, os.path.join(script_dir, "custom_model")) # Salva como 'custom_model.pkl'

                        st.success("Novo modelo treinado e salvo com sucesso!")
                        st.write("Você pode agora ir para a página 'Prever' ou 'Comparar Modelos' para usar/ver o novo modelo.")

                        # Atualiza o estado da sessão com o novo modelo e dados
                        st.session_state['current_model'] = final_custom_model
                        st.session_state['comparison_df'] = custom_comparison_df
                        st.session_state['current_model_features'] = [col for col in custom_df.columns if col != target_column]
                        st.session_state['current_model_target'] = target_column

                    except Exception as e:
                        st.error(f"Ocorreu um erro durante o treinamento do modelo: {e}")
                        st.info("Verifique se os dados estão limpos e no formato correto para classificação.")
            else:
                st.warning("Por favor, selecione a coluna alvo antes de treinar o modelo.")

# --- Página: Prever ---
elif page == "🧬 Prever":
    st.title("🌸 Previsão da Espécie de Íris ou Dados Customizados")

    # Recupera o modelo, features e target do st.session_state
    current_model = st.session_state.get('current_model')
    current_feature_names = st.session_state.get('current_model_features')
    current_target_name = st.session_state.get('current_model_target')

    if current_model is None:
        st.warning("Nenhum modelo carregado ou treinado. Por favor, adicione um novo banco de dados para treinar um modelo.")
    else:
        st.subheader("Insira os valores para a previsão:")

        # Tenta obter o DataFrame customizado do session_state para os sliders.
        # Se não houver, usa o DataFrame Iris padrão.
        data_for_sliders = st.session_state.get('uploaded_custom_df')
        if data_for_sliders is None:
            data_for_sliders = default_iris_df # Usa o DataFrame Iris padrão se nenhum customizado foi carregado

        input_values = {}
        for feature in current_feature_names:
            if feature in data_for_sliders.columns and \
               (data_for_sliders[feature].dtype == 'float64' or data_for_sliders[feature].dtype == 'int64'):
                min_val = float(data_for_sliders[feature].min())
                max_val = float(data_for_sliders[feature].max())
                default_val = float(data_for_sliders[feature].mean())
                input_values[feature] = st.slider(f"{feature}", min_val, max_val, default_val)
            else: # Se não for numérico ou não tiver um range conhecido
                input_values[feature] = st.text_input(f"{feature}", "")


        input_data = pd.DataFrame([input_values])
        # Garante que as colunas estejam na ordem esperada pelo modelo
        input_data = input_data[current_feature_names]

        if st.button("Prever"):
            prediction = predict_model(current_model, data=input_data)

            pred_label = prediction["prediction_label"][0]

            # A lógica para exibir a label deve ser robusta para Iris e customizados
            # Se o target é numérico e representa classes (como o Iris), é melhor ter um mapeamento.
            # Para datasets customizados, assume que a label de previsão é a classe diretamente.
            if current_target_name == default_target_name and isinstance(pred_label, (int, float)):
                # Se for o modelo Iris padrão e a previsão for um número (0, 1, 2)
                st.success(f"Espécie prevista: **{iris.target_names[int(pred_label)]}**")
            else:
                # Para modelos customizados ou quando a previsão já é a string da classe
                st.success(f"Classe prevista: **{pred_label}**")

            if 'prediction_score' in prediction:
                st.write(f"Confiança da previsão: {prediction['prediction_score'][0]:.2f}")


# --- Página: Comparar Modelos ---
elif page == "📊 Comparar Modelos":
    st.title("📊 Comparação de Modelos (Acurácia)")

    # Recupera o dataframe de comparação do st.session_state
    current_comparison_df = st.session_state.get('comparison_df')

    if current_comparison_df is None or current_comparison_df.empty:
        st.warning("Nenhum dado de comparação disponível. Treine um modelo para ver a comparação.")
    else:
        st.dataframe(current_comparison_df)
        st.bar_chart(current_comparison_df.set_index("Model")["Accuracy"])

# --- Nova Página: Análise de Sentimento ---
elif page == "💬 Análise de Sentimento":
    st.title("💬 Análise de Sentimento com 🤗 Transformers")
    st.write("Insira um texto ou uma lista de textos para analisar o sentimento.")

    # Inicializa o pipeline de sentimento com o modelo ESPECÍFICO
    @st.cache_resource # Usa st.cache_resource para armazenar o pipeline em cache
    def load_specific_sentiment_pipeline():
        st.info("Carregando modelo de Análise de Sentimento: `finiteautomata/bertweet-base-sentiment-analysis`... (Pode levar alguns segundos na primeira vez)")
        # Carrega o modelo específico aqui
        return pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")

    # Chama a função para carregar o pipeline com o modelo específico
    sentiment_pipeline = load_specific_sentiment_pipeline()

    text_input = st.text_area("Insira o texto (ou textos separados por nova linha):", height=150)

    if st.button("Analisar Sentimento"):
        if text_input:
            # Divide a entrada em uma lista de strings, uma por linha
            data_for_sentiment = [t.strip() for t in text_input.split('\n') if t.strip()]

            if data_for_sentiment:
                with st.spinner("Analisando sentimento..."):
                    # Executa o pipeline com os dados de entrada
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