
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Predição de Obesidade - Sistema e Painel", layout="wide")

# ============================
# 📂 Carregar Dados
# ============================
df = pd.read_csv('Obesity.csv')

# ✔️ Renomear colunas
df.rename(columns={
    'Gender': 'genero',
    'Age': 'idade',
    'Height': 'altura',
    'Weight': 'peso',
    'family_history_with_overweight': 'historico_familiar',
    'FAVC': 'consome_alta_calorias_frequente',
    'FCVC': 'consumo_vegetais',
    'NCP': 'qtde_refeicoes_principais',
    'CAEC': 'alimentacao_entre_refeicoes',
    'SMOKE': 'fuma',
    'CH2O': 'qtde_agua_diaria',
    'SCC': 'monitora_calorias',
    'FAF': 'freq_atividade_fisica',
    'TUE': 'tempo_uso_dispositivos',
    'CALC': 'freq_consumo_alcool',
    'MTRANS': 'meio_transporte_contumaz',
    'Obesity': 'nivel_obesidade'
}, inplace=True)

# ✔️ Mapear níveis de obesidade
mapa_obesidade = {
    'Insufficient_Weight': 'Abaixo do Peso',
    'Normal_Weight': 'Peso Normal',
    'Overweight_Level_I': 'Sobrepeso - Nível I',
    'Overweight_Level_II': 'Sobrepeso - Nível II',
    'Obesity_Type_I': 'Obesidade - Nível I',
    'Obesity_Type_II': 'Obesidade - Nível II',
    'Obesity_Type_III': 'Obesidade - Nível III'
}
ordem_obesidade = list(mapa_obesidade.values())

df['nivel_obesidade'] = df['nivel_obesidade'].map(mapa_obesidade)
df['nivel_obesidade'] = pd.Categorical(df['nivel_obesidade'], categories=ordem_obesidade, ordered=True)
df['imc'] = df['peso'] / (df['altura']**2)

# ============================
# 🔗 Carregar modelo e artefatos
# ============================
modelo = joblib.load('modelo_obesidade.pkl')
scaler = joblib.load('scaler.pkl')
le_target = joblib.load('label_encoder_target.pkl')
le_categoricas = joblib.load('label_encoders_categoricas.pkl')

# ============================
# 🧠 Layout - Menu
# ============================
st.sidebar.title("Menu")
aba = st.sidebar.radio("Escolha a seção:", ["Sistema Preditivo", "Painel Analítico"])

# ============================
# 🎯 Sistema Preditivo
# ============================
if aba == "Sistema Preditivo":
    st.title("🔬 Sistema Preditivo de Obesidade")

    genero = st.selectbox('Gênero', df['genero'].unique())
    idade = st.slider('Idade', 10, 100, 30)
    altura = st.slider('Altura (m)', 1.0, 2.2, 1.70)
    peso = st.slider('Peso (kg)', 30.0, 200.0, 70.0)
    historico = st.selectbox('Histórico Familiar', df['historico_familiar'].unique())
    consome_calorias = st.selectbox('Consome Alta Caloria', df['consome_alta_calorias_frequente'].unique())
    consumo_vegetais = st.slider('Consumo de Vegetais (1-3)', 1, 3, 2)
    qtde_refeicoes = st.slider('Refeições Principais', 1, 4, 3)
    alimentacao = st.selectbox('Alimentação entre Refeições', df['alimentacao_entre_refeicoes'].unique())
    fuma = st.selectbox('Fuma?', df['fuma'].unique())
    qtde_agua = st.slider('Água por dia (L)', 1.0, 3.0, 2.0)
    monitora_calorias = st.selectbox('Monitora Calorias?', df['monitora_calorias'].unique())
    atividade = st.selectbox('Atividade Física', df['freq_atividade_fisica'].unique())
    tempo_dispositivo = st.slider('Tempo em Dispositivos (h)', 0.0, 5.0, 2.0)
    alcool = st.selectbox('Consumo de Álcool', df['freq_consumo_alcool'].unique())
    transporte = st.selectbox('Meio de Transporte', df['meio_transporte_contumaz'].unique())

    entrada = pd.DataFrame([[
        genero, idade, altura, peso, historico, consome_calorias,
        consumo_vegetais, qtde_refeicoes, alimentacao, fuma, qtde_agua,
        monitora_calorias, atividade, tempo_dispositivo, alcool, transporte
    ]], columns=modelo.feature_names_in_)

    for col in le_categoricas.keys():
        entrada[col] = le_categoricas[col].transform(entrada[col])

    cols_numericas = ['idade', 'altura', 'peso', 'qtde_refeicoes_principais',
                       'qtde_agua_diaria', 'tempo_uso_dispositivos']
    entrada[cols_numericas] = scaler.transform(entrada[cols_numericas])

    if st.button("Realizar Previsão"):
        resultado = modelo.predict(entrada)
        classe = le_target.inverse_transform(resultado)[0]
        st.success(f"Resultado da Predição: **{classe}**")

# ============================
# 📊 Painel Analítico
# ============================
if aba == "Painel Analítico":
    st.title("📊 Painel Analítico de Obesidade")

    # ✔️ Gráfico 1 - Distribuição dos Níveis de Obesidade
    fig1, ax1 = plt.subplots(figsize=(6,4))
    sns.countplot(data=df, y='nivel_obesidade', color='red', order=ordem_obesidade, ax=ax1)
    st.pyplot(fig1)

    # ✔️ Gráfico 2 - Distribuição do IMC
    fig2, ax2 = plt.subplots(figsize=(7,4))
    sns.violinplot(data=df, x='nivel_obesidade', y='imc', palette='Reds', order=ordem_obesidade, ax=ax2)
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)

    # ✔️ Gráfico 3 - Idade, altura, peso
    fig3, axs = plt.subplots(1, 3, figsize=(15,4))
    sns.histplot(df['idade'], kde=True, color='red', ax=axs[0])
    sns.histplot(df['altura'], kde=True, color='orange', ax=axs[1])
    sns.histplot(df['peso'], kde=True, color='blue', ax=axs[2])
    st.pyplot(fig3)

