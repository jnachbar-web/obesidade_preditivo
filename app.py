
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ✔️ Configuração da página
st.set_page_config(
    page_title="Sistema Preditivo e Analítico de Obesidade",
    layout="wide"
)

st.sidebar.title("Menu")
aba = st.sidebar.radio("Escolha uma aba:", ["Sistema Preditivo", "Painel Analítico"])

# ✔️ Carregar a base de dados (para o painel analítico)
df = pd.read_csv('Obesity.csv')

# ✔️ Renomear colunas para português
df.rename(columns={
    'Gender':'genero', 'Age':'idade', 'Height':'altura', 'Weight':'peso',
    'family_history':'historico_familiar', 'FAVC':'consome_alta_calorias_frequente',
    'FCVC':'consumo_vegetais', 'NCP':'qtde_refeicoes_principais',
    'CAEC':'alimentacao_entre_refeicoes', 'SMOKE':'fuma', 'CH2O':'qtde_agua_diaria',
    'SCC':'monitora_calorias', 'FAF':'freq_atividade_fisica',
    'TUE':'tempo_uso_dispositivos', 'CALC':'freq_consumo_alcool',
    'MTRANS':'meio_transporte_contumaz', 'NObeyesdad':'nivel_obesidade'
}, inplace=True)

# ✔️ Carregar os arquivos salvos
modelo = joblib.load('modelo_obesidade.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder_target.pkl')

# ===================================================================
# 🎯 Aba — SISTEMA PREDITIVO
# ===================================================================

if aba == "Sistema Preditivo":
    st.title("🔬 Sistema Preditivo de Obesidade")

    st.subheader("Preencha os dados do paciente:")

    # ✔️ Inputs
    genero = st.selectbox("Gênero", ["Male", "Female"])
    idade = st.slider("Idade", 10, 100, 25)
    altura = st.slider("Altura (em metros)", 1.0, 2.2, 1.70)
    peso = st.slider("Peso (kg)", 30.0, 200.0, 70.0)
    historico = st.selectbox("Histórico Familiar de Obesidade", ["yes", "no"])
    consome_calorias = st.selectbox("Consome Alimentos Calóricos?", ["yes", "no"])
    consumo_vegetais = st.slider("Consumo de Vegetais (1 a 3)", 1.0, 3.0, 2.0)
    qtde_refeicoes = st.slider("Refeições Principais por Dia", 1.0, 4.0, 3.0)
    alimentacao = st.selectbox("Come entre Refeições?", ["no", "Sometimes", "Frequently", "Always"])
    fuma = st.selectbox("Fuma?", ["yes", "no"])
    qtde_agua = st.slider("Litros de Água por Dia", 1.0, 3.0, 2.0)
    monitora_calorias = st.selectbox("Monitora Calorias?", ["yes", "no"])
    atividade = st.selectbox("Nível de Atividade Física", ["Nunca", "Pouquíssima", "Moderada", "Frequente"])
    tempo_dispositivo = st.slider("Horas em Dispositivos por Dia", 0.0, 3.0, 1.0)
    alcool = st.selectbox("Consumo de Álcool", ["no", "Sometimes", "Frequently", "Always"])
    transporte = st.selectbox("Meio de Transporte Predominante", [
        "Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

    # ✔️ Mapear variáveis categóricas
    mapeamentos = {
        'genero': {'Male': 1, 'Female': 0},
        'historico_familiar': {'yes': 1, 'no': 0},
        'consome_alta_calorias_frequente': {'yes': 1, 'no': 0},
        'alimentacao_entre_refeicoes': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
        'fuma': {'yes': 1, 'no': 0},
        'monitora_calorias': {'yes': 1, 'no': 0},
        'freq_consumo_alcool': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
        'meio_transporte_contumaz': {
            'Automobile': 0, 'Bike': 1, 'Motorbike': 2,
            'Public_Transportation': 3, 'Walking': 4
        },
        'freq_atividade_fisica': {'Nunca': 0, 'Pouquíssima': 1, 'Moderada': 2, 'Frequente': 3},
            'consumo_vegetais': {1.0: 1, 2.0: 2, 3.0: 3}  # ✔️ 🔥 Este é o mapeamento que faltava!
    }

    # ✔️ Vetor categórico
    entrada_categorica = [
        mapeamentos['genero'][genero],
        mapeamentos['historico_familiar'][historico],
        mapeamentos['consome_alta_calorias_frequente'][consome_calorias],
        mapeamentos['consumo_vegetais'][consumo_vegetais],
        mapeamentos['alimentacao_entre_refeicoes'][alimentacao],
        mapeamentos['fuma'][fuma],
        mapeamentos['monitora_calorias'][monitora_calorias],
        mapeamentos['freq_consumo_alcool'][alcool],
        mapeamentos['meio_transporte_contumaz'][transporte],
        mapeamentos['freq_atividade_fisica'][atividade]
    ]

    # ✔️ Vetor numérico (será escalado)
    entrada_numerica = np.array([[
        idade,
        altura,
        peso,
        qtde_refeicoes,
        qtde_agua,
        tempo_dispositivo
    ]])

    entrada_numerica_escalada = scaler.transform(entrada_numerica)
    entrada_final = np.hstack([entrada_categorica, entrada_numerica_escalada[0]])

    if st.button("Realizar Previsão"):
        resultado = modelo.predict([entrada_final])
        classe = label_encoder.inverse_transform(resultado)[0]
        st.success(f"🔍 Resultado: **{classe.replace('_', ' ')}**")

# ===================================================================
# 🎯 Aba — PAINEL ANALÍTICO
# ===================================================================

if aba == "Painel Analítico":
    st.title("📊 Painel Analítico — Obesidade")

    abas = st.tabs(["Visão Geral", "Distribuições", "Correlação"])

    with abas[0]:
        st.subheader("Distribuição dos Níveis de Obesidade")
        fig, ax = plt.subplots(figsize=(8,5))
        sns.countplot(data=df, y='nivel_obesidade', order=df['nivel_obesidade'].value_counts().index, color='red', ax=ax)
        ax.set_xlabel('Quantidade')
        st.pyplot(fig)

    with abas[1]:
        st.subheader("Distribuição da Idade e Peso")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(df['idade'], kde=True, ax=ax, color='red')
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.histplot(df['peso'], kde=True, ax=ax, color='red')
            st.pyplot(fig)

        st.subheader("Distribuição de Tempo em Dispositivos")
        fig, ax = plt.subplots()
        sns.violinplot(data=df, x='nivel_obesidade', y='tempo_uso_dispositivos', palette='Reds', ax=ax)
        st.pyplot(fig)

    with abas[2]:
        st.subheader("Matriz de Correlação")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='Reds', ax=ax)
        st.pyplot(fig)
