
# ======================
# 📦 Importações
# ======================
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ======================
# 📂 Carregar Artefatos
# ======================
modelo = joblib.load('modelo_obesidade.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder_target.pkl')

df = pd.read_csv('Obesity.csv')

# ======================
# 🏷️ Renomear Colunas
# ======================
df.rename(columns={
    'Gender':'genero', 'Age':'idade', 'Height':'altura', 'Weight':'peso',
    'family_history':'historico_familiar', 'FAVC':'consome_alta_calorias_frequente',
    'FCVC':'consumo_vegetais', 'NCP':'qtde_refeicoes_principais',
    'CAEC':'alimentacao_entre_refeicoes', 'SMOKE':'fuma', 'CH2O':'qtde_agua_diaria',
    'SCC':'monitora_calorias', 'FAF':'freq_atividade_fisica',
    'TUE':'tempo_uso_dispositivos', 'CALC':'freq_consumo_alcool',
    'MTRANS':'meio_transporte_contumaz', 'NObeyesdad':'nivel_obesidade',
    'Obesity':'nivel_obesidade'
}, inplace=True)

# ======================
# 🎨 Sidebar Navegação
# ======================
st.sidebar.title("Menu")
aba = st.sidebar.radio("Escolha uma aba:", ["Sistema Preditivo", "Painel Analítico"])

# ===================================================
# 🔬 🧠 Aba — Sistema Preditivo
# ===================================================
if aba == "Sistema Preditivo":
    st.title("🔬 Sistema Preditivo de Obesidade")

    st.subheader("📄 Informe os dados do paciente:")

    # ======================
    # 🗺️ Mapeamentos
    # ======================
    genero_map = {'Masculino': 'Male', 'Feminino': 'Female'}
    historico_map = {'Sim': 'yes', 'Não': 'no'}
    alta_caloria_map = {'Sim': 'yes', 'Não': 'no'}
    alimentacao_map = {'Não': 'no', 'Às vezes': 'Sometimes', 'Frequente': 'Frequently', 'Sempre': 'Always'}
    fuma_map = {'Sim': 'yes', 'Não': 'no'}
    monitora_map = {'Sim': 'yes', 'Não': 'no'}
    alcool_map = {'Não': 'no', 'Às vezes': 'Sometimes', 'Frequente': 'Frequently', 'Sempre': 'Always'}
    transporte_map = {
        'Automóvel': 'Automobile',
        'Motocicleta': 'Motorbike',
        'Bicicleta': 'Bike',
        'Transporte Público': 'Public_Transportation',
        'Caminhada': 'Walking'
    }
    atividade_map = {'Nunca': 0, 'Pouquíssima': 1, 'Moderada': 2, 'Frequente': 3}

    # ======================
    # 🎯 Inputs do Usuário
    # ======================
    genero = st.selectbox('Gênero', list(genero_map.keys()))
    historico = st.selectbox('Histórico Familiar de Obesidade', list(historico_map.keys()))
    consome_calorias = st.selectbox('Consome alimentos calóricos com frequência?', list(alta_caloria_map.keys()))
    alimentacao = st.selectbox('Come entre as refeições?', list(alimentacao_map.keys()))
    fuma = st.selectbox('Fuma?', list(fuma_map.keys()))
    monitora_calorias = st.selectbox('Monitora as calorias ingeridas?', list(monitora_map.keys()))
    alcool = st.selectbox('Frequência de consumo de álcool', list(alcool_map.keys()))
    transporte = st.selectbox('Meio de transporte mais usado', list(transporte_map.keys()))
    atividade = st.selectbox('Frequência de atividade física', list(atividade_map.keys()))
    consumo_vegetais = st.selectbox('Consumo de vegetais nas refeições', 
                                     [1, 2, 3],
                                     format_func=lambda x: {1: 'Baixo', 2: 'Médio', 3: 'Alto'}.get(x))

    idade = st.slider('Idade', 10, 100, 30)
    altura = st.slider('Altura (em metros)', 1.0, 2.2, 1.70)
    peso = st.slider('Peso (kg)', 30.0, 200.0, 70.0)
    qtde_refeicoes = st.slider('Refeições principais por dia', 1, 4, 3)
    qtde_agua = st.slider('Litros de água por dia', 1.0, 3.0, 2.0)
    tempo_dispositivo = st.slider('Horas de uso de dispositivos por dia', 0.0, 5.0, 2.0)

    # ======================
    # 🏗️ Construir DataFrame de Entrada
    # ======================
    colunas = [
        'genero', 'idade', 'altura', 'peso', 'historico_familiar',
        'consome_alta_calorias_frequente', 'consumo_vegetais',
        'qtde_refeicoes_principais', 'alimentacao_entre_refeicoes', 'fuma',
        'qtde_agua_diaria', 'monitora_calorias', 'freq_atividade_fisica',
        'tempo_uso_dispositivos', 'freq_consumo_alcool', 'meio_transporte_contumaz'
    ]

    entrada = pd.DataFrame([[
        genero_map[genero],
        idade,
        altura,
        peso,
        historico_map[historico],
        alta_caloria_map[consome_calorias],
        consumo_vegetais,
        qtde_refeicoes,
        alimentacao_map[alimentacao],
        fuma_map[fuma],
        qtde_agua,
        monitora_map[monitora_calorias],
        atividade_map[atividade],
        tempo_dispositivo,
        alcool_map[alcool],
        transporte_map[transporte]
    ]], columns=colunas)

    # ======================
    # 🔧 Padronizar Numéricas
    # ======================
    colunas_numericas = ['idade', 'altura', 'peso', 'qtde_refeicoes_principais',
                          'qtde_agua_diaria', 'tempo_uso_dispositivos']

    entrada[colunas_numericas] = scaler.transform(entrada[colunas_numericas])

    # ======================
    # 🔍 Exibir Dados
    # ======================
    st.subheader("🔎 Dados para Predição")
    st.dataframe(entrada)

    # ======================
    # 🚀 Realizar Predição
    # ======================
    if st.button("Realizar Previsão"):
        resultado = modelo.predict(entrada)
        classe = label_encoder.inverse_transform(resultado)[0]
        st.success(f"🔍 Resultado: **{classe.replace('_', ' ')}**")

# ===================================================
# 📊 Painel Analítico
# ===================================================
if aba == "Painel Analítico":
    st.title("📊 Painel Analítico sobre Obesidade")

    st.subheader("Distribuição dos Níveis de Obesidade")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=df, y='nivel_obesidade', color='red', ax=ax)
    ax.set_xlabel('Quantidade')
    ax.set_ylabel('Nível de Obesidade')
    st.pyplot(fig)

    # ✔️ Layout em colunas
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição da Idade")
        fig1, ax1 = plt.subplots(figsize=(5,3))
        sns.histplot(df['idade'], kde=True, color='red', ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.subheader("Distribuição da Altura")
        fig2, ax2 = plt.subplots(figsize=(5,3))
        sns.histplot(df['altura'], kde=True, color='orange', ax=ax2)
        st.pyplot(fig2)

    st.subheader("Distribuição do Peso")
    fig3, ax3 = plt.subplots(figsize=(6,4))
    sns.histplot(df['peso'], kde=True, color='blue', ax=ax3)
    st.pyplot(fig3)

    st.subheader("Tempo em Dispositivos por Nível de Obesidade")
    fig4, ax4 = plt.subplots(figsize=(6,4))
    sns.violinplot(data=df, x='nivel_obesidade', y='tempo_uso_dispositivos', palette='Reds', ax=ax4)
    ax4.set_xlabel('Nível de Obesidade')
    ax4.set_ylabel('Horas por dia')
    ax4.tick_params(axis='x', rotation=45)
    st.pyplot(fig4)

