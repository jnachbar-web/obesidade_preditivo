
# 🔥 Importações
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# 🔥 Carregar modelo e artefatos
modelo = joblib.load('modelo_obesidade.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder_target.pkl')

# 🔥 Carregar base de dados para o painel
df = pd.read_csv('Obesity.csv')

# 🔥 Renomear colunas
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

# 🔥 Sidebar de navegação
st.sidebar.title("Menu")
aba = st.sidebar.radio("Escolha uma aba:", ["Sistema Preditivo", "Painel Analítico"])

# ===================================================================
# 🧠 🔍 Aba — Sistema Preditivo
# ===================================================================
if aba == "Sistema Preditivo":
    st.title("🔬 Sistema Preditivo de Obesidade")

    st.subheader("📄 Informe os dados do paciente:")

    # 🔥 Mapeamentos para exibir em português
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

    # 🔥 Inputs do usuário
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

    # 🔥 Vetor categórico
    entrada_categorica = [
        genero_map[genero],
        historico_map[historico],
        alta_caloria_map[consome_calorias],
        alimentacao_map[alimentacao],
        fuma_map[fuma],
        monitora_map[monitora_calorias],
        alcool_map[alcool],
        transporte_map[transporte],
        atividade_map[atividade],
        consumo_vegetais
    ]

    # 🔥 Vetor numérico
    entrada_numerica = np.array([[
        idade, altura, peso, qtde_refeicoes,
        qtde_agua, tempo_dispositivo
    ]])

    # 🔥 Escalonamento
    entrada_numerica_escalada = scaler.transform(entrada_numerica)

    # 🔥 Combinar entrada final
    entrada_final = np.hstack([entrada_categorica, entrada_numerica_escalada[0]])

    # 🔥 Checagem de consistência
    if len(entrada_final) != modelo.n_features_in_:
        st.error(f"❌ Erro na quantidade de variáveis. Esperado {modelo.n_features_in_}, recebido {len(entrada_final)}.")
        st.stop()

    if st.button("Realizar Previsão"):
        resultado = modelo.predict([entrada_final])
        classe = label_encoder.inverse_transform(resultado)[0]
        st.success(f"🔍 Resultado: **{classe.replace('_', ' ')}**")

# ===================================================================
# 📊 🔍 Aba — Painel Analítico
# ===================================================================
if aba == "Painel Analítico":
    st.title("📊 Painel Analítico sobre Obesidade")

    # ✔️ Layout em colunas
    st.subheader("Distribuição dos Níveis de Obesidade")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=df, y='nivel_obesidade', color='red', ax=ax)
    ax.set_xlabel('Quantidade')
    ax.set_ylabel('Nível de Obesidade')
    st.pyplot(fig)

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

