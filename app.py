
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Sistema de Predição de Obesidade', layout='wide')

# ✔️ Carregar artefatos
modelo = joblib.load('modelo_obesidade.pkl')
scaler = joblib.load('scaler.pkl')
le_target = joblib.load('label_encoder_target.pkl')
le_categoricas = joblib.load('label_encoders_categoricas.pkl')

# ✔️ Carregar dados para painel analítico
df = pd.read_csv('Obesity.csv')

# ✔️ Mapeamento de nível de obesidade
mapa_obesidade = {
    'Insufficient_Weight': 'Abaixo do Peso',
    'Normal_Weight': 'Peso Normal',
    'Overweight_Level_I': 'Sobrepeso - Nível I',
    'Overweight_Level_II': 'Sobrepeso - Nível II',
    'Obesity_Type_I': 'Obesidade - Nível I',
    'Obesity_Type_II': 'Obesidade - Nível II',
    'Obesity_Type_III': 'Obesidade - Nível III'
}
df['nivel_obesidade'] = df['nivel_obesidade'].map(mapa_obesidade)

# ✔️ Menu lateral
st.sidebar.title('Menu')
aba = st.sidebar.radio('Selecione a aba:', ['Sistema Preditivo', 'Painel Analítico'])

# ✔️ Aba: Sistema Preditivo
if aba == 'Sistema Preditivo':
    st.title('🔬 Sistema Preditivo de Obesidade')
    st.markdown('Preencha os dados do paciente para prever o nível de obesidade.')

    # ✔️ Entradas do usuário
    col1, col2 = st.columns(2)

    with col1:
        genero = st.selectbox('Gênero', le_categoricas['genero'].classes_)
        idade = st.slider('Idade', 10, 100, 30)
        altura = st.number_input('Altura (em metros)', 1.0, 2.5, 1.70)
        peso = st.number_input('Peso (kg)', 30.0, 200.0, 70.0)
        historico_familiar = st.selectbox('Histórico Familiar de Obesidade', le_categoricas['historico_familiar'].classes_)

    with col2:
        consome_alta_calorias = st.selectbox('Consome alimentos calóricos frequentemente?', le_categoricas['consome_alta_calorias_frequente'].classes_)
        consumo_vegetais = st.selectbox('Consumo de vegetais', le_categoricas['consumo_vegetais'].classes_)
        qtde_refeicoes = st.slider('Refeições principais por dia', 1, 4, 3)
        alimentacao_entre_refeicoes = st.selectbox('Lanches entre refeições', le_categoricas['alimentacao_entre_refeicoes'].classes_)
        fuma = st.selectbox('Fuma?', le_categoricas['fuma'].classes_)
        qtde_agua = st.slider('Consumo de água (L/dia)', 1, 5, 2)
        monitora_calorias = st.selectbox('Monitora calorias?', le_categoricas['monitora_calorias'].classes_)
        atividade_fisica = st.slider('Atividade física (horas/semana)', 0, 20, 2)
        tempo_dispositivos = st.slider('Tempo com dispositivos (horas/dia)', 0, 24, 4)
        consumo_alcool = st.selectbox('Frequência de consumo de álcool', le_categoricas['freq_consumo_alcool'].classes_)
        meio_transporte = st.selectbox('Meio de transporte', le_categoricas['meio_transporte_contumaz'].classes_)

    # ✔️ Processamento dos dados de entrada
    entrada_categorica = [
        le_categoricas['genero'].transform([genero])[0],
        le_categoricas['historico_familiar'].transform([historico_familiar])[0],
        le_categoricas['consome_alta_calorias_frequente'].transform([consome_alta_calorias])[0],
        le_categoricas['consumo_vegetais'].transform([consumo_vegetais])[0],
        le_categoricas['alimentacao_entre_refeicoes'].transform([alimentacao_entre_refeicoes])[0],
        le_categoricas['fuma'].transform([fuma])[0],
        le_categoricas['monitora_calorias'].transform([monitora_calorias])[0],
        le_categoricas['freq_consumo_alcool'].transform([consumo_alcool])[0],
        le_categoricas['meio_transporte_contumaz'].transform([meio_transporte])[0]
    ]

    entrada_numerica = scaler.transform([[
        idade, altura, peso, qtde_refeicoes,
        qtde_agua, atividade_fisica, tempo_dispositivos
    ]])[0]

    entrada_final = np.hstack([entrada_categorica, entrada_numerica])

    # ✔️ Previsão
    if st.button('Realizar Previsão'):
        resultado = modelo.predict([entrada_final])
        classe = le_target.inverse_transform(resultado)[0]
        st.success(f'🧠 Previsão: **{classe}**')

# ✔️ Aba: Painel Analítico
if aba == 'Painel Analítico':
    st.title('📊 Painel Analítico sobre Obesidade')

    st.markdown('### Distribuição dos Níveis de Obesidade')
    fig1, ax = plt.subplots(figsize=(8,4))
    sns.countplot(data=df, y='nivel_obesidade', color='red', ax=ax)
    plt.xlabel('Quantidade')
    plt.ylabel('Nível de Obesidade')
    st.pyplot(fig1)

    st.markdown('### Distribuição de Idade')
    fig2, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df['idade'], kde=True, color='red', bins=20, ax=ax)
    plt.xlabel('Idade')
    plt.ylabel('Frequência')
    st.pyplot(fig2)

    st.markdown('### Distribuição de Peso')
    fig3, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df['peso'], kde=True, color='red', bins=20, ax=ax)
    plt.xlabel('Peso')
    plt.ylabel('Frequência')
    st.pyplot(fig3)

    st.markdown('### Distribuição do IMC por Nível de Obesidade')
    df['imc'] = df['peso'] / (df['altura'] ** 2)
    fig4, ax = plt.subplots(figsize=(8,4))
    sns.violinplot(data=df, x='nivel_obesidade', y='imc', palette='Reds', ax=ax)
    plt.xlabel('Nível de Obesidade')
    plt.ylabel('IMC (kg/m²)')
    plt.xticks(rotation=45)
    st.pyplot(fig4)
