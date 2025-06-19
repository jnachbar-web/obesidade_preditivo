
# ============================
# 📦 Importações
# ============================
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ============================
# 📂 Carregar e preparar os dados
# ============================
df = pd.read_csv('Obesity.csv')

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

# ============================
# 🎯 Pré-processamento
# ============================
df_modelo = df.copy()

map_atividade = {'Nunca': 0, 'Pouquíssima': 1, 'Moderada': 2, 'Frequente': 3}
map_alimentacao = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
map_alcool = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}

df_modelo['freq_atividade_fisica'] = df_modelo['freq_atividade_fisica'].map(map_atividade)
df_modelo['alimentacao_entre_refeicoes'] = df_modelo['alimentacao_entre_refeicoes'].map(map_alimentacao)
df_modelo['freq_consumo_alcool'] = df_modelo['freq_consumo_alcool'].map(map_alcool)

colunas_categoricas = ['genero', 'historico_familiar', 'consome_alta_calorias_frequente',
                        'fuma', 'monitora_calorias', 'meio_transporte_contumaz']

le = LabelEncoder()
for col in colunas_categoricas:
    df_modelo[col] = le.fit_transform(df_modelo[col])

le_target = LabelEncoder()
df_modelo['nivel_obesidade'] = le_target.fit_transform(df_modelo['nivel_obesidade'])

colunas_numericas = ['idade', 'altura', 'peso', 'qtde_refeicoes_principais',
                      'qtde_agua_diaria', 'tempo_uso_dispositivos']

scaler = StandardScaler()
df_modelo[colunas_numericas] = scaler.fit_transform(df_modelo[colunas_numericas])

X = df_modelo.drop('nivel_obesidade', axis=1)
y = df_modelo['nivel_obesidade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

modelo = GradientBoostingClassifier()
modelo.fit(X_train, y_train)

# ============================
# 🎨 Sidebar
# ============================
st.sidebar.title("Menu")
aba = st.sidebar.radio("Escolha uma aba:", ["Sistema Preditivo", "Painel Analítico"])

# ===================================================
# 🔬 Sistema Preditivo
# ===================================================
if aba == "Sistema Preditivo":
    st.title("🔬 Sistema Preditivo de Obesidade")

    genero = st.selectbox('Gênero', df['genero'].unique())
    idade = st.slider('Idade', 10, 100, 30)
    altura = st.slider('Altura (em metros)', 1.0, 2.2, 1.70)
    peso = st.slider('Peso (kg)', 30.0, 200.0, 70.0)
    historico = st.selectbox('Histórico Familiar de Obesidade', df['historico_familiar'].unique())
    consome_calorias = st.selectbox('Consome alimentos calóricos frequentemente?', df['consome_alta_calorias_frequente'].unique())
    consumo_vegetais = st.selectbox('Consumo de vegetais (1-3)', [1,2,3])
    qtde_refeicoes = st.slider('Refeições principais por dia', 1, 4, 3)
    alimentacao = st.selectbox('Come entre as refeições?', df['alimentacao_entre_refeicoes'].unique())
    fuma = st.selectbox('Fuma?', df['fuma'].unique())
    qtde_agua = st.slider('Litros de água por dia', 1.0, 3.0, 2.0)
    monitora_calorias = st.selectbox('Monitora calorias?', df['monitora_calorias'].unique())
    atividade = st.selectbox('Frequência de atividade física', ['Nunca','Pouquíssima','Moderada','Frequente'])
    tempo_dispositivo = st.slider('Horas de uso de dispositivos por dia', 0.0, 5.0, 2.0)
    alcool = st.selectbox('Frequência de consumo de álcool', df['freq_consumo_alcool'].unique())
    transporte = st.selectbox('Meio de transporte mais usado', df['meio_transporte_contumaz'].unique())

    entrada = pd.DataFrame([[
        genero, idade, altura, peso, historico, consome_calorias, consumo_vegetais,
        qtde_refeicoes, alimentacao, fuma, qtde_agua, monitora_calorias,
        atividade, tempo_dispositivo, alcool, transporte
    ]], columns=X.columns)

    entrada['freq_atividade_fisica'] = map_atividade[atividade]
    entrada['alimentacao_entre_refeicoes'] = map_alimentacao[alimentacao]
    entrada['freq_consumo_alcool'] = map_alcool[alcool]

    for col in colunas_categoricas:
        entrada[col] = le.transform(entrada[col])

    entrada[colunas_numericas] = scaler.transform(entrada[colunas_numericas])

    if st.button("Realizar Previsão"):
        resultado = modelo.predict(entrada)
        classe = le_target.inverse_transform(resultado)[0]
        st.success(f"🔍 Resultado: **{classe.replace('_',' ')}**")

# ===================================================
# 📊 Painel Analítico
# ===================================================
if aba == "Painel Analítico":
    st.title("📊 Painel Analítico sobre Obesidade")

    # ✔️ Gráfico 1 - Distribuição dos Níveis de Obesidade
    st.subheader("Distribuição dos Níveis de Obesidade")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=df, y='nivel_obesidade', color='red', ax=ax)
    st.pyplot(fig)

    # ✔️ Gráfico 2 - Idade
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribuição da Idade")
        fig1, ax1 = plt.subplots(figsize=(5,3))
        sns.histplot(df['idade'], kde=True, color='red', ax=ax1)
        st.pyplot(fig1)

    # ✔️ Gráfico 3 - Altura
    with col2:
        st.subheader("Distribuição da Altura")
        fig2, ax2 = plt.subplots(figsize=(5,3))
        sns.histplot(df['altura'], kde=True, color='orange', ax=ax2)
        st.pyplot(fig2)

    # ✔️ Gráfico 4 - Peso
    st.subheader("Distribuição do Peso")
    fig3, ax3 = plt.subplots(figsize=(6,4))
    sns.histplot(df['peso'], kde=True, color='blue', ax=ax3)
    st.pyplot(fig3)

    # ✔️ Gráfico 5 - Tempo em Dispositivos
    st.subheader("Tempo em Dispositivos por Nível de Obesidade")
    fig4, ax4 = plt.subplots(figsize=(6,4))
    sns.violinplot(data=df, x='nivel_obesidade', y='tempo_uso_dispositivos', palette='Reds', ax=ax4)
    ax4.tick_params(axis='x', rotation=45)
    st.pyplot(fig4)

