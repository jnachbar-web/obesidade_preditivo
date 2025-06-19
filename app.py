
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Painel Analítico - Predição de Obesidade", layout="wide")

# ✔️ Carregar os dados
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
    'MTRANS': 'meio_transporte_contumaz'
}, inplace=True)

# ✔️ Conferir e renomear a coluna alvo
if 'Obesity' in df.columns:
    df.rename(columns={'Obesity': 'nivel_obesidade'}, inplace=True)
elif 'NObesity' in df.columns:
    df.rename(columns={'NObesity': 'nivel_obesidade'}, inplace=True)
else:
    st.error("❌ Erro: A coluna de nível de obesidade não foi encontrada no CSV.")
    st.stop()

# ✔️ Mapear os níveis de obesidade
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

# ✔️ Calcular IMC
df['imc'] = df['peso'] / (df['altura'] ** 2)

# ========== Streamlit Layout ==========
st.title('Painel Analítico - Predição de Obesidade')

# ✔️ Gráfico 1 - Distribuição dos Níveis de Obesidade
st.subheader('Distribuição dos Níveis de Obesidade')
fig1, ax1 = plt.subplots(figsize=(6,4))
sns.countplot(data=df, y='nivel_obesidade', color='red', order=ordem_obesidade, ax=ax1)
ax1.set_xlabel('Quantidade')
ax1.set_ylabel('Nível de Obesidade')
st.pyplot(fig1)

# ✔️ Gráfico 2 - Distribuição do IMC
st.subheader('Distribuição do IMC por Nível de Obesidade')
fig2, ax2 = plt.subplots(figsize=(7,4))
sns.violinplot(data=df, x='nivel_obesidade', y='imc', order=ordem_obesidade, palette='Reds', ax=ax2)
ax2.set_xlabel('Nível de Obesidade')
ax2.set_ylabel('IMC (kg/m²)')
ax2.tick_params(axis='x', rotation=45)
st.pyplot(fig2)

# ✔️ Gráfico 3 - Distribuição das Variáveis Numéricas
st.subheader('Distribuição das Variáveis Numéricas')
fig3, axs = plt.subplots(1, 3, figsize=(15,4))
sns.histplot(df['idade'], kde=True, bins=20, color='red', ax=axs[0])
axs[0].set_title('Idade')

sns.histplot(df['altura'], kde=True, bins=20, color='orange', ax=axs[1])
axs[1].set_title('Altura')

sns.histplot(df['peso'], kde=True, bins=20, color='blue', ax=axs[2])
axs[2].set_title('Peso')

for ax in axs:
    ax.set_ylabel('Frequência')

st.pyplot(fig3)
