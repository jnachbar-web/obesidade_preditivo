
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Carregar os dados já tratados (o mesmo CSV usado no modelo)
df = pd.read_csv('Obesity.csv')

df.rename(columns={'Gender':'genero',
                   'Age':'idade',
                   'Height':'altura',
                   'Weight':'peso',
                   'family_history':'historico_familiar',
                   'FAVC':'consome_alta_calorias',
                   'FCVC':'consome_vegetais',
                   'NCP':'qtde_refeicoes',
                   'CAEC':'consumo_entre_refeicoes',
                   'SMOKE':'fuma',
                   'CH2O':'qtde_agua',
                   'SCC':'monitora_calorias',
                   'FAF':'freq_atividade_fisica',
                   'TUE':'tempo_uso_dispositivos',
                   'CALC':'freq_consumo_alcool',
                   'MTRANS':'meio_transporte',
                   'Obesity':'nivel_obesidade'}, inplace=True)

# Recalcular o IMC (caso não esteja presente no CSV)
df['imc'] = df['peso'] / (df['altura'] ** 2)

# Carregar modelo e pré-processadores
modelo = joblib.load('modelo_obesidade.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder_target.pkl')

st.sidebar.title("Menu")
aba = st.sidebar.radio("Escolha uma aba:", ["Sistema Preditivo", "Painel Analítico"])

if aba == "Painel Analítico":
    st.title("Painel Analítico sobre Obesidade")

    st.markdown("### Distribuição dos Níveis de Obesidade")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='nivel_obesidade',
                  order=[
                      'Insufficient_Weight',
                      'Normal_Weight',
                      'Overweight_Level_I',
                      'Overweight_Level_II',
                      'Obesity_Type_I',
                      'Obesity_Type_II',
                      'Obesity_Type_III'
                  ],
                  color='red', ax=ax1)
    ax1.set_xlabel("Nível de Obesidade", fontsize=9)
    ax1.set_ylabel("Quantidade", fontsize=9)
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    st.pyplot(fig1)

    st.markdown("### Distribuição do IMC por Nível de Obesidade")
    fig2, ax2 = plt.subplots()
    sns.violinplot(data=df, x='nivel_obesidade', y='imc',
                   order=[
                       'Insufficient_Weight',
                       'Normal_Weight',
                       'Overweight_Level_I',
                       'Overweight_Level_II',
                       'Obesity_Type_I',
                       'Obesity_Type_II',
                       'Obesity_Type_III'
                   ],
                   palette='Reds', ax=ax2)
    ax2.set_xlabel("Nível de Obesidade", fontsize=9)
    ax2.set_ylabel("IMC (kg/m²)", fontsize=9)
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    st.pyplot(fig2)

    st.markdown("### Distribuição de Variáveis Numéricas")
    fig3, axs = plt.subplots(1, 3, figsize=(18, 4))
    sns.histplot(df['idade'], kde=True, bins=20, color='red', ax=axs[0])
    axs[0].set_title('Idade'); axs[0].set_xlabel('Idade'); axs[0].set_ylabel('Frequência')

    sns.histplot(df['altura'], kde=True, bins=20, color='orange', ax=axs[1])
    axs[1].set_title('Altura'); axs[1].set_xlabel('Altura'); axs[1].set_ylabel('Frequência')

    sns.histplot(df['peso'], kde=True, bins=20, color='blue', ax=axs[2])
    axs[2].set_title('Peso'); axs[2].set_xlabel('Peso'); axs[2].set_ylabel('Frequência')

    st.pyplot(fig3)

if aba == "Sistema Preditivo":
    st.title("🔬 Sistema Preditivo de Obesidade")
    st.write("Preencha os dados do paciente para prever o nível de obesidade.")

    # --- Entradas categóricas ---
    genero = st.selectbox("Gênero", ["Male", "Female"])
    historico_familiar = st.selectbox("Histórico Familiar de Obesidade", ["yes", "no"])
    consome_alta_calorias = st.selectbox("Consome alimentos calóricos com frequência?", ["yes", "no"])
    consumo_entre_refeicoes = st.selectbox("Come entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
    fuma = st.selectbox("Fuma?", ["yes", "no"])
    monitora_calorias = st.selectbox("Monitora as calorias ingeridas?", ["yes", "no"])
    freq_consumo_alcool = st.selectbox("Frequência de consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
    meio_transporte = st.selectbox("Meio de transporte mais usado", [
        "Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

    # --- Entradas numéricas ---
    idade = st.slider("Idade", 10, 100, 25)
    altura = st.slider("Altura (em metros)", 1.0, 2.2, 1.70)
    peso = st.slider("Peso (kg)", 30.0, 200.0, 70.0)
    imc = peso / (altura ** 2)
    consome_vegetais = st.slider("Consome vegetais (1 a 3)", 1.0, 3.0, 2.0)
    qtde_refeicoes = st.slider("Refeições principais por dia", 1.0, 4.0, 3.0)
    qtde_agua = st.slider("Litros de água por dia", 1.0, 3.0, 2.0)
    freq_atividade_fisica = st.slider("Horas de atividade física por semana", 0.0, 3.0, 1.0)
    tempo_uso_dispositivos = st.slider("Horas com dispositivos eletrônicos por dia", 0.0, 3.0, 1.0)

    # --- Mapeamento das variáveis categóricas ---
    mapeamentos = {
        'genero': {'Male': 1, 'Female': 0},
        'historico_familiar': {'yes': 1, 'no': 0},
        'consome_alta_calorias': {'yes': 1, 'no': 0},
        'consumo_entre_refeicoes': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
        'fuma': {'yes': 1, 'no': 0},
        'monitora_calorias': {'yes': 1, 'no': 0},
        'freq_consumo_alcool': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
        'meio_transporte': {
            'Automobile': 0, 'Bike': 1, 'Motorbike': 2,
            'Public_Transportation': 3, 'Walking': 4
        }
    }

    # --- Vetor categórico codificado ---
    entrada_categorica = [
        mapeamentos['genero'][genero],
        mapeamentos['historico_familiar'][historico_familiar],
        mapeamentos['consome_alta_calorias'][consome_alta_calorias],
        mapeamentos['consumo_entre_refeicoes'][consumo_entre_refeicoes],
        mapeamentos['fuma'][fuma],
        mapeamentos['monitora_calorias'][monitora_calorias],
        mapeamentos['freq_consumo_alcool'][freq_consumo_alcool],
        mapeamentos['meio_transporte'][meio_transporte]
    ]

    # --- Vetor numérico (será padronizado) ---
    entrada_numerica_escalada = scaler.transform([[ 
        idade, altura, peso, consome_vegetais,
        qtde_refeicoes, qtde_agua,
        freq_atividade_fisica, tempo_uso_dispositivos,
        imc
    ]])

    entrada_numerica_final = np.delete(entrada_numerica_escalada, [1, 2], axis=1)

    # --- Combina as duas partes (categorias + numericas padronizadas) ---
    entrada_final = np.hstack([entrada_categorica, entrada_numerica_final[0]])

    # --- Realiza a previsão ---
    if st.button("Realizar Previsão"):
        resultado = modelo.predict(np.array([entrada_final]))
        classe = label_encoder.inverse_transform(resultado)[0]
        st.success(f"**Previsão:** {classe.replace('_', ' ')}")
