
import streamlit as st
import numpy as np
import joblib

# Carregando os artefatos
modelo = joblib.load('modelo_obesidade.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder_target.pkl')

st.title("🔬 Sistema Preditivo de Obesidade")
st.write("Preencha os dados do paciente para prever o nível de obesidade.")

# Entrada de dados
genero = st.selectbox("Gênero", ["Male", "Female"])
idade = st.slider("Idade", 10, 100, 25)
altura = st.slider("Altura (em metros)", 1.0, 2.2, 1.70)
peso = st.slider("Peso (kg)", 30.0, 200.0, 70.0)
imc = peso / (altura ** 2)

historico_familiar = st.selectbox("Histórico Familiar de Obesidade", ["yes", "no"])
consome_alta_calorias = st.selectbox("Consome alimentos calóricos com frequência?", ["yes", "no"])
consome_vegetais = st.slider("Consome vegetais (de 1 a 3)", 1.0, 3.0, 2.0)
qtde_refeicoes = st.slider("Número de refeições principais por dia", 1.0, 4.0, 3.0)
consumo_entre_refeicoes = st.selectbox("Come entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
fuma = st.selectbox("Fuma?", ["yes", "no"])
qtde_agua = st.slider("Litros de água por dia", 1.0, 3.0, 2.0)
monitora_calorias = st.selectbox("Monitora as calorias ingeridas?", ["yes", "no"])
freq_atividade_fisica = st.slider("Atividade física (horas por semana)", 0.0, 3.0, 1.0)
tempo_uso_dispositivos = st.slider("Horas por dia usando celular, TV, computador...", 0.0, 3.0, 1.0)
freq_consumo_alcool = st.selectbox("Consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
meio_transporte = st.selectbox("Meio de transporte mais usado", [
    "Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

# Codificação manual igual ao treinamento
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

# Criar vetor de entrada com os dados tratados
entrada = np.array([[
    mapeamentos['genero'][genero],
    idade,
    mapeamentos['historico_familiar'][historico_familiar],
    mapeamentos['consome_alta_calorias'][consome_alta_calorias],
    consome_vegetais,
    qtde_refeicoes,
    mapeamentos['consumo_entre_refeicoes'][consumo_entre_refeicoes],
    mapeamentos['fuma'][fuma],
    qtde_agua,
    mapeamentos['monitora_calorias'][monitora_calorias],
    freq_atividade_fisica,
    tempo_uso_dispositivos,
    mapeamentos['freq_consumo_alcool'][freq_consumo_alcool],
    mapeamentos['meio_transporte'][meio_transporte],
    imc
]])

entrada_escalada = scaler.transform(entrada)

# Botão para prever
if st.button("Realizar Previsão"):
    resultado = modelo.predict(entrada_escalada)
    classe = label_encoder.inverse_transform(resultado)[0]
    st.success(f"**Previsão:** {classe.replace('_', ' ')}")
