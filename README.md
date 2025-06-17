# Sistema Preditivo e Analítico de Obesidade

Este projeto tem como objetivo auxiliar profissionais da saúde na **análise e predição dos níveis de obesidade**, além de oferecer **painel analítico interativo** com insights a respeito do perfil populacional.

O sistema foi desenvolvido com **Machine Learning, Streamlit e Python**, objetivando um modelo preditivo robusto (Gradient Boosting) e uma interface amigável.

---

## Funcionalidades

- **Sistema Preditivo:** Permite prever o nível de obesidade de uma pessoa com base em características pessoais, hábitos alimentares, prática de atividades físicas e estilo de vida.
  
- **Painel Analítico:** Dashboards interativos para análise exploratória dos dados, identificação de padrões e extração de insights para apoio à decisão.

---

## Modelagem

- ✔️ Pipeline completo de Machine Learning.  
- ✔️ Pré-processamento com escalonamento de variáveis numéricas.  
- ✔️ Codificação de variáveis categóricas.  
- ✔️ Modelo treinado com **Gradient Boosting**, garantindo alta acurácia.  
- ✔️ Artefatos (.pkl) prontos para deploy.

---

## 🗂️ Estrutura dos Arquivos

```plaintext
├── app.py                         # Aplicação em Streamlit
├── Obesity.csv                    # Base de dados para o painel analítico
├── modelo_obesidade.pkl            # Modelo treinado
├── scaler.pkl                      # Escalonador das variáveis numéricas
├── label_encoder_target.pkl        # Label Encoder do target (nível de obesidade)
├── requirements.txt                # Dependências do projeto
└── README.md                       # Documentação
