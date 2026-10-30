# 💳 Projeto de Detecção de Fraudes em Cartões de Crédito

## 📌 Sobre o projeto

Este projeto tem como objetivo desenvolver modelos de Machine Learning capazes de identificar transações fraudulentas em cartões de crédito.

Foi utilizado um conjunto de dados com 284.807 transações, contendo registros de operações normais e fraudulentas.

O principal desafio é o desequilíbrio entre as classes: aproximadamente 99,83% das transações são normais e apenas 0,17% são fraudulentas.

## 🛠️ Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost

## 🔎 Etapas do projeto

1. Importação e exploração dos dados.
2. Análise da distribuição entre transações normais e fraudulentas.
3. Análise estatística dos valores das transações.
4. Transformação da variável Amount com `Amount_log`.
5. Separação dos dados em treino, validação e teste.
6. Padronização dos dados para a Regressão Logística.
7. Treinamento dos modelos de Machine Learning.
8. Avaliação e comparação dos resultados.
9. Análise da importância das características.

## 🤖 Modelos utilizados

- Regressão Logística
- Random Forest
- XGBoost

## 📊 Resultados finais — Random Forest

O Random Forest apresentou os seguintes resultados no conjunto de teste:

| Métrica | Resultado |
|---|---:|
| Precisão | 90% |
| Recall | 81,6% |
| F1-score | 85,6% |
| ROC-AUC | 0,9519 |

### Matriz de confusão

- Verdadeiros negativos: 56.855
- Falsos positivos: 9
- Falsos negativos: 18
- Verdadeiros positivos: 80

O modelo identificou 80 das 98 fraudes existentes no conjunto de teste.

## 🔍 Características importantes

Na análise de importância das características do Random Forest, destacaram-se:

- V14
- V10
- V12
- V11
- V4

A V14 apresentou a maior importância no modelo, seguida pela V10.

## 📝 Conclusão

O projeto demonstrou a aplicação de técnicas de análise de dados e Machine Learning para detecção de fraudes em cartões de crédito.

A comparação entre os três algoritmos permitiu observar diferentes comportamentos em relação à precisão e ao recall.

O Random Forest apresentou o maior F1-score na validação e obteve resultados consistentes na avaliação final.

Este projeto foi desenvolvido como prática de estudos em Python, análise de dados e Machine Learning.