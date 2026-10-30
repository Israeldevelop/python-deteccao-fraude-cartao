import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


print("Projeto de detecção de fraudes")
print("Pandas instalado com sucesso!")


df = pd.read_csv("data/creditcard.csv")

print(df.head())

# ==============================================================================
# O atributo shape retorna uma tupla com:
# - Primeiro número: quantidade de linhas.
# - Segundo número: quantidade de colunas.
# ==============================================================================

print("Dimensões:", df.shape)

# ==============================================================================
# Verificar os nomes das colunas
# Você verá as colunas Time, V1 até V28, Amount e Class.
# ==============================================================================

print("Colunas:", df.columns)

# ==============================================================================
# Contar as transações de cada tipo
# 0: transações legítimas.
# 1: transações fraudulentas.
# ==============================================================================

print("Quantidade por classe:")
print(df["Class"].value_counts())

# ==============================================================================
# Calcular a porcentagem de fraudes
# Aqui, normalize=True calcula a proporção de cada categoria. 
# Multiplicamos por 100 para obter a porcentagem.
# Por exemplo, se apenas 1 em cada 1.000 transações for fraude, a proporção será de 0,1%.
# ==============================================================================

print("Percentual por classe:")
print(df["Class"].value_counts(normalize=True) * 100)


# ==============================================================================
# Visualizar a distribuição das classes
# O gráfico permite comparar a quantidade de transações legítimas e fraudulentas.
# - value_counts(): conta as ocorrências de cada classe.
# - sort_index(): coloca as classes na ordem 0, 1.
# - plot(kind="bar"): cria um gráfico de barras.
# - plt.title(): define o título.
# - plt.show(): exibe o gráfico.
# ==============================================================================


df["Class"].value_counts().sort_index().plot(
    kind="bar",
    color=["steelblue", "tomato"]
)

plt.title("Distribuição das transações")
plt.xlabel("Classe (0 = legitima, 1 = fraude)")
plt.ylabel("Quantidade de transações")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


#==============================================================================
# Analisar os valores das transações
# O describe() apresenta estatísticas descritivas das colunas numéricas.
# ==============================================================================

print("Estatisticas das transações:")
print(df["Amount"].describe())


# ==============================================================================
# Comparar os valores conforme a classe
# Agrupamos as transações por Class e calculamos a média de Amount.
# ==============================================================================

print("Valor médio por classe: ")
print(df.groupby("Class")["Amount"].mean())

# ==============================================================================
# Criar uma variável com o logaritmo do valor
# O log1p calcula log(1 + Amount), permitindo tratar valores iguais a zero.
# - df["Amount"]: seleciona a coluna com os valores.
# - np.log1p(...): calcula o logaritmo natural de 1 mais o valor.
# - df["Amount_log"] = ...: cria uma nova coluna no DataFrame.
# - df[["Amount", "Amount_log"]]: mostra as duas colunas lado a lado.
# ==============================================================================

df["Amount_log"] = np.log1p(df["Amount"])

print("Comparação entre Amount e Amount_log:")
print(df[["Amount", "Amount_log"]].head())


# ==============================================================================
# Separar as variáveis de entrada (X) e a variável-alvo (y)
# X contém as informações utilizadas pelo modelo.
# y contém a classe que queremos prever: 0 ou 1.
# ==============================================================================


x = df.drop(columns=["Class"])
y = df["Class"]

print("Dimensões de x:", x.shape)
print("Dimensões de y:", y.shape)

print("Dimensões atuais de df:", df.shape)
print("Dimensões de x:", x.shape)
print("Dimensões de y:", y.shape)
print("A coluna Class está em x?", "Class" in x.columns)


# ==============================================================================
# Dividir os dados em treino e teste
# stratify=y preserva aproximadamente a proporção das classes nos dois grupos.
# random_state=42 permite reproduzir a mesma divisão.
# ==============================================================================

# Separar as características (X) e o alvo (y)

x = df.drop(columns=["Class"])
y = df["Class"]

# Separar 20% para teste final

x_temp, x_test, y_temp, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Dividir os 80% restantes em treino (60%) e validação (20%)

x_train, x_val, y_train, y_val = train_test_split(
    x_temp, 
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print("Treino:", x_train.shape)
print("Validação:", x_val.shape)
print("Teste final:", x_test.shape)

print("Fraudes no treino:", (y_train == 1).sum())
print("Fraudes na validação:", (y_val == 1).sum())
print("Fraudes no teste final:", (y_test == 1).sum())

# ==============================================================================
# Padronizar as variáveis numéricas
# O StandardScaler coloca as variáveis em escalas comparáveis.
# fit_transform() aprende os parâmetros usando o treino e transforma os dados.
# transform() aplica os mesmos parâmetros ao conjunto de teste.
# ==============================================================================


scaler = StandardScaler()

# Aprender a escala usando somente os dados de treino
x_train_scaled = scaler.fit_transform(x_train)

# Aplicar a mesma escala na validação e no teste
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

print("Dimensões do treino padronizado:", x_train_scaled.shape)
print("Dimensões da validação padronizada:", x_val_scaled.shape)
print("Dimensões do teste padronizado:", x_test_scaled.shape)


# ==============================================================================
# Treinar o modelo de regressão logística
# class_weight="balanced" atribui pesos maiores à classe minoritária.
# max_iter=1000 permite mais iterações para o algoritmo convergir.
# ==============================================================================


modelo = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

modelo.fit(x_train_scaled, y_train)

print("Modelo treinado com sucesso!")

# ==============================================================================
# Fazer previsões usando os dados de teste
# O modelo tentará prever a classe de cada transação.
# ==============================================================================

y_pred_val = modelo.predict(x_val_scaled)

print("\nPrimeiras 20 previsões da validação:")
print(y_pred_val[:20])


# ==============================================================================
# Avaliar o modelo
# O relatório apresenta precisão, recall e F1-score.
# ==============================================================================


print("\n=== RELATÓRIO DE CLASSIFICAÇÃO — VALIDAÇÃO ===")
print(classification_report(
    y_val, 
    y_pred_val,
    zero_division=0
))

print("Matriz de confusão — Validação:")
print(confusion_matrix(y_val, y_pred_val))


# ==============================================================================
# Obter as probabilidades estimadas de fraude
# A coluna 1 representa a probabilidade estimada da classe fraudulenta.
# ==============================================================================


y_prob = modelo.predict_proba(x_test_scaled)[:,1]


# ==============================================================================
# Obter as probabilidades estimadas de fraude
# A coluna 1 representa a probabilidade estimada da classe fraudulenta.
# ==============================================================================

limiar = 0.20

y_pred_limiar = (y_prob >= limiar).astype(int)

print("Resultados com limiar de", limiar)
print(classification_report(
    y_test,
    y_pred_limiar,
    zero_division=0
))

print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred_limiar))


# ==============================================================================
# Calcular a curva de precisão e recall
# A curva mostra como as métricas mudam conforme o limiar de decisão.
# ==============================================================================


precisoes, recalls, limiares = precision_recall_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8, 5))
plt.plot(recalls, precisoes)

plt.title("Curva de precisão e recall")
plt.xlabel("Recall")
plt.ylabel("Precisão")
plt.grid(True)
plt.tight_layout()
plt.show()


# ==============================================================================
# Calcular e visualizar a curva ROC
# A curva compara a taxa de verdadeiros positivos com a de falsos positivos.
# ==============================================================================


fpr, tpr, limiares_roc = roc_curve(y_test, y_prob)

auc_roc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"Regressão Logistica (AUC = {auc_roc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Referencia aleatoria")

plt.title("Curva ROC")
plt.xlabel("Taxa fr falsos positivos")
plt.ylabel("Taxa de verdadeiros positivos")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("ROC-AUC:", auc_roc)


# ==============================================================================
# Treinar o modelo Random Forest
# class_weight="balanced" considera o desbalanceamento das classes.
# n_estimators=100 define a quantidade de árvores.
# random_state=42 permite reproduzir o experimento.
# ==============================================================================


modelo_rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1

)

modelo_rf.fit(x_train, y_train)

print("Random Forest treinado com sucesso!")


# ==============================================================================
# Avaliar o Random Forest
# ==============================================================================

# Previsões do Random Forest na validação

y_pred_rf_val = modelo_rf.predict(x_val)
y_prob_rf_val = modelo_rf.predict_proba(x_val)[:,1]

print("\n=== RANDOM FOREST — VALIDAÇÃO ===")
print(classification_report(
    y_val,
    y_pred_rf_val,
    zero_division=0
))

print("Matriz de confusão - Random Forest:")
print(confusion_matrix(y_val, y_pred_rf_val))



# ==============================================================================
# Ajustar o limiar do Random Forest
# Um limiar menor pode aumentar o recall, mas também os falsos positivos.
# ==============================================================================


limiar_rf = 0.30

y_pred_rf_limiar = (y_pred_rf_val >= limiar_rf).astype(int)

print("Random Forest com limiar:", limiar_rf)

print(classification_report(
    y_val,
    y_pred_rf_limiar,
    zero_division=0

))

print("Matriz de confusão:")
print(confusion_matrix(y_val, y_pred_rf_limiar))


# ==============================================================================
# === IMPORTÂNCIA DAS CARACTERÍSTICAS ==========================================
# ==============================================================================

importancias = modelo_rf.feature_importances_

ranking = pd.DataFrame({
    "Caracteristica": x_train.columns,
    "Importancia": importancias
})

ranking = ranking.sort_values(
    by="Importancia",
    ascending=False
)

print("\n=== 10 CARACTERÍSTICAS MAIS IMPORTANTES ===")
print(ranking.head(10).to_string(index=False))


# ==============================================================================
# Gráfico das 10 características mais importantes ==============================
# ==============================================================================

top_10 = ranking.head(10).sort_values(
    by="Importancia"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_10["Caracteristica"],
    top_10["Importancia"]
)

plt.title("10 caracteristicas mais importantes - Random Forest")
plt.xlabel("Importancia")
plt.ylabel("Caracteristica")
plt.tight_layout()
plt.show()

print("\n=== V10 POR CLASSE ===")

print(
    df.groupby("Class")["V10"]
    .agg(["mean", "median", "std"])
)


# Comparar V14 entre transações normais e fraudulentas

print("\n=== V14 POR CLASSE ===")

print(
    df.groupby("Class")["V14"].agg(["mean", "median", "std"])
)

# Comparar a distribuição da V14 por classe

plt.figure(figsize=(10, 6))

for classe, nome in [(0, "Normais"), (1, "Fraudes")]:
    plt.hist(
        df.loc[df["Class"] == classe, "V14"],
        bins=50,
        alpha=0.6,
        density=True,
        label=nome
    )

plt.title("Distribuição da V14: transações normais x fraudes")
plt.xlabel("Valor da V10")
plt.ylabel("Densidade")
plt.legend()
plt.tight_layout()
plt.show()


# ==============================================================================
# Treinar o XGBoost
# scale_pos_weight aumenta o peso da classe fraude durante o treinamento.
# O valor inicial é calculado pela proporção entre classes no treino.
# ==============================================================================


quantidade_normais = (y_train == 0).sum()
quantidade_fraudes  = (y_train == 1).sum()

peso_fraude = quantidade_normais / quantidade_fraudes

modelo_xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight =peso_fraude,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

modelo_xgb.fit(x_train, y_train)
y_pred_xgb_val = modelo_xgb.predict(x_val)

print("XGBoost treinado com sucesso!")

# ==============================================================================
# Avaliar o XGBoost
# ==============================================================================

y_pred_xgb_val = modelo_xgb.predict(x_val)
y_prob_xgb_val = modelo_xgb.predict_proba(x_val)[:,1]

matriz_xgb_val = confusion_matrix(y_val, y_pred_xgb_val)


# Criar a matriz de confusão
print("Matriz de confusão - XGBoost:")
print(matriz_xgb_val)

print(
    "ROC-AUC - XGBoost:",
    roc_auc_score(y_val, y_prob_xgb_val)
)



# ==============================================================================
# COMPARAÇÃO DOS MODELOS 
# ==============================================================================


# Modelo - Regressão Logística
relatorio_log = classification_report(
    y_val, y_pred_val, output_dict=True
)

# Modelo - Random Forest
relatorio_rf = classification_report(
    y_val, y_pred_rf_val, output_dict=True
)

# Modelo - XGBoost
relatorio_xgb = classification_report(
    y_val, y_pred_xgb_val, output_dict=True
)


comparacao = pd.DataFrame({
    "Modelo": [
        "Regressão Logistica",
        "Random Forest",
        "XGBoost"
    ],
    "Precisão": [
        relatorio_log["1"]["precision"],
        relatorio_rf["1"]["precision"],
        relatorio_xgb["1"]["precision"]
    ],
    "Recall": [
        relatorio_log["1"]["recall"],
        relatorio_rf["1"]["recall"],
        relatorio_xgb["1"]["recall"]
    ],
    "F1-score": [
        relatorio_log["1"]["f1-score"],
        relatorio_rf["1"]["f1-score"],
        relatorio_xgb["1"]["f1-score"]
    ]
})

print("\n=== COMPARAÇÃO DOS MODELOS ===")
print(comparacao.round(3).to_string(index=False))

# ============================================================
# AVALIAÇÃO FINAL — RANDOM FOREST
# ============================================================

# Previsões no conjunto de teste
y_pred_rf_final = modelo_rf.predict(x_test)

print("\n=== RESULTADO FINAL — RANDOM FOREST ===")

print(classification_report(
    y_test,
    y_pred_rf_final,
    zero_division=0
))

print("Matriz de confusão - Teste final:")
print(confusion_matrix(y_test, y_pred_rf_final))

# ROC-AUC

y_prob_rf_final = modelo_rf.predict_proba(x_test)[:,1]

print(
    "ROC-AUC final:",
    roc_auc_score(y_test, y_prob_rf_final)
)


# ==============================================================================
# === GRÁFICO DE COMPARAÇÃO DOS MODELOS ===
# ==============================================================================


comparacao_plot = comparacao.set_index("Modelo")

comparacao_plot.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Comparação dos modelos de detecção de fraudes")
plt.ylabel("Pontuação")
plt.xlabel("Modelo")
plt.ylim(0,1)
plt.xticks(rotation=0)
plt.legend(title="Métrica")
for container in plt.gca().containers:
    plt.gca().bar_label(container, fmt="%.2f")
plt.tight_layout()
plt.show()
















