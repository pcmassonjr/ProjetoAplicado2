# -*- coding: utf-8 -*-

# Import dos pacotes
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Opcoes para visualizar todas as linhas e colunas ao exibir a base de dados
#pd.set_option("display.max_columns", None)
#pd.set_option("display.max_rows", None)

df_health = pd.read_csv('https://archive.ics.uci.edu/static/public/887/data.csv', sep=",", encoding='latin-1')
#print(df_health.head)
print(df_health.dtypes)

# Verificacao de Dados ausentes
# Proporção de ausentes em cada atributo:
df_proporcao=df_health.isnull().sum()/len(df_health)
#print('\nLinhas antes de dropna: ', len(df_health))
#df=df_health.dropna()
#print('\nLinhas depois de dropna: ', len(df_health))
# Criacao de um DataFrame pandas com o resultado da operacao isnull()
df_isnull = df_health.isnull()
# Contabilizacao dos dados ausentes por atributo
print(df_health.isnull().sum())

# Verificacao de Linhas duplicadas
total_duplicados = df_health.duplicated(keep=False).sum()
print(f'Total de linhas duplicadas: {total_duplicados}')

# Gráfico de dispersão - pares de variáveis
sns.pairplot(df_health)
plt.show()

# Gráfico Pair Plot ou Scatterplot Matrix
sns.pairplot(df_health)
plt.show()


# Correlação
# Usar one-hot encoding para colunas categóricas
df_encoded = pd.get_dummies(df_health, drop_first=False)
# Calcular a matriz de correlação
correlation_matrix = df_encoded.corr()
#correlation_matrix = df_sample.corr()
print(correlation_matrix)
# Criar um mapa de calor da matriz de correlação
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
plt.title("Matriz de Correlação")
plt.show()

print(df_encoded.dtypes)


# Outliers
print(df_health.columns)
# Avaliacao de outiliers 
# Para calcular múltiplos quantis (ex: 0.25, 0.5, 0.75)
print('Coluna com mais correlação com Senior x Adult = LBXGLT:')
quantis_coluna=df_health['LBXGLT'].quantile([0.25, 0.5, 0.75])
print(quantis_coluna)
resumo_coluna=df_health['LBXGLT'].describe()
print(resumo_coluna)

# Treinamento etc...


# define os atributos dependentes e independente:
X=df_health.drop(columns=['age_group', 'RIDAGEYR'])
y=df_health['age_group']

# define o scaler, prepara (aprende) e executa normalização
scaler = MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)

X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y, test_size=0.3, random_state=123)


# Logistic regression com grid search:
base_estimator = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularização
    'penalty': ['l1', 'l2']         # Tipos de penalidade
}

# Configurando o GridSearchCV
clf = GridSearchCV(base_estimator, param_grid, cv=5, scoring='accuracy')
clf.fit(X_train, y_train)

# Resultados
print("Melhores parâmetros:", clf.best_params_)
print("Acurácia no conjunto de teste:", clf.score(X_test, y_test))

print(clf.best_estimator_)

print('\nDetailed classification report:\n')
y_pred=clf.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred)
# Exibindo a matriz de confusão com Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Matriz de Confusão - Acuracidade")
plt.show()


# Abaixo usando o F1score:
    
# Logistic regression com grid search usando F1-score como métrica
clf_f1 = GridSearchCV(base_estimator, param_grid, cv=5, scoring='f1_weighted')  # Alterado para f1_weighted
clf_f1.fit(X_train, y_train)

# Resultados
print("Melhores parâmetros com base no F1-score:", clf_f1.best_params_)
print("F1-score no conjunto de teste:", clf_f1.score(X_test, y_test))

print(clf_f1.best_estimator_)

print('\nDetailed classification report:\n')
y_pred_f1 = clf_f1.predict(X_test)
print(classification_report(y_test, y_pred_f1, zero_division=0))

# Matriz de Confusão
conf_matrix_f1 = confusion_matrix(y_test, y_pred_f1)
# Exibindo a matriz de confusão com Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_f1, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Matriz de Confusão (F1-score)")
plt.show()


# random forest:
    
rf = RandomForestClassifier(class_weight='balanced')  # Ajustar o peso das classes
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

clf_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='f1_weighted')
clf_rf.fit(X_train, y_train)

print("Melhores parâmetros com base no F1-score:", clf_rf.best_params_)
print("F1-score no conjunto de teste:", clf_rf.score(X_test, y_test))

y_pred_rf = clf_rf.predict(X_test)
print(classification_report(y_test, y_pred_rf, zero_division=0))

# matriz de confusao no random forest

y_pred_rf = clf_rf.predict(X_test)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Adult', 'Senior'], yticklabels=['Adult', 'Senior'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Matriz de Confusão (Random Forest)")
plt.show()
# Relatório de classificação
print(classification_report(y_test, y_pred_rf, zero_division=0))

