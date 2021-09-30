import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

from sklearn.datasets import load_boston
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = datasets.load_boston()
# print(data)

df = pd.DataFrame(data=data['data'], columns=data['feature_names'])

# print(df.head())
# print(df.shape)
# print(df.describe())
# print(data.feature_names)
# print(df.head())

# Preço das casas
print(data.target)

df['PRICE'] = data.target
print(df.head())

x = df.drop('PRICE', axis=1)
y = df.PRICE

plt.scatter(df.RM, y)
plt.xlabel('Número de Quarto')
plt.ylabel('Preço da Casa')
plt.title('Relação entre Número de Quartos e Preço')
plt.show()

regr = LinearRegression()

# Tipo de Objeto
print(regr)

# Treinando o Modelo
regr.fit(x, y)

# Comparando preços original X preço previstos
plt.scatter(df.PRICE, regr.predict(x))
plt.xlabel('Preço Original')
plt.ylabel('Preço Previsto')
plt.title('Preço Original X preço Previsto')
plt.show()

# Mean Squared Error
msel = np.mean((df.PRICE - regr.predict(x)) ** 2)
print(msel)

# Aplicando regressão linear para apenas um variável e calculando o MSE
regr = LinearRegression()
regr.fit(x[['PTRATIO']], df.PRICE)
mse2 = np.mean((df.PRICE - regr.predict(x[['PTRATIO']])) ** 2)
print(mse2)

# Dividindo X em dados de treino e de teste
x_treino = x[:-50]
x_teste = x[-50:]

# Dividindo Y em dados de treino e de teste
y_treino = df.PRICE[:-50]
y_teste = df.PRICE[-50:]

# Imprimindo o shape dos dataset
print(x_treino, x_teste, y_treino, y_teste)

# Dividindo x e y em dados de treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, df.PRICE, test_size=0.30, random_state=5)

# Imprimindo o shape dos datasets
# print(x_treino.shape, x_teste.shape, y_treino.shape, y_teste.shape)

# Contruindo um modelo de regressão
regr = LinearRegression()

# Treinando o modelo
regr.fit(x_treino, y_treino)

# Definindo os dados de treino e teste
pred_treino = regr.predict(x_treino)
pred_teste = regr.predict(x_teste)

# Comparando os preços originais
plt.scatter(regr.predict(x_treino), regr.predict(x_treino) - y_treino, c='b', s=40, alpha=0.5)
plt.scatter(regr.predict(x_teste), regr.predict(x_teste) - y_teste, c='g', s=40, alpha=0.5)
plt.hlines(y=0, xmin=0, xmax=50)
plt.ylabel("Resíduo")
plt.title("Resítuo Plot - Treino(Azul), Teste(Verde)")
plt.show()