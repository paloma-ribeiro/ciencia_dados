"""
Projeto Ciência de Dados - Previsão de Vendas

Desafio

Conseguir prever as vendas que vamos ter em um determinado período com base nos gastos
em anúncios nas 3 grandes redes que a empresa Hashtag investe: TV, Jornal e Rádio.

- TV, Jornal e rádio estão em milhares de reais
- Vendas estão em milhôes

Base de Dados: https://drive.google.com/drive/folders/1o2lpxoi9heyQV1hIlsHXWSfDkBPtze-V

Passo a Passo de um Projeto de Ciência de Dados:
Passo 1: Entendimento do desafio
Passo 2: Entendimento da área/empresa
Passo 3: Extração/Obtenção de dados
Passo 4: Ajuste de dados (Tratamento/limpeza)
Passo 5: Análise exploratória
Passo 6: Modelagem + algoritmos (aqui entra a inteligência artificial, se necessário)
Passo 7: Interpretação de resultados

matplotlib: grafico
seaborn: grafico
scikit-learn: inteligência artificial
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Importar a base de dados
tabela = pd.read_csv('bd/advertising.csv')

# Análise exploratória
# Visualizar como as informações de cada item estão distribuídas
# Verificar a correlação entre cada um dos itens

# Criar o gráfico
sns.heatmap(tabela.corr(), cmap='Wistia', annot=True)
# Exibir o gráfico
plt.show()

# Separar as informações em treinos e testes
# y -> quem você quer prever
# x -> quem será usado para fazer a previsão
y = tabela['Vendas']
x = tabela[['TV', 'Radio', 'Jornal']]

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

# Escolher os modelos que serão utilizados
# Regressão linear
# RandomForest (árvore de decisão)

# Criar os modelos
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

# Treinar os modelos
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

# Teste da AI e avaliação do melhor modelo
# Usar R**2 -> diz a % que o nosso modelo consegue explicar o que acontece
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

print(metrics.r2_score(y_teste, previsao_regressaolinear))
print(metrics.r2_score(y_teste, previsao_arvoredecisao))

# arvore de decisão é o melhor modelo, vamos utiliza-lo para fazer as previsões

# Visualização gráfica das previsões
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar['y_teste'] = y_teste
tabela_auxiliar['Previsão Regressão Linear'] = previsao_regressaolinear
tabela_auxiliar['Previsão ArvoreDecisao'] = previsao_arvoredecisao

plt.figure(figsize=(15, 6))
sns.lineplot(data=tabela_auxiliar)
plt.show()

# Fazer uma nova previsão
nova_tabela = pd.read_csv('bd/novos.csv')
print(nova_tabela)
previsao = modelo_arvoredecisao.predict(nova_tabela)
print(previsao)

