# atividade-titanic
Atividade IA - Unisociesc
# Análise de Desempenho do Modelo k-NN no Conjunto de Dados Titanic

## Descrição do Projeto

Este projeto utiliza um classificador k-NN (k-Nearest Neighbors) para prever a sobrevivência de passageiros do Titanic com base em características como classe, sexo, idade e tarifa paga.

## Etapas do Projeto

1. **Montar o Google Drive**: Inicialmente, montamos o Google Drive para acessar o arquivo CSV.
   
   ```python
   # Montar o Google Drive
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Importar Bibliotecas**: Importamos as bibliotecas necessárias, incluindo pandas para manipulação de dados.

   ```python
   # Importar bibliotecas
   import pandas as pd
   ```

3. **Carregamento e Exploração dos Dados**: Carregamos o arquivo CSV do Titanic em um DataFrame e exploramos suas primeiras linhas e informações básicas.

   ```python
   # Exemplo de caminho no Google Drive
   file_path = '/content/drive/MyDrive/Aula-IA/train.csv'

   # Carregar o arquivo CSV em um DataFrame
   df = pd.read_csv(file_path)

   # Mostrar as primeiras linhas do DataFrame
   print("Primeiras linhas do DataFrame:")
   print(df.head())

   # Informações sobre o DataFrame
   print("\nInformações sobre o DataFrame:")
   print(df.info())
   ```

4. **Tratamento de Dados Ausentes**: Identificamos e tratamos valores ausentes nas colunas relevantes, como 'Age' e outras colunas numéricas.

   ```python
   # Identificar valores ausentes
   print("\nValores nulos por coluna:")
   print(df.isnull().sum())

   # Preencher valores nulos em 'Age' com a média
   df['Age'] = df['Age'].fillna(df['Age'].mean())

   # Preencher valores nulos apenas nas colunas numéricas
   numeric_cols = df.select_dtypes(include=['number']).columns
   df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

   # Verificar se todos os valores nulos foram tratados
   print("\nValores nulos após tratamento:")
   print(df.isnull().sum())
   ```

5. **Seleção de Variáveis e Preparação dos Dados**: Selecionamos as variáveis relevantes para a análise e convertimos variáveis categóricas em numéricas usando dummy variables.

   ```python
   # Seleção das colunas para análise
   cols_selecionadas = ['Pclass', 'Sex', 'Age', 'Fare', 'Survived']

   # Criar um DataFrame apenas com as colunas selecionadas
   df_selecionado = df[cols_selecionadas]

   # Preencher valores nulos em 'Age' com a média (caso existam)
   df_selecionado.loc[:, 'Age'] = df_selecionado['Age'].fillna(df_selecionado['Age'].mean())

   # Converter a coluna 'Sex' (categórica) em variáveis numéricas usando pd.get_dummies()
   df_selecionado = pd.get_dummies(df_selecionado, columns=['Sex'], drop_first=True)
   ```

6. **Divisão dos Dados e Treinamento do Modelo k-NN**: Dividimos os dados em conjuntos de treino e teste, e treinamos um modelo k-NN com k=3.

   ```python
   # Definir X (variáveis de entrada) e y (variável de saída)
   X = df_selecionado[['Pclass', 'Sex_male', 'Age', 'Fare']]  # Colunas de entrada
   y = df_selecionado['Survived']  # Coluna alvo

   # Dividir os dados entre treino e teste (70% para treino, 30% para teste)
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

   # Importar o modelo k-NN e treiná-lo com k=3
   from sklearn.neighbors import KNeighborsClassifier
   knn = KNeighborsClassifier(n_neighbors=3)
   knn.fit(X_train, y_train)
   ```

7. **Avaliação do Modelo**: Avaliamos o modelo utilizando métricas como acurácia e matriz de confusão.

   ```python
   # Fazer previsões no conjunto de teste
   y_pred = knn.predict(X_test)

   # Calcular a acurácia
   from sklearn.metrics import accuracy_score
   accuracy = accuracy_score(y_test, y_pred)
   print(f'Acurácia: {accuracy:.2f}')

   # Calcular e exibir a matriz de confusão
   from sklearn.metrics import confusion_matrix
   cm = confusion_matrix(y_test, y_pred)
   print("\nMatriz de Confusão:")
   print(cm)
   ```

## Conclusões e Análise

Este README fornece uma visão detalhada das etapas realizadas para implementar um modelo k-NN no conjunto de dados do Titanic. As métricas de desempenho, como acurácia e matriz de confusão, são utilizadas para avaliar a eficácia do modelo na previsão de sobrevivência dos passageiros com base nas características selecionadas.
