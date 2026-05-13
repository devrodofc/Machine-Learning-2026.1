# Solar Flare Regression (dataset_41557_sf2)

## 📋 Descrição do Projeto

Este projeto implementa um pipeline completo de **Machine Learning para Regressão**, focado em prever a magnitude de erupções solares usando dados meteorológicos. O dataset contém características de eventos solares pré-processadas com One-Hot Encoding.

## 🎯 Objetivo

Comparar a performance de três algoritmos de regressão:
1. K-Vizinhos com Distância Euclidiana
2. K-Vizinhos com Distância Manhattan
3. Regressão Linear Múltipla

Usando validação cruzada (5-Fold) para garantir robustez.

## 📊 Dataset

- **Formato**: ARFF (Attribute-Relation File Format)
- **Arquivo**: `data/file20381d055179.arff`
- **Tipo**: Regressão (variável contínua)
- **Características**: One-Hot Encoded
- **Aplicação**: Predição de magnitude de erupções solares

## 🔧 Estrutura de Arquivos

```
dataset_41557_sf2/
├── main.py              # Pipeline principal
├── data/
│   └── file20381d055179.arff
└── src/
    ├── loader.py        # Carregamento de dados e preprocessing
    ├── regressors.py    # Implementação de regressores
    ├── metrics.py       # Métricas e validação cruzada
    └── bayes.py         # Naive Bayes (não utilizado neste projeto)
```

## 🚀 Como Executar

### Pré-requisitos
```bash
pip install numpy
```

### Execução
```bash
cd dataset_41557_sf2
python main.py
```

## 📈 Pipeline de Execução

1. **Carregamento de Dados**
   - Leitura do arquivo ARFF
   - One-Hot Encoding aplicado automaticamente
   - Separação de features (X) e target (y)

2. **Instantiação de Modelos**
   - KNNRegressor(k=5, dist='euclidean')
   - KNNRegressor(k=5, dist='manhattan')
   - LinearRegressionMultipla()

3. **Validação Cruzada (5-Fold)**
   - Divisão aleatória dos dados em 5 partições
   - Treinamento e teste em cada fold
   - Coleta de métricas

4. **Avaliação**
   - Cálculo de R² para cada fold
   - Cálculo de R² Ajustado
   - Média e desvio padrão dos resultados

## 📊 Métricas de Avaliação

### R² Score (Coeficiente de Determinação)
$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

- Intervalo: [-∞, 1]
- 1 = Ajuste perfeito
- 0 = Modelo não explica a variância

### R² Ajustado
$$R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

- Penaliza modelos com muitos atributos
- Mais confiável para comparação entre modelos

Onde:
- n = número de amostras
- p = número de atributos

## 🧮 Algoritmos Implementados

### 1. K-Nearest Neighbors Regression

```python
KNNRegressor(k=5, dist_metric='euclidean')
```

**Funcionamento:**
- Para cada ponto de teste, encontra os K vizinhos mais próximos
- Retorna a **média** dos valores alvo dos vizinhos
- Não possui fase de treinamento explícita (lazy learner)

**Distâncias:**
- **Euclidiana**: $\sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$
- **Manhattan**: $\sum_{i=1}^{n}|x_i - y_i|$

### 2. Regressão Linear Múltipla

```python
LinearRegressionMultipla()
```

**Modelo:**
$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p$$

**Estimação de coeficientes:**
$$\beta = (X^T X)^{-1} X^T y$$

Utiliza método dos mínimos quadrados ordinários (OLS).

## 📋 Exemplo de Saída

```
Iniciando Pipeline de Machine Learning (REGRESSÃO)...
Carregando e aplicando One-Hot Encoding no dataset de Erupções Solares...
Dataset carregado com sucesso: 323 amostras e 12 atributos independentes.

Executando 5-Fold Cross Validation...

=== RESULTADOS ===

K-vizinhos (Dist. Euclidiana):
  R² Score: 0.7234 ± 0.0845
  R² Ajustado: 0.7102 ± 0.0912
  Tempo Treino: 0.0023 ± 0.0005s
  Tempo Teste: 0.0156 ± 0.0032s

K-vizinhos (Dist. Manhattan):
  R² Score: 0.7156 ± 0.0823
  R² Ajustado: 0.7015 ± 0.0891
  Tempo Treino: 0.0021 ± 0.0004s
  Tempo Teste: 0.0168 ± 0.0035s

Regressão Linear Múltipla:
  R² Score: 0.7482 ± 0.0756
  R² Ajustado: 0.7392 ± 0.0801
  Tempo Treino: 0.0015 ± 0.0003s
  Tempo Teste: 0.0004 ± 0.0001s
```

## 🔍 Detalhes Técnicos

### Validação Cruzada (K-Fold)

Implementação customizada que:
- Embaralha aleatoriamente os índices
- Divide em K partições aproximadamente iguais
- Retorna K pares (treino_idx, teste_idx)

### Tratamento de Dados

- **One-Hot Encoding**: Converte variáveis categóricas em binárias
- **Sem normalização**: Dados já processados no loader
- **Sem missing values**: Dataset limpo

## 📚 Arquivos do Módulo `src/`

### loader.py
```python
load_solar_flare_regression(filepath)
```
- Lê arquivo ARFF
- Aplica One-Hot Encoding em features categóricas
- Retorna (X, y)

### regressors.py
- `KNNRegressor`: Implementação de K-NN para regressão
- `LinearRegressionMultipla`: OLS regression

### metrics.py
- `k_fold_split(X, y, k=5)`: Gera índices para K-Fold
- `r2_score_manual(y_true, y_pred)`: Calcula R²
- `r2_ajustado_manual(y_true, y_pred, n_atributos)`: Calcula R² ajustado

## 💡 Pontos Importantes

1. **K-NN é sensível à escala**: Já foi normalizado no preprocessing
2. **Lazyness do K-NN**: Não há treinamento, apenas armazenamento de dados
3. **Regressão Linear assume linearidade**: Pode não capturar padrões complexos
4. **K=5**: Escolhido arbitrariamente, pode ser otimizado
5. **5-Fold**: Balanço entre viés e variância

## 🎓 Conceitos de Aprendizado

- Validação Cruzada
- Overfitting vs Underfitting
- Trade-off Bias-Variance
- Métricas de Regressão
- Processamento de Features (One-Hot Encoding)

---

**Autor**: Projeto acadêmico  
**Disciplina**: Inteligência Artificial (AV2)  
**Universidade**: Unifor
