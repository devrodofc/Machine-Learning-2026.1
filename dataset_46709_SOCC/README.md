# Text Classification with KNN & Naive Bayes (dataset_46709_SOCC)

## 📋 Descrição do Projeto

Pipeline de **Machine Learning para Classificação de Textos** utilizando dois algoritmos principais: K-Nearest Neighbors (KNN) e Naive Bayes. O projeto implementa um extrator customizado de características usando Bag of Words e utiliza validação cruzada para avaliação robusta.

## 🎯 Objetivo

Comparar a performance de dois algoritmos de classificação em dados de texto:
1. K-Vizinhos (KNN) com diferentes métricas de distância
2. Naive Bayes (univariado e multivariado)

Usando pré-processamento customizado com Bag of Words.

## 📊 Dataset

- **Formato**: ARFF (Attribute-Relation File Format)
- **Arquivo**: `data/dataset_46709.arff`
- **Tipo**: Classificação (variável categórica)
- **Domínio**: Textos para classificação
- **Aplicação**: Classificação automática de documentos/textos

## 🔧 Estrutura de Arquivos

```
dataset_46709_SOCC/
├── main.py              # Pipeline principal
├── data/
│   └── dataset_46709.arff
└── src/
    ├── loader.py        # Carregamento de dados
    ├── knn.py           # Implementação de KNN
    ├── bayes.py         # Implementação de Naive Bayes
    └── metrics.py       # Métricas e validação cruzada
```

## 🚀 Como Executar

### Pré-requisitos
```bash
pip install numpy
```

### Execução
```bash
cd dataset_46709_SOCC
python main.py
```

## 📈 Pipeline de Execução

1. **Carregamento de Dados**
   - Leitura do arquivo ARFF
   - Extração de textos e labels

2. **Pré-processamento**
   - Limpeza de pontuação
   - Conversão para minúsculas
   - Tokenização

3. **Extração de Características**
   - Construção customizada de Bag of Words
   - Seleção das Top N palavras frequentes
   - Vetorização de textos

4. **Instantiação de Modelos**
   - KNN (k=5, distância euclidiana/manhattan)
   - Naive Bayes (univariado e multivariado)

5. **Validação Cruzada (5-Fold)**
   - Treinamento e teste em cada fold
   - Coleta de métricas

6. **Avaliação**
   - Cálculo de acurácia
   - Média e desvio padrão dos resultados

## 🔤 Pré-processamento de Texto

### Bag of Words Customizado

```python
simple_bag_of_words(textos, vocab_size=15)
```

**Etapas:**

1. **Limpeza**
   - Remove pontuação
   - Converte para minúsculas
   - Mantém apenas caracteres alfanuméricos

2. **Tokenização**
   - Divide textos em palavras
   - Remove espaços extras

3. **Filtragem de Stopwords**
   - Ignora palavras com < 4 caracteres
   - Exemplos: "o", "a", "de", "com"

4. **Seleção de Vocabulário**
   - Conta frequência global de palavras
   - Seleciona Top N palavras mais frequentes
   - Default: 15 palavras

5. **Vetorização**
   - Cria matriz numérica (n_textos × vocab_size)
   - Cada célula = contagem da palavra no texto
   - Formato: Bag of Words (contagem/frequência)

**Exemplo:**

```
Texto 1: "Python é uma linguagem de programação poderosa"
Texto 2: "Python é fácil de aprender"

Após BoW:
        python  linguagem  programação  poderosa  fácil  aprender
Texto1:   1         1            1           1       0       0
Texto2:   1         0            0           0       1       1
```

## 📊 Métricas de Avaliação

### Acurácia

$$\text{Acurácia} = \frac{TP + TN}{TP + TN + FP + FN}$$

- Intervalo: [0, 1]
- 1 = Classificação perfeita
- 0.5 = Aleatório (binary classification)

Onde:
- TP = True Positives
- TN = True Negatives
- FP = False Positives
- FN = False Negatives

## 🧮 Algoritmos Implementados

### 1. K-Nearest Neighbors (KNN)

```python
KNN(k=5, dist_metric='euclidean')
```

**Funcionamento:**
- Para cada ponto de teste, encontra os K vizinhos mais próximos
- **Votação majoritária**: A classe mais frequente entre os K vizinhos vence
- Sem fase de treinamento explícita (lazy learner)

**Distâncias:**
- **Euclidiana**: $\sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$
- **Manhattan**: $\sum_{i=1}^{n}|x_i - y_i|$

**Vantagens:**
- Simples e intuitivo
- Não assume distribuição dos dados
- Bom para dados com padrões locais

**Desvantagens:**
- Lento em tempo de teste (lazy learning)
- Sensível a outliers
- Sensível à escala de features

### 2. Naive Bayes

```python
NaiveBayes(multivariado=False)
```

**Modelo Probabilístico:**
$$P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}$$

**Funcionamento:**
1. Calcula probabilidade a priori: $P(C)$
2. Estima probabilidade condicional: $P(X|C)$
3. Classifica argumento com maior probabilidade posterior

**Versões:**

#### Naive Bayes Univariado
- Assume distribuição Gaussiana
- Cada feature é independente
- Menor complexidade computacional

$$P(x_i|C) = \frac{1}{\sqrt{2\pi\sigma_C^2}} e^{-\frac{(x_i - \mu_C)^2}{2\sigma_C^2}}$$

#### Naive Bayes Multivariado
- Utiliza matriz de covariância completa
- Captura correlações entre features
- Maior poder expressivo
- Requer mais dados de treinamento

**Vantagens:**
- Treinamento rápido
- Teste rápido
- Funciona bem com alta dimensionalidade
- Interpretável

**Desvantagens:**
- Assume independência entre features (violado frequentemente)
- Pode ser afetado por probabilidade zero
- Menos flexível que árvores

## 📋 Exemplo de Saída

```
Iniciando Pipeline de Machine Learning...
Vetorizando textos (Bag of Words manual)...
Vocabulário selecionado: ['python', 'programação', 'linguagem', ...]

Dataset processado: 150 amostras e 15 atributos

Executando 5-Fold Cross Validation...

=== RESULTADOS ===

KNN (Distância Euclidiana):
  Acurácia: 0.82 ± 0.05
  Tempo Médio: 0.025s

KNN (Distância Manhattan):
  Acurácia: 0.80 ± 0.06
  Tempo Médio: 0.028s

Naive Bayes (Univariado):
  Acurácia: 0.85 ± 0.04
  Tempo Médio: 0.003s

Naive Bayes (Multivariado):
  Acurácia: 0.83 ± 0.05
  Tempo Médio: 0.004s
```

## 🔍 Detalhes Técnicos

### Matriz de Covariância no Naive Bayes Multivariado

```python
cov_matrix = np.cov(X_c.T)  # Matriz covariância
np.fill_diagonal(cov_matrix, cov_matrix.diagonal() + 1e-6)  # Regularização
cov_inv = np.linalg.inv(cov_matrix)  # Inversa
cov_det = np.linalg.det(cov_matrix)  # Determinante
```

- **Regularização**: Adiciona pequeno valor diagonal para evitar singularidade
- **Inversa**: Necessária para cálculo de probabilidade
- **Determinante**: Usado na função de densidade Gaussiana multivariada

### Validação Cruzada (K-Fold)

- Embaralha aleatoriamente os índices
- Divide em K partições aproximadamente iguais
- Retorna K pares (treino_idx, teste_idx)

### Tratamento de Dados

- **Bag of Words**: Sem normalização (contagem bruta)
- **Sem normalização L2**: Features em escala de contagem
- **Sem missing values**: Dataset limpo após BoW

## 📚 Arquivos do Módulo `src/`

### loader.py
```python
load_arff(filepath)
process_data(X, y)
```
- Lê arquivo ARFF
- Processa dados para formato numérico
- Retorna (textos, labels)

### knn.py - KNN
```python
class KNN:
    def __init__(self, k=5, dist_metric='euclidean')
    def fit(X, y)
    def predict(X)
```

### bayes.py - Naive Bayes
```python
class NaiveBayes:
    def __init__(self, multivariado=False)
    def fit(X, y)
    def predict(X)
```

### metrics.py
- `k_fold_split(X, y, k=5)`: Gera índices para K-Fold
- `calculate_metrics(y_true, y_pred)`: Calcula acurácia

## 💡 Pontos Importantes

1. **Bag of Words é simples mas efetivo**: Bom baseline para classificação
2. **KNN é sensível a outliers**: Uma observação pode enviesá-lo
3. **Naive Bayes assume independência**: Violado em textos reais
4. **K=5 é empírico**: Poderia ser otimizado via grid search
5. **Multivariado vs Univariado**: Trade-off entre expressividade e dados necessários
6. **Stopwords**: Removidos arbitrariamente por tamanho (< 4 caracteres)

## 🎓 Conceitos de Aprendizado

- Classificação de Textos
- Bag of Words
- Feature Extraction
- K-Nearest Neighbors
- Naive Bayes
- Validação Cruzada
- Probabilidade Bayesiana
- Votação Majoritária

## 🔗 Referências Matemáticas

- **Função de Densidade Gaussiana Multivariada**:
$$f(x|μ,Σ) = \frac{1}{(2π)^{d/2}|Σ|^{1/2}} e^{-\frac{1}{2}(x-μ)^T Σ^{-1}(x-μ)}$$

- **Teorema de Bayes**:
$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

- **Distância Euclidiana**:
$$d(x,y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

- **Distância Manhattan**:
$$d(x,y) = \sum_{i=1}^{n}|x_i - y_i|$$

---

**Autor**: Projeto acadêmico  
**Disciplina**: Inteligência Artificial (AV2)  
**Universidade**: Unifor
