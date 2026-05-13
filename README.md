# Trabalho IA AV2 - Projetos de Machine Learning

Projeto acadêmico de Inteligência Artificial contendo dois pipelines de aprendizado de máquina com diferentes abordagens e datasets.

## 📁 Estrutura do Projeto

Este repositório contém dois projetos independentes de Machine Learning:

```
trablho IA AV2/
├── dataset_41557_sf2/           # Projeto de Regressão
│   ├── main.py
│   ├── data/                    # Dataset de Erupções Solares (ARFF)
│   └── src/
│       ├── bayes.py
│       ├── loader.py
│       ├── metrics.py
│       └── regressors.py
│
└── dataset_46709_SOCC/          # Projeto de Classificação
    ├── main.py
    ├── data/                    # Dataset de Textos (ARFF)
    └── src/
        ├── bayes.py
        ├── knn.py
        ├── loader.py
        └── metrics.py
```

---

## 🔴 Projeto 1: Solar Flare Regression (dataset_41557_sf2)

### Descrição
Pipeline de regressão para prever a magnitude de erupções solares utilizando dados meteorológicos processados com One-Hot Encoding.

### Algoritmos Implementados
- **K-Vizinhos Regressão (KNN)** - Distância Euclidiana
- **K-Vizinhos Regressão (KNN)** - Distância Manhattan  
- **Regressão Linear Múltipla** - Método dos mínimos quadrados

### Métricas de Avaliação
- **R² Score** - Coeficiente de determinação
- **R² Ajustado** - R² corrigido por número de atributos

### Validação
- **5-Fold Cross Validation** com divisão aleatória de dados

### Como Executar
```bash
cd dataset_41557_sf2
python main.py
```

### Estrutura de Arquivos
- `main.py` - Pipeline principal com K-Fold e avaliação de modelos
- `data/file20381d055179.arff` - Dataset em formato ARFF
- `src/loader.py` - Carregamento e processamento de dados (One-Hot Encoding)
- `src/regressors.py` - Implementação de KNN Regressor e Regressão Linear Múltipla
- `src/metrics.py` - Cálculo de K-Fold, R² e R² Ajustado
- `src/bayes.py` - Implementação de Naive Bayes

### Saída Esperada
```
Média R²: 0.XXXX ± 0.XXXX
Média R² Ajustado: 0.XXXX ± 0.XXXX
Tempo Total de Treinamento: X.XXs
```

---

## 🟢 Projeto 2: Text Classification with KNN & Naive Bayes (dataset_46709_SOCC)

### Descrição
Pipeline de classificação de textos utilizando Bag of Words como extractor de características, com algoritmos KNN e Naive Bayes.

### Algoritmos Implementados
- **K-Vizinhos (KNN)** - Classificação por votação majoritária
  - Distância Euclidiana e Manhattan
- **Naive Bayes** - Classificação probabilística
  - Univariado e Multivariado

### Pré-processamento
- **Bag of Words (BoW)** customizado
- Limpeza de pontuação
- Conversão para minúsculas
- Remoção de stopwords (palavras com < 4 caracteres)
- Top N palavras mais frequentes como vocabulário

### Validação
- **K-Fold Cross Validation** (k=5)

### Como Executar
```bash
cd dataset_46709_SOCC
python main.py
```

### Estrutura de Arquivos
- `main.py` - Pipeline principal com vetorização BoW e K-Fold
- `data/dataset_46709.arff` - Dataset em formato ARFF
- `src/loader.py` - Carregamento e processamento de dados
- `src/knn.py` - Implementação do algoritmo KNN
- `src/bayes.py` - Implementação do Naive Bayes (univariado e multivariado)
- `src/metrics.py` - Cálculo de K-Fold e métricas de classificação

### Saída Esperada
```
Acurácia KNN: 0.XX ± 0.XX
Acurácia Naive Bayes: 0.XX ± 0.XX
```

---

## 🛠️ Requisitos

- Python 3.7+
- NumPy
- Bibliotecas padrão (time, collections)

### Instalação de Dependências
```bash
pip install numpy
```

---

## 📊 Conceitos de Machine Learning Utilizados

### Validação Cruzada (K-Fold)
- Divisão dos dados em K partições
- Rotação de conjunto de teste/treino
- Melhor estimativa de performance em dados limitados

### Algoritmos
- **K-Nearest Neighbors (KNN)**: Baseado em distância, sem fase de treinamento
- **Naive Bayes**: Algoritmo probabilístico baseado em Teorema de Bayes
- **Regressão Linear**: Ajuste de hiperplano aos dados

### Métricas
- **Acurácia**: (TP + TN) / Total
- **R² Score**: Proporção de variância explicada
- **R² Ajustado**: R² corrigido pelo número de atributos

---

## 📝 Notas Importantes

1. Os datasets devem estar nos respectivos diretórios `data/`
2. Ambos os projetos usam validação cruzada para robustez
3. Implementações são feitas do zero (não usam scikit-learn)
4. Atributos são processados automaticamente no pipeline

---

## 👨‍💼 Autoria

Projeto acadêmico - Universidade de Fortaleza (Unifor)  
Disciplina: Inteligência Artificial  
Avaliação: AV2

---

## 📄 Licença

Projeto acadêmico - Uso exclusivo educacional
