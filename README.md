# 🩺 Classificação de Diabetes com Random Forest e SMOTE

## 📘 Descrição do Projeto

Este projeto apresenta a implementação de um **pipeline automatizado em Python** para **classificação de diabetes tipo II**, utilizando o algoritmo **Random Forest** em conjunto com a técnica **SMOTE (Synthetic Minority Over-sampling Technique)**.  

O objetivo é investigar a influência do balanceamento de classes no desempenho de modelos de aprendizado supervisionado, aplicando técnicas de pré-processamento, validação cruzada e otimização estocástica de hiperparâmetros.  

---
 
## ⚙️ Metodologia

### 1. Pré-processamento
- Tratamento de valores **inválidos (0)** em variáveis fisiologicamente impossíveis: `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`.  
- **Imputação estatística** pela mediana (`SimpleImputer`).  
- **Normalização z-score** com `StandardScaler`.  
- Divisão **estratificada** em treino e teste (80/20).

### 2. Balanceamento
- Aplicação do **SMOTE** (*Chawla et al., 2002*) apenas sobre o conjunto de treino, para gerar amostras sintéticas da classe minoritária.

### 3. Modelagem e Otimização
- **Modelo base:** `RandomForestClassifier` (*Breiman, 2001*).  
- **Busca de hiperparâmetros:** `RandomizedSearchCV` com validação cruzada 5-fold e métrica de otimização **F1-score**.  

Parâmetros testados:
```python
param_grid = {
    'classifier__n_estimators': [100, 300, 500],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__max_features': ['sqrt', 'log2']
}
```

### 4. Avaliação
- Métricas utilizadas: *precision*, *recall*, *F1-score* e *accuracy*.  
- Visualizações geradas:
  - Distribuição das classes originais.  
  - Matriz de confusão normalizada.  
  - Curva ROC e cálculo da AUC.  

---

## 🗂️ Dataset

O conjunto de dados utilizado neste projeto é o **Pima Indians Diabetes Database**, disponibilizado publicamente no Kaggle.  

📂 **Fonte:**  
> [Diabetes Data Set – Kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data)  
> *Base derivada do Pima Indians Diabetes Database (UCI Machine Learning Repository).*

### 📈 Descrição Geral
O dataset contém **768 observações** de mulheres de origem Pima (população indígena norte-americana), com **8 variáveis preditoras** e **1 variável alvo binária** (`Outcome`), que indica se a paciente foi diagnosticada com diabetes tipo II.

### 🔍 Estrutura das Variáveis

| Variável | Descrição | Tipo | Intervalo / Unidade |
|-----------|------------|------|---------------------|
| `Pregnancies` | Número de gestações | Numérica (int) | 0 – 17 |
| `Glucose` | Concentração de glicose plasmática (2h) | Numérica (int) | 0 – 199 mg/dL |
| `BloodPressure` | Pressão arterial diastólica | Numérica (int) | 0 – 122 mmHg |
| `SkinThickness` | Espessura da dobra cutânea do tríceps | Numérica (int) | 0 – 99 mm |
| `Insulin` | Nível sérico de insulina em 2h | Numérica (int) | 0 – 846 µU/mL |
| `BMI` | Índice de massa corporal | Numérica (float) | 0 – 67.1 kg/m² |
| `DiabetesPedigreeFunction` | Histórico familiar de diabetes | Numérica (float) | 0.078 – 2.42 |
| `Age` | Idade (anos) | Numérica (int) | 21 – 81 |
| `Outcome` | Diagnóstico (0 = Não Diabético, 1 = Diabético) | Categórica (binária) | — |

### 💡 Observações Importantes
- Variáveis com valor “0” foram tratadas como **valores ausentes** em atributos fisiologicamente impossíveis.  
- O dataset apresenta **desbalanceamento moderado**, com aproximadamente **65% da classe 0** (não diabéticos) e **35% da classe 1** (diabéticos).  
- É amplamente utilizado como benchmark em tarefas de **classificação médica supervisionada**.  

> **Uso:** Os dados foram utilizados exclusivamente para fins **acadêmicos e de pesquisa**, respeitando as condições de uso descritas no Kaggle.

---

## 📊 Resultados

| Métrica | Classe 0 (Não Diabetes) | Classe 1 (Diabetes) | Média Ponderada |
|----------|------------------------|----------------------|----------------|
| Precision | 0.84 | 0.61 | 0.76 |
| Recall | 0.74 | 0.74 | 0.74 |
| F1-Score | 0.79 | 0.67 | **0.74** |
| Acurácia | — | — | **0.74** |

**Melhores parâmetros:**  
`{'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 10}`  

**F1-Score médio (validação cruzada):** 0.6868  

A combinação **Random Forest + SMOTE** proporcionou um aumento significativo no *recall* da classe minoritária, aspecto crítico em diagnósticos clínicos sensíveis.

---

## 🧠 Discussão

O uso do **SMOTE** mitigou o impacto do desbalanceamento de classes, reduzindo falsos negativos sem causar sobreajuste perceptível.  
O modelo apresentou **robustez estatística** e **boa generalização**, validando o uso de pipelines integrados de balanceamento e modelagem não linear em contextos biomédicos.

---

## 🧩 Execução

### 📦 Dependências
```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn
```

### ▶️ Execução do Script
1. Edite o caminho do dataset no script:
   ```python
   file_path = '/caminho/para/seu/arquivo/diabetes.csv'
   ```
2. Execute:
   ```bash
   python randomforest+smote.py
   ```

Todos os resultados e gráficos são gerados automaticamente no terminal e em janelas interativas (*matplotlib*).

---

## 📁 Estrutura do Projeto

```
📦 RandomForest_SMOTE/
 ├── randomforest+smote.py        # Script principal (executável)
 ├── README.md                    # Documento técnico e metodológico
 └── data/
     └── diabetes.csv             # Base de dados
```

---

## 🧾 Referências Bibliográficas

- **Breiman, L.** (2001). *Random Forests*. *Machine Learning*, 45(1), 5–32.  
  https://doi.org/10.1023/A:1010933404324  
- **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P.** (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. *Journal of Artificial Intelligence Research*, 16, 321–357.  
  https://doi.org/10.1613/jair.953  
- **Pedregosa, F., et al.** (2011). *Scikit-learn: Machine Learning in Python*. *Journal of Machine Learning Research*, 12, 2825–2830.  
- **Lemaître, G., Nogueira, F., & Aridas, C. K.** (2017). *Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning*. *Journal of Machine Learning Research*, 18(17), 1–5.  

---

📄 *Documento técnico-acadêmico revisado, padronizado e alinhado ao formato BreaKHis, com ênfase em clareza metodológica, reprodutibilidade e integridade científica.*
