# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 22:50:31 2025

@author: Marcos
"""

# In[0.1]: Instalação dos pacotes

!pip install pandas
!pip install numpy
!pip install -U seaborn
!pip install matplotlib
!pip install plotly
!pip install scipy
!pip install statsmodels
!pip install scikit-learn
!pip install statstests

# In[0.2]: Importação dos pacotes

import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
from scipy.interpolate import UnivariateSpline # curva sigmoide suavizada
import statsmodels.api as sm # estimação de modelos
import statsmodels.formula.api as smf # estimação do modelo logístico binário
from statstests.process import stepwise # procedimento Stepwise
from scipy import stats # estatística chi2
import plotly.graph_objects as go # gráficos 3D
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
from statsmodels.discrete.discrete_model import MNLogit # estimação do modelo
                                                        #logístico multinomial
import warnings
warnings.filterwarnings('ignore')

# In[Carregando os Dados]

df_heart = pd.read_csv('heart.csv', delimiter=',')
df_heart

# Caracteristicas das Variavéis do Dataset
df_heart.info()

# Estátisticas Univariadas
df_heart.describe()

# Alterando os nomes nas colunas
df_heart.columns = ['Idade', # Int
                    'Sexo', # Obj
                    'TipoDorToracica', # Obj
                    'PressaoArterialRepouso', # Int
                    'Colesterol', # Int
                    'GlicoseJejum', # Int
                    'ECGRepouso', # Obj
                    'FrequenciaCardiacaMáxima', # Int
                    'AnginaExercício', # Obj
                    'OldPeak', # Float
                    'InclinacaoST', # Obj
                    'DoencaCardiaca'] # Int Target
print(df_heart) 

df_heart_dummies = df_heart

# In[Tabela de Frequência absolutas das Variavéis Qualitativas]

df_heart['Sexo'].value_counts().sort_index()
df_heart['TipoDorToracica'].value_counts().sort_index()
df_heart['ECGRepouso'].value_counts().sort_index()
df_heart['AnginaExercício'].value_counts().sort_index()
df_heart['InclinacaoST'].value_counts().sort_index()
df_heart['DoencaCardiaca'].value_counts().sort_index()

# In[Transformaremos as Variavéis OBJ em INT para Rodar no Modelo]

# Aplicando Isso para a Coluna SEXO

df_heart.loc[df_heart['Sexo']=='M', 'Sexo'] = 1
df_heart.loc[df_heart['Sexo']=='F', 'Sexo'] = 0

df_heart['Sexo'] = df_heart['Sexo'].astype('int64')

df_heart['Sexo']

# In[Transformaremos as Variavéis OBJ em INT para Rodar no Modelo]
    
# Aplicando Isso para a Coluna TipoDorToracica

df_heart.loc[df_heart['TipoDorToracica']=='TA', 'TipoDorToracica'] = 1
df_heart.loc[df_heart['TipoDorToracica']=='ATA', 'TipoDorToracica'] = 2
df_heart.loc[df_heart['TipoDorToracica']=='NAP', 'TipoDorToracica'] = 3
df_heart.loc[df_heart['TipoDorToracica']=='ASY', 'TipoDorToracica'] = 4

df_heart['TipoDorToracica'] = df_heart['TipoDorToracica'].astype('int64')

df_heart['TipoDorToracica']

# In[Transformaremos as Variavéis OBJ em INT para Rodar no Modelo]

# Aplicando Isso para a Coluna ECGRepouso

df_heart.loc[df_heart['ECGRepouso']=='Normal', 'ECGRepouso'] = 1
df_heart.loc[df_heart['ECGRepouso']=='ST', 'ECGRepouso'] = 2
df_heart.loc[df_heart['ECGRepouso']=='LVH', 'ECGRepouso'] = 3

df_heart['ECGRepouso'] = df_heart['ECGRepouso'].astype('int64')

df_heart['ECGRepouso']

# In[Transformaremos as Variavéis OBJ em INT para Rodar no Modelo]

# Aplicando Isso para a Coluna AnginaExercicio

df_heart.loc[df_heart['AnginaExercício']=='Y', 'AnginaExercício'] = 1
df_heart.loc[df_heart['AnginaExercício']=='N', 'AnginaExercício'] = 0

df_heart['AnginaExercício'] = df_heart['AnginaExercício'].astype('int64')

df_heart['AnginaExercício']

# In[Transformaremos as Variavéis OBJ em INT para Rodar no Modelo]

# Aplicando Isso para a Coluna InclinacaoST

df_heart.loc[df_heart['InclinacaoST']=='Up', 'InclinacaoST'] = 1
df_heart.loc[df_heart['InclinacaoST']=='Flat', 'InclinacaoST'] = 2
df_heart.loc[df_heart['InclinacaoST']=='Down', 'InclinacaoST'] = 3

df_heart['InclinacaoST'] = df_heart['InclinacaoST'].astype('int64')

df_heart['InclinacaoST']

# In[Verificando novamente as Variáveis]

# Todas estão INT64, aptas para o modelo
df_heart.info()

# In[Outro Metodo é Dummizar as Variáveis]

df_heart_dummies = pd.get_dummies(df_heart_dummies,
                                  columns=['Sexo',
                                           'TipoDorToracica',
                                           'ECGRepouso',
                                           'AnginaExercício',
                                           'InclinacaoST'],
                                  dtype=int,
                                  drop_first=True)

df_heart_dummies

# In[Vamos Estimar o Modelo Logistico Binário]

# Retirando a Target
list_columns = list(df_heart_dummies.drop(columns=['DoencaCardiaca']))

formula_dummies_modelo = ' + '.join(list_columns)
formula_dummies_modelo = "DoencaCardiaca ~ " + formula_dummies_modelo
print(formula_dummies_modelo)

# Modelo usando o SM.LOGITO
modelo_heart = sm.Logit.from_formula(formula_dummies_modelo,
                                     df_heart_dummies).fit()

# In[Análisando os Resultados]

modelo_heart.summary()

# Procedimento de Stepwise
from statstests.process import stepwise

#Estimação do modelo por meio do procedimento Stepwise
step_modelo_heart = stepwise(modelo_heart, pvalue_limit=0.05)

# In[Construindo a Função para a Matriz de Confusão]

from sklearn.metrics import confusion_matrix, accuracy_score,\
    ConfusionMatrixDisplay, recall_score

def matriz_confusao(predicts, observado, cutoff):
    
    values = predicts.values
    
    predicao_binaria = []
        
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)
           
    cm = confusion_matrix(predicao_binaria, observado)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('True')
    plt.ylabel('Classified')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
        
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    #Visualizando os principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Sensitividade':[sensitividade],
                                'Especificidade':[especificidade],
                                'Acurácia':[acuracia]})
    return indicadores

# In[Construção da matriz de confusão]

# Adicionando os valores previstos de probabilidade na base de dados
df_heart_dummies['phat'] = step_modelo_heart.predict()

# Matriz de confusão para cutoff = 0.5
matriz_confusao(observado=df_heart_dummies['DoencaCardiaca'],
                predicts=df_heart_dummies['phat'],
                cutoff=0.50)

# In[Igualando critérios de especificidade e de sensitividade, para fiins didáticos]

def espec_sens(observado,predicts):
    
    # adicionar objeto com os valores dos predicts
    values = predicts.values
    
    # range dos cutoffs a serem analisados em steps de 0.01
    cutoffs = np.arange(0,1.01,0.01)
    
    # Listas que receberão os resultados de especificidade e sensitividade
    lista_sensitividade = []
    lista_especificidade = []
    
    for cutoff in cutoffs:
        
        predicao_binaria = []
        
        # Definindo resultado binário de acordo com o predict
        for item in values:
            if item >= cutoff:
                predicao_binaria.append(1)
            else:
                predicao_binaria.append(0)
                
        # Cálculo da sensitividade e especificidade no cutoff
        sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
        especificidadee = recall_score(observado, predicao_binaria, pos_label=0)
        
        # Adicionar valores nas listas
        lista_sensitividade.append(sensitividade)
        lista_especificidade.append(especificidadee)
        
    # Criar dataframe com os resultados nos seus respectivos cutoffs
    resultado = pd.DataFrame({'cutoffs':cutoffs,'sensitividade':lista_sensitividade,'especificidade':lista_especificidade})
    return resultado

# In[Extraindo os Vetores]

dados_plotagem = espec_sens(observado = df_heart_dummies['DoencaCardiaca'],
                            predicts = df_heart_dummies['phat'])
dados_plotagem

# In[3.11]: Plotagem de um gráfico que mostra a variação da especificidade e da
#sensitividade em função do cutoff

plt.figure(figsize=(15,10))
with plt.style.context('seaborn-v0_8-whitegrid'):
    plt.plot(dados_plotagem.cutoffs,dados_plotagem.sensitividade, marker='o',
         color='indigo', markersize=8)
    plt.plot(dados_plotagem.cutoffs,dados_plotagem.especificidade, marker='o',
         color='limegreen', markersize=8)
plt.xlabel('Cuttoff', fontsize=20)
plt.ylabel('Sensitividade / Especificidade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.legend(['Sensitividade', 'Especificidade'], fontsize=20)
plt.show()

# In[Construção da curva ROC]

from sklearn.metrics import roc_curve, auc

# Função 'roc_curve' do pacote 'metrics' do sklearn

fpr, tpr, thresholds =roc_curve(df_heart_dummies['DoencaCardiaca'],
                                df_heart_dummies['phat'])
roc_auc = auc(fpr, tpr)

# Cálculo do coeficiente de GINI
gini = (roc_auc - 0.5)/(0.5)

# Plotando a curva ROC
plt.figure(figsize=(15,10))
plt.plot(fpr, tpr, marker='o', color='darkorchid', markersize=10, linewidth=3)
plt.plot(fpr, fpr, color='gray', linestyle='dashed')
plt.title('Área abaixo da curva: %g' % round(roc_auc, 4) +
          ' | Coeficiente de GINI: %g' % round(gini, 4), fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitividade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()