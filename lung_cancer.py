#BIBLIOTECAS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

pd.set_option('display.max_columns',21)

#CARREGA ARQUIVO
arquivo = pd.read_csv('survey_lung_cancer.csv') #Disponivel em: https://www.kaggle.com/datasets/jillanisofttech/lung-cancer-detection

#carrega o arquivo
arquivo.head()

#TRTAMENTO DOS DADOS
#Trocando M&F YES&NO por 0&1
GENDER = arquivo['GENDER']
GENDER_INT = []
for i in GENDER:
  if(i == "M"):
    GENDER_INT.append(1)
  else:
    GENDER_INT.append(0)

#comutando trocas
arquivo.drop('GENDER',axis=1, inplace =True)
arquivo.insert(0, "GENDER", GENDER_INT, True)

arquivo.head()

#SEPARA DATASET
#Define saidas
y = arquivo['LUNG_CANCER']

#Define entradas
x = arquivo.drop('LUNG_CANCER', axis=1)

#Separando os dados em treino e teste
x_treino,x_teste,y_treino,y_teste = train_test_split(x,y,test_size=0.3)

#Criando o modelo ETC
modelo_ETC = ExtraTreesClassifier()
modelo_ETC.fit(x_treino,y_treino)
modelo_ETC.score(x_teste,y_teste)

newPatient = pd.DataFrame({'GENDER' : [1],
                  'AGE' : [40],
                  'SMOKING' : [1],
                  'YELLOW_FINGERS' : [2],
                  'ANXIETY' : [2],
                  'PEER_PRESSURE' : [1],
                  'CHRONIC DISEASE' : [2],
                  'FATIGUE ' : [2],
                  'ALLERGY ' : [2],
                  'WHEEZING' : [2],
                  'ALCOHOL CONSUMING' : [2],
                  'COUGHING' : [2],
                  'SHORTNESS OF BREATH' : [2],
                  'SWALLOWING DIFFICULTY' : [2],
                  'CHEST PAIN' : [1]})

newPatient

modelo_ETC.predict(newPatient)[0]