#!pip install plotly

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

formato = lambda x:datetime.strptime(x,"%d/%m/%y")

archivo = os.getcwd() + "/../Escritorio/-data-mining/Consumo_cerveza.csv"
datos=pd.read_csv(archivo, parse_dates=["Fecha"])
datos = datos.dropna()

datos.columns = ['fecha', 'Temp_Media', 'Temp_Min', 'Temp_Max', 'Precipitacion ', 'Finde', 'Consumo_Litros']

dias_entre_semana=sum(datos[datos.Finde==0]['Consumo_Litros'])
fin_de_semana=sum(datos[datos.Finde==1]['Consumo_Litros'])

etiqueta=['Dia entre semana','Fin de semana']
valores=[dias_entre_semana,fin_de_semana]
colores=['crimson']

fig=go.Figure(data=[go.Bar(x=etiqueta,y=valores,marker_color=colores)])
fig.show()

datos['fecha']=pd.to_datetime(datos['fecha'],format="%d/%m/%Y")
datos['Mes']=datos['fecha'].apply(lambda x: x.strftime('%B'))
datos['Dia']=datos['fecha'].apply(lambda x: x.strftime('%A'))
datos

figura = px.box(datos,x="Dia",y="Consumo_Litros",color="Dia",orientation="v",notched=True,title="Consumo de litros de cerveza por día de la semana")
figura.show()

figura_mes = px.box(datos,x="Mes",y="Consumo_Litros",color="Mes",orientation="v",notched=True,title="Consumo de litros de cerveza por mes del año")
figura_mes.show()
