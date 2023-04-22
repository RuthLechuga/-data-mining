!pip install pandas
!pip install mlxtend
import pandas as pd
import os
import csv
from mlxtend.preprocessing import TransactionEncoder

archivo = os.getcwd() + "/../Escritorio/-data-mining/GroceryStoreDS.txt"
with open(archivo,'r',encoding='utf-8') as f:
  contents = f.read()
  print(contents)

f=open(archivo,'r',encoding='utf-8')
lista=[]
lista=[line.split() for line in f]
lista

te=TransactionEncoder()
te_ary=te.fit(lista).transform(lista)
datos=pd.DataFrame(te_ary,columns=te.columns_)
datos

archivo_csv = os.getcwd() + "/../Escritorio/-data-mining/GroceryStoreDS.csv"
documentocsv = csv.reader(open(archivo_csv, encoding='utf-8'), skipinitialspace=True)
listas = []
for lista in documentocsv:
    listas.append(lista)

listas
