#!pip install pandas
#!pip install mlxtend
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

#!pip install sklearn

from sklearn import tree
from mlxtend.frequent_patterns import apriori

conjunto_elementos60=apriori(datos,min_support=0.6,use_colnames=True)
conjunto_elementos60

conjunto_elementos=apriori(datos,min_support=0.4,use_colnames=True)
conjunto_elementos

conjunto_elementos['length'] = conjunto_elementos['itemsets'].apply(lambda x:len(x))
conjunto_elementos[conjunto_elementos['length']>1]

conjunto_elementos[conjunto_elementos['itemsets']=={'Camisa','Zapatos,'}]

conjunto_elementos[(conjunto_elementos['length']>2)&(conjunto_elementos['support']==0.4)]
