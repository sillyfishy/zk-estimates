import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import *
from sklearn.linear_model import *
from sklearn.model_selection import *


df = pd.read_csv("rosamanhattandata.csv")
#cols=list(df.columns.values)
#print(cols)
df['Registration Date'] = pd.to_datetime(df['Registration Date'])    
df['date_delta'] = (df['Registration Date'] - df['Registration Date'].min())
a=df['Registration Date'].apply(pd.Timestamp.toordinal)
x=a
y=df["Derived Carpet Area Rate"]
lr=LogisticRegression()
lr.fit(x.values.reshape(1,-1),y)
 