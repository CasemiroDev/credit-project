import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

current_directory = os.getcwd()

credit_file = os.path.join(current_directory, 'credit.csv')

base_credit = pd.read_csv(credit_file)

X_credit = base_credit.iloc[:, [0,1,6]].values

y_credit = base_credit.iloc[:, 8].values

scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)

X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(X_credit, y_credit, test_size= 0.25, random_state = 0)
