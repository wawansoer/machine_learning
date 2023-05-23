import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = pd.read_csv("weight-height.csv", index_col=0)
lr = LinearRegression()

data.rename(columns={'Height(Inches)': 'Height',
            'Weight(Pounds)': 'Weight'}, inplace=True)
# print(data.columns)
print(data.head())

# konversi inci ke cm
data['Height'] = data['Height'] * (2.54)

# konversi pound to kg
data['Weight'] = data['Weight'] / (2.205)

print(data.head())
x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

print(x)
print(y)
# data di inputkan
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=20, random_state=0)

print(x_test)
lr.fit(x_train, y_train)

tinggi = int(input("Masukan Tinggi Badan Untuk Prediksi Berat Badan : "))

# rumus regresi linear y = m * x + c
m = lr.coef_
c = lr.intercept_
x = tinggi
# hasil prediksi
y = m * x + c

print("Hasil Prediksi adalah : ", y)
