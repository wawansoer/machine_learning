import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

car = pd.read_csv('Cleaned_Data.csv')
print(car.info())

# untuk membuat boxplot
# plt.subplots(figsize=(15, 7))
# a = sns.boxplot(x='company', y='Price', data=car)
# plt.ticklabel_format(style='plain', axis='y')
# plt.show()

# # relplot
# a = sns.relplot(x='kms_driven', y='Price', data=car)
# plt.ticklabel_format(style='plain', axis='y')
# plt.show()

# a = sns.boxplot(x='fuel_type', y='Price', data=car)
# plt.ticklabel_format(style='plain', axis='y')
# plt.show()

# a = sns.relplot(x='company', y='Price', data=car,
#                 hue='fuel_type', size='year', height=6, aspect=2)
# plt.ticklabel_format(style='plain', axis='y')
# plt.show()

X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99)

ohe = OneHotEncoder()

ohe.fit(X[['name', 'company', 'fuel_type']])

column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_), [
                                       'name', 'company', 'fuel_type']), remainder='passthrough')
lr = LinearRegression()

pipe = make_pipeline(column_trans, lr)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

r2_score(y_test, y_pred)

car_name = input("Masukan Nama Mobil : ")
company = input("Masukan Nama Merk : ")
year = int(input("Masukan Tahun Dibuat : "))
kms_driven = int(input("Masukan Total Jarak Tempuh (KM) : "))
fuel_type = input("Masukan Jenis Bahan Bakar : ")

PredictPrice = pipe.predict(pd.DataFrame(columns=X_test.columns, data=np.array(
    [car_name, company, year, kms_driven, fuel_type]).reshape(1, 5)))

print("Mobil Yang Anda Inginkan harganya berkisar ", round(PredictPrice[0], 2))
