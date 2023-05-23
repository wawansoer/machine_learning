import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

car = pd.read_csv('cars_data.csv')
print(car.info())

# menampilkan semua data tanpa ada duplikasi pada kolom tahun
print(car['year'].unique())
print("--------------------------------------------------------------------")
# membuat copy disimpan dalam variabel backup
backup = car.copy()

# cleaning column year
car = car[car['year'].str.isnumeric()]
print(car['year'].unique())
print("--------------------------------------------------------------------")

# cleaning column price
print(car['Price'].unique())
print("--------------------------------------------------------------------")
car = car[car['Price'] != 'Ask For Price']
print(car['Price'].unique())
print("--------------------------------------------------------------------")
car['Price'] = car['Price'].str.replace(',', '').astype(int)
print(car['Price'].unique())
print("--------------------------------------------------------------------")

# cleaning km kendaraan
print(car['kms_driven'].unique())
print("--------------------------------------------------------------------")
car['kms_driven'] = car['kms_driven'].str.split(
    ' ').str.get(0).str.replace(',', '')
print(car['kms_driven'].unique())
print("--------------------------------------------------------------------")
car = car[car['kms_driven'].str.isnumeric()]
print(car['kms_driven'].unique())
print("--------------------------------------------------------------------")
car['kms_driven'] = car['kms_driven'].astype(int)
print(car['kms_driven'].unique())
print("--------------------------------------------------------------------")

# clean fuel type
print(car['fuel_type'].unique())
print("--------------------------------------------------------------------")
car = car[~car['fuel_type'].isna()]
print(car['fuel_type'].unique())
print("--------------------------------------------------------------------")

# clean name
print(car['name'].unique())
print("--------------------------------------------------------------------")
car['name'] = car['name'].str.split(' ').str.slice(0, 3).str.join(' ')
print(car['fuel_type'].unique())
print("--------------------------------------------------------------------")

# print(car.info())
# print(backup.info())

car.to_csv('Cleaned_Data.csv')
