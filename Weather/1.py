import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('weather.csv')

print(df.shape)
print(df.columns)
print(df.info())

# perulangan untuk mengekstrak data dari faile csv yang ditampung dalam variabel categori dimana nilainya merupakan yang tidak bertipe data float
categorical = [var for var in df.columns if df[var].dtype != float]
print('There are {} categories variables \n '.format(len(categorical)))
print('The categories are : ', categorical)
print(df[categorical])

print(df[categorical].isnull().sum())

# perulangan untuk menampilkan jumlah data setiap kategori
for var in categorical:
    print(df[var].value_counts())

# perulangan untuk menampilkan jumlah frekuensi data setiap kategori
for var in categorical:
    print(df[var].value_counts()/len(df))

# perulangan untuk menampilkan jumlah data setiap kategori yang di grouping berdasarkan kategori
for var in categorical:
    print(var, ' contain ', len(df[var].unique()), ' labels')

df['Date'] = pd.to_datetime(df['Date'])

print(df.info())
# ekstrak tanggal menjadi int
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df.drop('Date', axis=1, inplace=True)
print(df.info())

# find categorical variables
categorical = [var for var in df.columns if df[var].dtype == '0']
print('There are {} categories variables \n '.format(len(categorical)))
print('The categories are : ', categorical)
print(df[categorical])

print(df[categorical].isnull().sum())

print(df['Location'].unique())
print(df.Location.value_counts())

pd.get_dummies(df.Location, drop_first=True).head()

print('WindGustDir Contains ', len(df['WindGustDir'].unique()), 'labels')
print(df['WindGustDir'].unique())
pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()
pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0)

print('WindDir9am Contains ', len(df['WindDir9am'].unique()), 'labels')
print(df['WindDir9am'].unique())
pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()
pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0)

print('WindDir3pm Contains ', len(df['WindDir3pm'].unique()), 'labels')
print(df['WindDir3pm'].unique())
pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()
pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0)

print('RainToday Contains ', len(df['RainToday'].unique()), 'labels')
print(df['RainToday'].unique())
pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()
pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).sum(axis=0)

print('RainTomorrow Contains ', len(df['RainTomorrow'].unique()), 'labels')
print(df['RainTomorrow'].unique())
pd.get_dummies(df.RainTomorrow, drop_first=True, dummy_na=True).head()
pd.get_dummies(df.RainTomorrow, drop_first=True, dummy_na=True).sum(axis=0)

# find numerical value

numerical = [var for var in df.columns if df[var].dtype != object]
print('There are {} numericals variables \n '.format(len(numerical)))
print('The numericals variables are : ', numerical)
print(df[numerical].isnull().sum())

print(round(df[numerical].describe()), 2)

# visualisasi data

# plt.figure(figsize=(15, 10))
#
# plt.subplot(2, 2, 1)
# fig = df.boxplot(column='Rainfall')
# fig.set_title('')
# fig.set_ylabel('Rainfall')
#
# plt.subplot(2, 2, 2)
# fig = df.boxplot(column='Evaporation')
# fig.set_title('')
# fig.set_ylabel('Evaporation')
#
# plt.subplot(2, 2, 3)
# fig = df.boxplot(column='WindSpeed9am')
# fig.set_title('')
# fig.set_ylabel('WindSpeed9am')
#
# plt.subplot(2, 2, 4)
# fig = df.boxplot(column='WindSpeed3pm')
# fig.set_title('')
# fig.set_ylabel('WindSpeed3pm')
#
# plt.show()

# plt.figure(figsize=(15, 10))
#
# plt.subplot(2, 2, 1)
# fig = df.Rainfall.hist(bins=10)
# fig.set_xlabel('Rainfall')
# fig.set_ylabel('RainTommorow')
#
# plt.subplot(2, 2, 2)
# fig = df.Evaporation.hist(bins=10)
# fig.set_xlabel('Evaporation')
# fig.set_ylabel('RainTommorow')
#
# plt.subplot(2, 2, 3)
# fig = df.WindSpeed9am.hist(bins=10)
# fig.set_xlabel('WindSpeed9am')
# fig.set_ylabel('RainTommorow')
#
# plt.subplot(2, 2, 4)
# fig = df.WindSpeed3pm.hist(bins=10)
# fig.set_xlabel('WindSpeed3pm')
# fig.set_ylabel('RainTommorow')
#
# plt.show()


# find outlier using interquartile

iqr = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
low = df.Rainfall.quantile(0.25) - (iqr * 3)
up = df.Rainfall.quantile(0.75) + (iqr * 3)
print('Rainfall Outliers are values < {Low} or > {up}'.format(Low=low, up=up))

iqr = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
low = df.Evaporation.quantile(0.25) - (iqr * 3)
up = df.Evaporation.quantile(0.75) + (iqr * 3)
print('Evaporation Outliers are values < {low} or > {up}'.format(
    low=low, up=up))

iqr = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
low = df.WindSpeed9am.quantile(0.25) - (iqr * 3)
up = df.WindSpeed9am.quantile(0.75) + (iqr * 3)
print('WindSpeed9am Ouliers are values < {low} or > {up}'.format(
    low=low, up=up))

iqr = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
low = df.WindSpeed3pm.quantile(0.25) - (iqr * 3)
up = df.WindSpeed3pm.quantile(0.75) + (iqr * 3)
print('WindSpeed3pm Ouliers are values < {low} or > {up}'.format(
    low=low, up=up))

X = df.drop(['RainTomorrow'], axis=1)
y = df['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

print(X_train.shape, X_test.shape)
print(X_train.dtypes)

categorical = [col for col in X_train.columns if X_train[col].dtype == object]
print('There are {} categories variables \n '.format(len(categorical)))
print('The categories are : ', categorical)
print(df[categorical])

numerical = [col for col in X_train.columns if X_train[col].dtype != object]
print('There are {} categories variables \n '.format(len(numerical)))
print('The categories are : ', numerical)
print(df[numerical])

print(X_test[numerical].isnull().sum())

# mengisi data yang kosong dengan menggunakan median
for df1 in [X_train, X_test]:
    for col in numerical:
        col_median = X_train[col].median()
        df1[col].fillna(col_median, inplace=True)

print(X_test[numerical].isnull().sum())

print(X_test[categorical].isnull().sum())

# mengisi data yang kosong dengan menggunakan modus
for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)

print(X_test[categorical].isnull().sum())

print("Cek Disini")
print(X_train.isnull().sum())


def max_value(df3, variable, top):
    return np.where(df3[variable] > top, top, df3[variable])


for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)

encoder = ce.BinaryEncoder(cols=['RainToday'])
print(encoder.fit_transform(X_train))
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

print(X_train.head())

X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location),
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)])

print(X_train.head())

X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                    pd.get_dummies(X_test.Location),
                    pd.get_dummies(X_test.WindGustDir),
                    pd.get_dummies(X_test.WindDir9am),
                    pd.get_dummies(X_test.WindDir3pm)])
print(X_test.head())

scaler = MinMaxScaler()

cols = X_train.columns

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

y_train.fillna(y_train.mode()[0], inplace=True)
y_test.fillna(y_train.mode()[0], inplace=True)

print("This Is X_Train")
print(X_train)
print(X_test.describe())

logreg = LogisticRegression(solver='liblinear', random_state=0)

logreg.fit(X_train, y_train)
y_pred_test = logreg.predict(X_test)
