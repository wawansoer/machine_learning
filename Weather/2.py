from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
from sklearn.model_selection import train_test_split
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # data visualization
import seaborn as sns  # statistical data visualization

# it ignores the warnings which may come up they are not important.
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('weather.csv')

col_names = df.columns

# find categorical variables
categorical = [var for var in df.columns if df[var].dtype == 'O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :', categorical)

# check missing values in categorical variables
print(df[categorical].isnull().sum())

# print categorical variables containing missing values
cat1 = [var for var in categorical if df[var].isnull().sum() != 0]
print(df[cat1].isnull().sum())

# view frequency of categorical variables
for var in categorical:
    print(df[var].value_counts())

# view frequency distribution of categorical variables
for var in categorical:
    print(df[var].value_counts()/np.float64(len(df)))

# check for cardinality in categorical variables
for var in categorical:
    print(var, ' contains ', len(df[var].unique()), ' labels')

# parse the dates, currently coded as strings, into datetime format
df['Date'] = pd.to_datetime(df['Date'])

# extract year from date
df['Year'] = df['Date'].dt.year

# extract month from date
df['Month'] = df['Date'].dt.month

# extract day from date
df['Day'] = df['Date'].dt.day

# drop the original Date variable
df.drop('Date', axis=1, inplace=True)

# find categorical variables
categorical = [var for var in df.columns if df[var].dtype == 'O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :', categorical)

# check for missing values in categorical variables
print(df[categorical].isnull().sum())

# print number of labels in Location variable
print('Location contains', len(df.Location.unique()), 'labels')

# check labels in location variable
print(df.Location.unique())

# check frequency distribution of values in Location variable
print(df.Location.value_counts())

# let's do One Hot Encoding of Location variable
# get k-1 dummy variables after One Hot Encoding
# preview the dataset with head() method
df.location = pd.get_dummies(df.Location, drop_first=True)
print(df.location)

# print number of labels in WindGustDir variable
print('WindGustDir contains', len(df['WindGustDir'].unique()), 'labels')

# check labels in WindGustDir variable
print(df['WindGustDir'].unique())

# check frequency distribution of values in WindGustDir variable
print(df.WindGustDir.value_counts())

# let's do One Hot Encoding of WindGustDir variable
# get k-1 dummy variables after One Hot Encoding
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method
print(pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True))

# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category
pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0)

# print number of labels in WindDir9am variable
print('WindDir9am contains', len(df['WindDir9am'].unique()), 'labels')

# check labels in WindDir9am variable
print(df['WindDir9am'].unique())

# check frequency distribution of values in WindDir9am variable
df['WindDir9am'].value_counts()

# let's do One Hot Encoding of WindDir9am variable
# get k-1 dummy variables after One Hot Encoding
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method
print(pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head())

# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

print(pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0))
# drop_first=True is important to use,
# as it helps in reducing the extra column created during dummy variable creation. Hence it reduces the correlations created among dummy variables.

# print number of labels in WindDir3pm variable
print('WindDir3pm contains', len(df['WindDir3pm'].unique()), 'labels')

# check labels in WindDir3pm variable
print(df['WindDir3pm'].unique())

# check frequency distribution of values in WindDir3pm variable
print(df['WindDir3pm'].value_counts())

# let's do One Hot Encoding of WindDir3pm variable
# get k-1 dummy variables after One Hot Encoding
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method
print(pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head())

# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category
print(pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0))

# print number of labels in RainToday variable
print('RainToday contains', len(df['RainToday'].unique()), 'labels')

# check labels in WindGustDir variable
df['RainToday'].unique()

# let's do One Hot Encoding of RainToday variable
# get k-1 dummy variables after One Hot Encoding
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method
print(pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head())

# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category
pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).sum(axis=0)

# find numerical variables
numerical = [var for var in df.columns if df[var].dtype != 'O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)

# view the numerical variables
print(df[numerical].head())

# check missing values in numerical variables
print(df[numerical].isnull().sum())

# view summary statistics in numerical variables
print(round(df[numerical].describe()), 2)

# draw boxplots to visualize outliers
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')

plt.subplot(2, 2, 2)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')

plt.subplot(2, 2, 3)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')

plt.subplot(2, 2, 4)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')

# plot histogram to check distribution
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')

plt.subplot(2, 2, 2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')

plt.subplot(2, 2, 3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')

plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')
# plt.show()

# find outliers for Rainfall variable
IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(
    lowerboundary=Lower_fence, upperboundary=Upper_fence))

# find outliers for Evaporation variable
IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(
    lowerboundary=Lower_fence, upperboundary=Upper_fence))

# find outliers for WindSpeed9am variable
IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(
    lowerboundary=Lower_fence, upperboundary=Upper_fence))

# find outliers for WindSpeed3pm variable
IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(
    lowerboundary=Lower_fence, upperboundary=Upper_fence))

X = df.drop(['RainTomorrow'], axis=1)

y = df['RainTomorrow']

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# check the shape of X_train and X_test
print(X_train.shape, X_test.shape)

# check data types in X_train
print(X_train.dtypes)

# display categorical variables
categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
print(categorical)

# display numerical variables
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']
print(numerical)

# check missing values in numerical variables in X_train
print(X_train[numerical].isnull().sum())

# check missing values in numerical variables in X_test
print(X_test[numerical].isnull().sum())

# print percentage of missing values in the numerical variables in training set
for col in numerical:
    if X_train[col].isnull().mean() > 0:
        print(col, round(X_train[col].isnull().mean(), 4))

# impute missing values in X_train and X_test with respective column median in X_train
for df1 in [X_train, X_test]:
    for col in numerical:
        col_median = X_train[col].median()
        df1[col].fillna(col_median, inplace=True)

# check again missing values in numerical variables in X_train
print(X_train[numerical].isnull().sum())

# check missing values in numerical variables in X_test
print(X_test[numerical].isnull().sum())

# print percentage of missing values in the categorical variables in training set
print(X_train[categorical].isnull().mean())

# print categorical variables with missing data
for col in categorical:
    if X_train[col].isnull().mean() > 0:
        print(col, (X_train[col].isnull().mean()))

# impute missing categorical variables with most frequent value
for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)

# check missing values in categorical variables in X_train
print(X_train[categorical].isnull().sum())

# check missing values in categorical variables in X_test
print(X_test[categorical].isnull().sum())

# check missing values in X_train
print(X_train.isnull().sum())

# check missing values in X_test
print(X_test.isnull().sum())


def max_value(df3, variable, top):
    return np.where(df3[variable] > top, top, df3[variable])


for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)

print(X_train.Rainfall.max(), X_test.Rainfall.max())
print(X_train.Evaporation.max(), X_test.Evaporation.max())
print(X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max())
print(X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max())
print(X_train[numerical].describe())
print(categorical)
print(X_train[categorical].head())

# encode RainToday variable
encoder = ce.BinaryEncoder(cols=['RainToday'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

print(X_train.head())

X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location),
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)
print(X_train.head())

X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                    pd.get_dummies(X_test.Location),
                    pd.get_dummies(X_test.WindGustDir),
                    pd.get_dummies(X_test.WindDir9am),
                    pd.get_dummies(X_test.WindDir3pm)], axis=1)
print(X_test.head())

print(X_train.describe())
cols = X_train.columns
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

print(X_test.head())
print(X_train.describe())

y_train.fillna(y_train.mode()[0], inplace=True)
y_test.fillna(y_train.mode()[0], inplace=True)

# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)

# fit the model
logreg.fit(X_train, y_train)

# predict result
y_pred_test = logreg.predict(X_test)
print(y_pred_test)

# probability of getting output as 0 - no rain
logreg.predict_proba(X_test)[:, 0]

# probability of getting output as 1 - rain
logreg.predict_proba(X_test)[:, 1]

print('Model accuracy score: {0:0.4f}'. format(
    accuracy_score(y_test, y_pred_test)))

y_pred_train = logreg.predict(X_train)

print(y_pred_train)
print(
    'Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

# print the scores on training and test set
print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))

# check class distribution in test set
y_test.value_counts()

# check null accuracy score
null_accuracy = (22067/(22067+6372))
print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

# Print the Confusion Matrix and slice it into four pieces
cm = confusion_matrix(y_test, y_pred_test)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0, 0])
print('\nTrue Negatives(TN) = ', cm[1, 1])
print('\nFalse Positives(FP) = ', cm[0, 1])
print('\nFalse Negatives(FN) = ', cm[1, 0])
