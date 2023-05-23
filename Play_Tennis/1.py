import joblib
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # data visualization
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# fitting model
from sklearn import tree
# to show graphic from tree
import graphviz
from sklearn.metrics import confusion_matrix, accuracy_score
# it ignores the warnings which may come up they are not important.
import warnings

warnings.filterwarnings('ignore')

Le = LabelEncoder()

# loading the PlayTennis data
PlayTennis = pd.read_csv("PlayTennis.csv")

# find the null value dataset
print(PlayTennis.isnull().sum())

# get unique label from dataset
print(PlayTennis['outlook'].unique())
print(PlayTennis['temp'].unique())
print(PlayTennis['humidity'].unique())
print(PlayTennis['windy'].unique())
print(PlayTennis['play'].unique())

# using label encoder
PlayTennis['outlook'] = Le.fit_transform(PlayTennis['outlook'])
PlayTennis['temp'] = Le.fit_transform(PlayTennis['temp'])
PlayTennis['humidity'] = Le.fit_transform(PlayTennis['humidity'])
PlayTennis['windy'] = Le.fit_transform(PlayTennis['windy'])
PlayTennis['play'] = Le.fit_transform(PlayTennis['play'])

y = PlayTennis['play']
x = PlayTennis.drop(['play'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.9)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

# visualize the tree using tree plot
tree.plot_tree(clf)

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
print(graph)

x_pred = clf.predict(x_test)
print(x_test)

confmat1 = confusion_matrix(x_pred, y_test)
print(accuracy_score(x_pred, y_test))

joblib.dump(clf, "playTennis.pkl")
model = joblib.load("playTennis.pkl")

outlook = int(input("Enter Outlook : "))
wind = int(input("Enter Wind : "))
temp = int(input("Enter temp : "))
humidity = int(input("Enter humidity : "))

play = model.predict([[outlook, wind, temp, humidity]])[0]

if play == 1:
    print("You Can Play")
else:
    print("Weather is not fit to play tennis")
