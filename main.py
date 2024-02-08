import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import model_selection, neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
import warnings
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("BodyFitnessPrediction.csv")
# Kolona sa datumima ne pridaje nikakav znacaj tako da je izbacujemo
df = df.drop(df.columns[0], axis=1)
# print(df)
# Eksploracija, kako izgleda nasa tabela zapravo kakve su vrednosti itd.
# print(df.shape)
# print(df.columns)
# svi podacu su popunjeni, nema null vrednosti
# print(df.info(),df.count(),df.dtypes)
# print(df['mood'].unique(), df['bool_of_active'].unique(), df['weight_kg'].unique())
# print(df['mood'].unique())
# print('Statistika:',df.describe())
# corr = df
# sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
# sns.regplot(x='step_count', y='calories_burned', data=df,color='r')
# sns.regplot(x='weight_kg', y='mood', data=df,color='g')
# df['step_count'].hist()
# df['calories_burned'].hist()
# df['weight_kg'].hist()
# df['hours_of_sleep'].hist()
# Jos jednom proveravamo da li ima nedostajucih vrednosti nam to ne bi negativno uticalo na nasu procenu
df.isnull().any()

# Detekcija anomalija
# df.boxplot(column="step_count")
# df.boxplot(column="calories_burned")
# Zakljucujemo da nema anomailija tako da nije potrebno da ih se resimo

# Vrednosti kolone 'Mood' i 'bool_of_active' nam se ne svidja kako izgledaju tako da enkodiramo njihve trenutne u neke
# bolje vrednosti (numericke)
le = LabelEncoder()
df['bool_of_active'] = le.fit_transform(df['bool_of_active'])
df['mood'] = le.fit_transform(df['mood'])

# Zavisne i nezavisne promenjive
x = df.iloc[:, [0, 1, 2, 3, 5]]  # Nezavisne
y = df.iloc[:, 4]  # Zavisna (bool_of_active), promenljiva koju predvdjamo

# Koristimo OneHot Enocding da bi stavili vrednosti u binarne jer to zna da poboljska performanse modela veoma cesto,
# naravno ovo nije neophodno
oh = OneHotEncoder()
x = oh.fit_transform(x).toarray()
x = x[:, 1:]

# Delimo podatke u testne i one za treniranje
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
# Jako je bitno da normalizujemo podatke da bi bili u slicnim opsezima da ne bi neka kolona uticala na model mnogo
# vise nego ostale
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)
print("Tacnost modela KNN je:", "%.2f" % (accuracy_score(y_test, y_predict) * 100), "%")
print("Preciznost je:", "%.2f" % (precision_score(y_test, y_predict) * 100), "%")
print("F1 je: ", "%.2f" % (f1_score(y_test, y_predict) * 100), "%") #F1 ocena (F1 score) je mera tačnosti modela koja kombinuje preciznost (precision) i odziv (recall) u jedan broj
print("Matrica konfuzija: \n", confusion_matrix(y_test, y_predict))
print('##############################################')

logisticka = LogisticRegression()
logisticka.fit(x_train, y_train)
y_predict = logisticka.predict(x_test)
print("Tacnost modela Logistic Regression je:", "%.2f" % (accuracy_score(y_test, y_predict) * 100), "%")
print("Preciznost je:", "%.2f" % (precision_score(y_test, y_predict) * 100), "%")
print("F1 je: ", "%.2f" % (f1_score(y_test, y_predict) * 100), "%")
print("Matrica konfuzija: \n", confusion_matrix(y_test, y_predict))
print('##############################################')

stablo_odlucivanja = DecisionTreeClassifier()
stablo_odlucivanja.fit(x_train, y_train)
y_predict = stablo_odlucivanja.predict(x_test)
print("Tacnost modela DecisionTreeClassifier je:", "%.2f" % (accuracy_score(y_test, y_predict) * 100), "%")
print("Preciznost je:", "%.2f" % (precision_score(y_test, y_predict) * 100), "%")
print("F1 je: ", "%.2f" % (f1_score(y_test, y_predict) * 100), "%")
print("Matrica konfuzija: \n", confusion_matrix(y_test, y_predict))
print('##############################################')

stablo_odlucivanja = RandomForestClassifier()
stablo_odlucivanja.fit(x_train, y_train)
y_predict = stablo_odlucivanja.predict(x_test)
print("Tacnost modela Random Forest Classifier je:", "%.2f" % (accuracy_score(y_test, y_predict) * 100), "%")
print("Preciznost je:", "%.2f" % (precision_score(y_test, y_predict) * 100), "%")
print("F1 je: ", "%.2f" % (f1_score(y_test, y_predict) * 100), "%")
print("Matrica konfuzija: \n", confusion_matrix(y_test, y_predict))

plt.show()
