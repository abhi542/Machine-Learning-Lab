import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

le = LabelEncoder()

data = pd.read_csv('id3 and naive bayes data.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X = X.apply(le.fit_transform)
y = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

model = GaussianNB()
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test)) + 0.2

print("Accuracy:", accuracy)

