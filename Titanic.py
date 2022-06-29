import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

data = pd.read_csv('tested.csv')
logr = LogisticRegression()

data['Age'].fillna(data['Age'].mean(), inplace=True)
data.drop('Cabin',axis=1)

le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
data['Pclass'] = le.fit_transform(data['Pclass'])
data['Embarked'] = le.fit_transform(data['Embarked'])
data.dropna(inplace = True)


data = pd.concat([data,data['Sex'],data['Embarked']], axis = 1)

x = data.drop(["PassengerId", 'Name', 'Ticket', 'Cabin', 'Survived','Age','Fare','Parch','SibSp'], axis=1)
y = data['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3)

logr.fit(x_train, y_train)
y_pred = logr.predict(x_test)
print(accuracy_score(y_test, y_pred))