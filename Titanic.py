import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

df=pd.read_csv('tested.csv')

df=df.drop(['PassengerId', 'Cabin','Name','Ticket'],axis=1)

df['Fare'].fillna(df['Fare'].mean(),inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df['Sex'])
df['Sex']=le.transform(df['Sex'])

le = LabelEncoder()
le.fit(df['Embarked'])
df['Embarked']=le.transform(df['Embarked'])
df['Age'].fillna(df['Age'].median(),inplace=True)
df1=df.drop('Survived',axis=1)
et = ExtraTreesClassifier()
et.fit(df1,df['Survived'])

feat_imp=pd.Series(et.feature_importances_,index=df1.columns)
feat_imp.nlargest(7).plot(kind='barh')
# plt.show()
df=df.drop(['Age','Embarked','Pclass'],axis=1)

x=df.drop('Survived',axis=1)
y=df['Survived']

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df['SibSp']=pd.cut(df['SibSp'],2,labels=[0,1])

mnb=MultinomialNB()

x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=0, train_size=0.3)

mnb.fit(x_train,y_train)
y_pred=mnb.predict(x_test)
print("MultinomialNB: ",accuracy_score(y_test,y_pred))

'''
MultinomialNB:  0.825938566552901
'''
