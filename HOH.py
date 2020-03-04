import pandas as pd

data  = pd.read_excel("E:\data\HOH bannerghatta sales data form 2 (1).xlsx", index = False)

data.head()

data.isnull().sum()

data.rename(columns = {"typeofsociety" : "TypeofSociety"}, inplace = True)                      

df = data[["BookingDate", "TypeofSociety"]]

df.head()

df.TypeofSociety.fillna(method = "ffill",limit = 5, inplace = True)

df.TypeofSociety.fillna(method = "bfill",limit = 5, inplace = True)

df.BookingDate.fillna(method = "bfill",limit = 5, inplace = True)

df.BookingDate.fillna(method = "ffill",limit = 5, inplace = True)

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df.TypeofSociety = le.fit_transform(df.TypeofSociety)

df.head()

import matplotlib.pyplot as plt

df.plot()

plt.ylabel("Type of society")
                                                    

df.set_index('BookingDate',inplace=True)

df1 = df.iloc [:, [1]] 
df1

df1.head()

data.apartmentname = le.fit_transform(data.apartmentname)

data.typeofsociety = le.fit_transform(data.typeofsociety)

data.BookingCompleted = le.fit_transform(data.BookingCompleted)

data.isnull().sum()

x = data[["apartmentname", "typeofsociety"]]
y = data["BookingCompleted"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state = 1 )


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
dtree = clf.fit(x_train, y_train)
pred = dtree.predict(x_test)
pred


from sklearn import metrics
a = metrics.confusion_matrix(y_test,pred)
print("CONFUSION MATRIX: \n" ,a )
b = metrics.accuracy_score(y_test, pred)
print("ACCURACY OF THE MODEL IS :\n", b)

import seaborn as sns

sns.pairplot(data, hue = "BookingDate")

sns.pairplot(data, hue = "BookingCompleted")

data['typeofsociety'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='green')

data['BookingDate'].value_counts(normalize=True).plot(figsize=(10,12),kind='bar',color='yellow')

date = data.groupby(['BookingDate','typeofsociety']).size().unstack().plot.bar().figsize=(20,15)
