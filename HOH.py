import pandas as pd

data  = pd.read_excel("E:\data\HOH bannerghatta sales data form 2 (1).xlsx", index = False)

data.head()

data.isnull().sum()

data.dtypes

data.describe()

data.typeofsociety.fillna(method = "ffill",limit = 10, inplace = True)

data.BookingDate.fillna(method = "bfill",limit = 10, inplace = True)

data.BookingDate.fillna(method = "ffill",limit = 10, inplace = True)

data.rename(columns = {"Booking Completed" : "BookingCompleted", "apartment name" : "apartmentname"}, inplace = True)                      

data.BookingCompleted.fillna("No", inplace = True)

data.BookingCompleted.value_counts()

data['typeofsociety'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='green')

data['BookingDate'].value_counts(normalize=True).plot(figsize=(10,12),kind='bar',color='yellow')

date = data.groupby(['BookingDate','typeofsociety']).size().unstack().plot.bar().figsize=(20,15)

import seaborn as sns

sns.pairplot(data, hue = "BookingDate")


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data.typeofsociety = le.fit_transform(df.typeofsociety)

data.dtypes

data.isnull().sum()

data.head()

data[["BookingDate", "typeofsociety"]]

data.set_index('BookingDate',inplace=True)

data.head()

df1 = data.iloc [:, [1]] 
df1

df1.head()

import matplotlib.pyplot as plt

df1.plot()

plt.ylabel("Type of society")
                                                    
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

