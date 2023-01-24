# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 16:42:55 2023

@author: Krithika
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
#import wordcloud
#from wordcloud import WordCloud
#from matplotlib import colors

df = pd.read_csv("bestsellers with categories.csv")
df.head()

#Exploratory data analysis
df.describe() #statistical properties of every numbers
df.info()
df.shape    #(rows, columns)
df.nunique()
df.isnull().sum()
df.columns

df.rename(columns={"User Rating": "User_Rating"}, inplace=True)
df.Name.value_counts()
df.User_Rating.value_counts()
df['User_Rating'].value_counts() # counting the number of times each rating is repeted
df.Genre.value_counts()
df[["User_Rating", "Genre"]]

#Data Visualization
#1)
plt.figure(figsize=(10,8))
sns.histplot(df["User_Rating"], color="purple")
plt.title("user rating", fontdict={'fontweight':'bold', "fontsize":22})
plt.xlabel("ratings")
plt.ylabel("count")
plt.show()
#books with 4.8 rating are more popular

#2)
plt.figure(figsize=(12,6))
labels=["Fiction","Non Fiction"]
colors=["grey","yellow"]
plt.pie(df['Genre'].value_counts(), labels=labels, colors=colors,  autopct="%.2f%%")
plt.title("Genre",fontdict={"fontweight": "bold", "fontsize":20})
plt.show()
#fiction books are more popular than non-fiction books by 56.36%

#3)
plt.figure(figsize=(12,6))
sns.scatterplot(x="Year",y="Reviews", data=df, hue="Genre")
plt.title("Number of Reviews Per Year",fontdict={"fontweight": "bold", "fontsize":20})
plt.xlabel("Year",fontdict={"fontsize":20})
plt.ylabel("Reviews", fontdict={"fontsize":20})
plt.show()
#reviews of book  y genre is highest with respect to fiction books over non-fiction ones

#4)
df1 = df[df['User_Rating']==4.9]
df1.head()

x= df1.groupby("Name").Reviews.mean().sort_values(ascending=False)
plt.figure(figsize=(10,10))
sns.set_style("darkgrid")
sns.barplot(x=x.values,y=x.index,)
plt.xlabel("Reviews", fontdict={"fontsize":15})
plt.ylabel("Names", fontdict={"fontsize":15})
plt.title( "Books with 4.9 Rating ", fontdict={"fontweight":"bold", "fontsize":22})
plt.show()
#reviews of every book with 4.9 rating

#5)
plt.figure(figsize=(12,6))
sns.barplot(x="Year", y="Price", data= df, hue="Genre")
plt.title("Price Of Books/ Year", fontdict={"fontweight": "bold", "fontsize":22})
plt.xlabel("year", fontdict={ "fontsize":20})
plt.ylabel("Price", fontdict={"fontsize":20})
plt.show()
#prices of books both ficture and non-fiction are plotted
"""
#6) which author has most books
g=sns.pairplot(
    df,
    x_vars=["User_Rating", "Reviews", "Price"],
    y_vars=["Year"],    plot_kws=dict(marker="+", linewidth=1),
    palette="ocean", hue="Genre"
)
g.fig.set_size_inches(15,15)
plt.show()

#7)
df.Author.value_counts()
plt.figure(figsize=(20 ,10 ))
plt.xticks(rotation=90)
sns.countplot(data = df , x =df['Author'] ,palette='PuBu',order=df['Author'].value_counts().index[0:50])
plt.show()

#8)Most visited books  (doubt))
oldest_us_series=df.sort_values(by='Reviews',ascending=False)[0:550]

fig = go.Figure(data=[go.Table(header=dict(values=['Name', 'Reviews'],fill_color='lightsalmon'),cells=dict(values=[oldest_us_series['Name'],oldest_us_series['Reviews']],fill_color='light blue'))])
fig.show()
"""

#no of fiction and non fiction               -----
#books published after a particular year ---
#books having a higher rating  ---
#books having a high review  ---
# no of books published by each author --- (sort it ascending)
# cheap books having high ratings ---
#higher rating book are they fiction or non fiction ---
# author with most high rating books ---
#highest rated fiction ---
#highest rated non fiction  ---
#plot per rating
#price of books 
#lowest rated book of a particular author
#display books wrt their rating 
# display no of fiction and non fiction       -----


import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("bestsellers with categories.csv")
"""
#1
fiction_count=df['Genre'].value_counts()['Fiction']
nonfiction_count=df['Genre'].value_counts()['Non Fiction']
plotting=pd.DataFrame({'lable':['fiction','non-fiction'],'value':[fiction_count,nonfiction_count]})
plotting.plot.bar(x='lable',y='value',title='No of fiction/non fiction',width=0.1,grid=True,xlabel="Genre",ylabel="No of books")
value=[fiction_count,nonfiction_count]
lable=['fiction','non-fiction']
print(fiction_count,nonfiction_count)
#plt.bar(lable,value)"""


#grouping
def check(grp):
    return grp.mean()
print(df.groupby('Genre')['User Rating'].apply(check))
#df.groupby('Genre')['User Rating'].mean()
print(df.groupby(['Genre','Year'])['User Rating'].mean())
print(df.groupby(['Genre','Year'])['User Rating'].agg(['mean','median']))
print(df.groupby(['Genre','Year'],as_index=False)['User Rating'].mean())


#2
print(df[df['Year']>=2018])

#3
print(df[df['User Rating']>=4.8])
no=df['User Rating'].value_counts()
print(df.groupby('User Rating').size())
print(no)
#plt.bar(df['User Rating'],df['User Rating'].value_counts())

#4
ratings=df.sort_values(by=['User Rating','Reviews'])
print(ratings.tail(1))

#5
print(df.sort_values(by=['Author']).groupby('Author').size())

#6
print(df.sort_values(by=['Price','User Rating'],ascending=['True','False']).head(1))

#7
print(df.sort_values(by=['User Rating']).groupby('Genre').last())

#8
print(df.sort_values(by=['Author']).groupby('Author').size())




"""
pd.set_option('display.max_columns', None)
print(df.head())
print(df[df['User Rating']>=4.8])
print(df.isna().sum())
print(df.describe())
print(df.dtypes)
df.replace('Fiction',1,inplace=True)
df.loc[df['Genre']=='Non Fiction','Genre']=2
print(df.head(10))"""



#concusion: fiction books are more popular among readers and one of the reasons is due to its lesser price
#which can be seen from the graph