# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:58:00 2023

@author: Krithika
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("insurance.csv")
df.head()

#Exploratory data analysis
df.describe()
#from this we 
df.info()
df.shape
df.columns
df.isnull().sum()
df.duplicated().sum()

df.region.value_counts()
df.age.value_counts()
df.children.value_counts()

df = df.drop('index', axis=1)

#distributed chart for age
fig, ax = plt.subplots()
sns.kdeplot(data=df, x='age', color='red', ax = ax)
ax.set_xlabel('age')
ax.set_title('Distribution of Age')
plt.show()

#this plot is built to see the relationship between age and charges as
#younger people can pay less compared to those of older ones
fig,ax = plt.subplots()
sns.scatterplot(data=df,x='age',y='charges',ax=ax)
plt.show()
#here we can see a linear relationship between age and charges in a particular segment
#the reason for this may be reflected by the lifestyle of the person(health)
#from the coloumns we can take smokers and bmi to be parameters

fig,ax = plt.subplots()
sns.scatterplot(data=df,x='age',y='charges',ax=ax,hue='smoker')
plt.show()
#people who are not smokers have less insurance charges than those who smoke

fig,ax = plt.subplots()
sns.scatterplot(data=df,x='age',y='charges',ax=ax,hue='bmi')
plt.show()
#we cant really infer from age and bmi

#count of each sex
df.sex.value_counts()
#almost the same

#average bmi for each sex
fig, ax = plt.subplots() 
sns.barplot(data=df, x='sex', y='bmi', ci=None, ax=ax, palette='Set1')
ax.bar_label(ax.containers[0])
ax.set_ylim((0, 50))
ax.set_title("Average BMI for each Sex")
plt.show()
#both male and female have almost same bmi

#proportion of smokers
df['smoker'] = df['smoker'].apply(lambda x: 1 if x=='yes' else 0)
df.groupby('sex')['smoker'].mean().to_frame().reset_index().rename(columns={'smoker': 'Proportion of Smokers'})
#men is slightly higher than women in terms of smoking proportion

#sex with respect to charges
fig,ax = plt.subplots(1,2,figsize = (8,4))
sns.histplot(data = df[df['sex']=='male'],x='charges',kde=True,ax= ax[0],color='purple')
ax[0].set_title('Males')
sns.histplot(data = df[df['sex']=='female'],x='charges',kde=True,ax= ax[1],color='red')
ax[1].set_title('Female')
plt.show()

#sex with rescpect to bmi
fig, ax = plt.subplots()
sns.kdeplot(data=df, x='bmi', color='brown', ax = ax)
ax.set_xlabel('BMI')
ax.set_title('Distribution of BMI')
plt.show()
#the histogram is perfect curve hence the proportion is equal

#average bmi for smokers and non smokers
#barplot
fig, ax = plt.subplots() 
sns.barplot(data=df, x='smoker', y='bmi', ax=ax, palette='Set1', ci=None) 
ax.set_title("Average BMI for Smoker and Non-Smoker")
ax.set_xlabel("Smoker(1-yes, 0-no)")
ax.set_ylabel("BMI")
ax.bar_label(ax.containers[0])
ax.set_ylim((0, 45))
plt.show()

#scatterplot(bmi vs charges)
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='bmi', y='charges', ax=ax)
ax.set_title('bmi vs charges')
plt.show()
#there was a band of area where the scatter points was concentrated 

#check with respect to smokers(filter)
fig,ax = plt.subplots()
sns.scatterplot(data=df,x='bmi',y='charges',ax=ax,hue='smoker')
plt.show()
# here we can notice that the non smoker that pay less have a band of people in its region

df.children.value_counts()
#most people do not have children

#average number of people with children in different regions
fig, ax = plt.subplots() 
sns.barplot(data=df, x='children', y='region', ci=None, ax=ax, palette='Set1')
ax.set_xlim((0, 1.5))
ax.set_title("Average children for each region")
plt.show()

#distribution of charges with number of children
fig, ax = plt.subplots()
sns.kdeplot(data=df, x='charges', hue='children',color='brown', ax = ax)
ax.set_xlabel('charges')
ax.set_title('Distribution of charges over number of children')
plt.show()

df.region.value_counts()
#almost all the 4 regions are the same

fig, ax = plt.subplots() 
sns.barplot(data=df, x='charges', y='r', ci=None, ax=ax, palette='Set1')
#ax.set_xlim((0, 200))
ax.set_title("Average charge for each region")
plt.show()
#here south east region has the max charges

#distribution of charges
fig, ax = plt.subplots()
sns.kdeplot(data=df,x='charges',color = 'purple',ax=ax)
ax.set_title('Distribution of charges')
ax.set_xlabel('Charges')
plt.show()

fig, ax = plt.subplots()
sns.kdeplot(data=df, x='charges', color='brown', ax = ax)
ax.set_xlabel('Charge')
ax.set_title('Distribution of Charge')
plt.show()



