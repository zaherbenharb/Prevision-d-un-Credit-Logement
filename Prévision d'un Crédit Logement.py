#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[3]:


df_credit = pd.read_csv('train.csv')


# In[4]:


df_credit.head()


# In[5]:


df_credit.info()


# In[ ]:





# In[6]:


df_credit.shape    


# In[7]:


#verifier si on a des valeur manquantes
df_credit.isnull().sum()


# In[8]:


df_credit.columns


# In[9]:


#supprimer les valeurs manquantes 
df_credit['Gender'].fillna(df_credit['Gender'].mode()[0], inplace=True)
df_credit['Married'].fillna(df_credit['Married'].mode()[0], inplace=True)
df_credit['Dependents'].fillna(df_credit['Dependents'].mode()[0], inplace=True)
df_credit['Self_Employed'].fillna(df_credit['Self_Employed'].mode()[0], inplace=True)
df_credit['Credit_History'].fillna(df_credit['Credit_History'].mode()[0], inplace=True)


# In[10]:


df_credit['Loan_Amount_Term'].fillna(df_credit['Loan_Amount_Term'].median(), inplace=True)
df_credit['LoanAmount'].fillna(df_credit['LoanAmount'].median(), inplace=True)


# In[11]:


df_credit.isnull().sum()


# In[12]:


var_cat=['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Credit_History','Loan_Status']
var_num=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term']


# In[14]:


df_credit['Education'].value_counts()


# In[198]:


df_credit['Loan_Status'].value_counts()


# In[199]:


df_credit['Loan_Status'].value_counts(normalize=True)*100


# In[200]:


df_credit['Loan_Status'].value_counts(normalize=True).plot.bar('title:creditaccordé')


# In[201]:


df_credit['Gender'].value_counts(normalize=True)*100


# In[202]:


df_credit['Gender'].value_counts().plot.bar()


# In[203]:


df_credit[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term']].describe()


# In[204]:


#Dataviz les variavbles catégoriques

fig,axes=plt.subplots(4,2,figsize=(15,12))
for idx,cat_col in enumerate(var_cat):
    row,col=idx//2,idx%2
    sns.countplot(x=cat_col,data=df_credit,hue='Loan_Status',ax=axes[row,col])


# In[205]:


#correlation des variables numérique
matrix=df_credit.corr()
f,ax=plt.subplots(figsize=(10,12))
sns.heatmap(matrix,vmax=.8,square=True,cmap='BuPu',annot=True)


# # Création du Modèle

# In[206]:


df_cat=df_credit[var_cat]
df_cat


# In[207]:


df_cat=pd.get_dummies(df_cat,drop_first=True)
df_cat


# In[208]:


df_num=df_credit[var_num]
df_encoded=pd.concat([df_cat,df_num],axis=1)
df_encoded


# In[209]:


Y=df_encoded['Loan_Status_Y']
X=df_encoded.drop('Loan_Status_Y',axis=1)
X


# In[210]:


#spécifier la partie test et la partie train
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=6)


# In[211]:


#instancier le model
clf=LogisticRegression()
#entrainer le model
clf.fit(X_train,Y_train)


# In[213]:


pred=clf.predict(X_test)
pred


# In[214]:


accuracy_score(Y_test,pred)


# In[215]:


X.columns


# In[216]:


#faire une prévision
profil_test=[[1,1,1,0,0,0,1,0,1,0,100,0,400,360,]]
clf.predict(profil_test)


# In[217]:


#enregister la modèle
import pickle
pickle.dump(clf,open('Prévision_d_un Crédit Logement.pkl','wb'))


# In[ ]:




