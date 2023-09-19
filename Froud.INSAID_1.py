#!/usr/bin/env python
# coding: utf-8

# # Import Libraries
# 

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


from matplotlib.patches import Rectangle


# In[6]:


from pprint import pprint as pp


# In[7]:


from pathlib import Path


# In[8]:


import seaborn as sbn


# In[9]:


from itertools import product


# In[10]:


import string


# In[11]:


import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


# In[12]:


from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline 


# In[13]:


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import homogeneity_score, silhouette_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans, DBSCAN


# In[14]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import itertools
from collections import Counter
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# In[15]:


import gensim
from gensim import corpora


# In[16]:


from scipy import stats


# In[17]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[18]:


from sklearn.preprocessing import LabelEncoder


# # Load the Dataset
# 

# In[19]:


df = pd.read_csv("C:/Users/PCC/Downloads/Fraud (1).csv")


# In[20]:


df


# In[21]:


df.info()


# # Analysis

# # Handle Missing Values
# 
# 

# In[22]:


# Check for missing values and decide how to handle them. You can either remove rows with missing values or impute 
# them with appropriate values (mean, median, mode, etc.). To check for missing values:


df.isnull()


# In[23]:


df.shape


# In[24]:


df.corr()


# In[25]:


df.head()


# In[26]:


df.tail(5)


# In[27]:


df.describe()


# In[28]:


df.min()


# In[29]:


df.count()


# In[30]:


# To remove rows with missing values:

df.dropna()


# In[31]:


# Check for null values
df.isnull().values.any()


# In[32]:


legit = len(df[df.isFraud == 0])
fraud = len(df[df.isFraud == 1])
legit_percent = (legit / (fraud + legit)) * 100
fraud_percent = (fraud / (fraud + legit)) * 100

print("Number of Legit transactions: ", legit)
print("Number of Fraud transactions: ", fraud)
print("Percentage of Legit transactions: {:.4f} %".format(legit_percent))
print("Percentage of Fraud transactions: {:.4f} %".format(fraud_percent))


# In[33]:


# Merchants
X = df[df['nameDest'].str.contains('M')]
X.head()


# # VISUALISATION
# 

# In[34]:


corr=df.corr()

plt.figure(figsize=(10,6))
sbn.heatmap(corr,annot=True)


# # NUMBER OF LEGIT AND FRAUD TRANSACTIONS

# In[35]:


plt.figure(figsize=(4,8))
labels = ["Legit", "Fraud"]
count_classes = df.value_counts(df['isFraud'], sort= True)
count_classes.plot(kind = "bar", rot = 0)
plt.title("Visualization of Labels")
plt.ylabel("Count")
plt.xticks(range(2), labels)
plt.show()


# # PROBLEM SOLVING

# In[36]:


#creating a copy of original dataset to train and test models

new_df=df.copy()
new_df.head()


# In[37]:


# Checking how many attributes are dtype: object

objList = new_df.select_dtypes(include = "object").columns
print (objList)


# In[38]:


#Label Encoding for object to numeric conversion

le = LabelEncoder()

for feat in objList:
    new_df[feat] = le.fit_transform(new_df[feat].astype(str))

print (new_df.info())


# In[39]:


new_df.head()


# # MULTICOLINEARITY

# In[40]:


def calc_vif(df):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

    return(vif)

calc_vif(new_df)


# In[41]:


new_df['Actual_amount_orig'] = new_df.apply(lambda x: x['oldbalanceOrg'] - x['newbalanceOrig'],axis=1)
new_df['Actual_amount_dest'] = new_df.apply(lambda x: x['oldbalanceDest'] - x['newbalanceDest'],axis=1)
new_df['TransactionPath'] = new_df.apply(lambda x: x['nameOrig'] + x['nameDest'],axis=1)

#Dropping columns
new_df = new_df.drop(['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','step','nameOrig','nameDest'],axis=1)

calc_vif(new_df)


# In[42]:


corr=new_df.corr()

plt.figure(figsize=(10,6))
sbn.heatmap(corr,annot=True)


# # NORMALIZING AMOUNT
# 

# In[43]:


scaler = StandardScaler()
new_df["NormalizedAmount"] = scaler.fit_transform(new_df["amount"].values.reshape(-1, 1))
new_df.drop(["amount"], inplace= True, axis= 1)

Y = new_df["isFraud"]
X = new_df.drop(["isFraud"], axis= 1)


# # TRAIN-TEST SPLIT

# In[44]:


(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size= 0.3, random_state= 42)

print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)


# # MODEL TRAINIG

# In[45]:


# DECISION TREE

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)

Y_pred_dt = decision_tree.predict(X_test)
decision_tree_score = decision_tree.score(X_test, Y_test) * 100


# In[46]:


# RANDOM FOREST

random_forest = RandomForestClassifier(n_estimators= 100)
random_forest.fit(X_train, Y_train)

Y_pred_rf = random_forest.predict(X_test)
random_forest_score = random_forest.score(X_test, Y_test) * 100


# # EVALUATION

# In[48]:


# Print scores of our classifiers

print("Decision Tree Score: ", decision_tree_score)
print("Random Forest Score: ", random_forest_score)


# In[49]:


# key terms of Confusion Matrix - DT

print("TP,FP,TN,FN - Decision Tree")
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred_dt).ravel()
print(f'True Positives: {tp}')
print(f'False Positives: {fp}')
print(f'True Negatives: {tn}')
print(f'False Negatives: {fn}')


# In[52]:


# key terms of Confusion Matrix - DT

print("TP,FP,TN,FN - Decision Tree")
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred_dt).ravel()
print(f'True Positives: {tp}')
print(f'False Positives: {fp}')
print(f'True Negatives: {tn}')
print(f'False Negatives: {fn}')

print("----------------------------------------------------------------------------------------")

# key terms of Confusion Matrix - RF

print("TP,FP,TN,FN - Random Forest")
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred_rf).ravel()
print(f'True Positives: {tp}')
print(f'False Positives: {fp}')
print(f'True Negatives: {tn}')
print(f'False Negatives: {fn}')


# # Here Random Forest looks good.

# In[53]:


# confusion matrix - DT

confusion_matrix_dt = confusion_matrix(Y_test, Y_pred_dt.round())
print("Confusion Matrix - Decision Tree")
print(confusion_matrix_dt,)

print("----------------------------------------------------------------------------------------")

# confusion matrix - RF

confusion_matrix_rf = confusion_matrix(Y_test, Y_pred_rf.round())
print("Confusion Matrix - Random Forest")
print(confusion_matrix_rf)


# In[54]:


# classification report - DT

classification_report_dt = classification_report(Y_test, Y_pred_dt)
print("Classification Report - Decision Tree")
print(classification_report_dt)

print("----------------------------------------------------------------------------------------")

# classification report - RF

classification_report_rf = classification_report(Y_test, Y_pred_rf)
print("Classification Report - Random Forest")
print(classification_report_rf)


# # With Such a good precision and hence F1-Score, Random Forest comes out to be better as expected.

# In[55]:


# visualising confusion matrix - DT


disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_dt)
disp.plot()
plt.title('Confusion Matrix - DT')
plt.show()

# visualising confusion matrix - RF
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_rf)
disp.plot()
plt.title('Confusion Matrix - RF')
plt.show()


# In[56]:


# AUC ROC - DT
# calculate the fpr and tpr for all thresholds of the classification

fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_pred_dt)
roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC - DT')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# AUC ROC - RF
# calculate the fpr and tpr for all thresholds of the classification

fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_pred_rf)
roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC - RF')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # CONCLUSION
We have seen that Accuracy of both Random Forest and Decision Tree is equal, although teh precision of Random Forest is more.
In a fraud detection model, Precision is highly important because rather than predicting normal transactions correctly we want 
Fraud transactions to be predicted correctly and Legit to be left off.If either of the 2 reasons are not fulfiiled we may catch
the innocent and leave the culprit. This is also one of the reason why Random Forest and Decision Tree are used unstead of other algorithms.


Also the reason I have chosen this model is because of highly unbalanced dataset (Legit: Fraud :: 99.87:0.13). Random forest 
makes multiple decision trees which makes it easier (although time taking) for model to understand the data in a simpler way 
since Decision Tree makes decisions in a boolean way.
# # Models like XGBoost, Bagging, ANN, and Logistic Regression may give good accuracy but they won't give good precision and recall values.

# # What are the key factors that predict fraudulent customer?
# 
# 
1. The source of request is secured or not ?
2. Is the name of organisation asking for money is legit or not ?
3. Transaction history of vendors
# # Do these factors make sense? If yes, How? If not, How not? 
# 
To determine whether the factors in the dataset make sense, we need to conduct exploratory data analysis (EDA) 
and evaluate the data in the context of our research question or problem statement. In this case, we have a dataset 
named “froud.csv” and we want to assess whether the factors in this dataset make sense or not.

# # What kind of prevention should be adopted while company update its infrastructure?
# 
# 
1. Use smart vertified apps only.
2. Browse through secured websites.
3. Use secured internet connections (USE VPN).
4. Keep your mobile and laptop security updated.
5. Don't respond to unsolicited calls/SMS(s/E-mails.
6. If you feel like you have been tricked or security compromised, contact your bank immidiately.
# # Assuming these actions have been implemented, how would you determine if they work?
# 
# 
1. Bank sending E-statements.
2. Customers keeping a check of their account activity.
3. Always keep a log of your payments.
# In[ ]:




