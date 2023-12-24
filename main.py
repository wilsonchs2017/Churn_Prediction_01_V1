#Data Analysis
#---------------
#Import Libraries
import numpy as np #linear algebra
import pandas as pd #data processing, csv file
import matplotlib.pyplot as plt #for visualization
import seaborn as sns
plt.show() #%matplotlib incline #for jupyter

#Dataset
df = pd.read_csv("E:\ML Projects\Churn Prediction\Data Sets\Telco-Customer-Churn.csv")
#CSV link https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download
df.shape

#missing value check
df.isna().sum().sum()

#target variable
df.Churn.value_counts()

#column check
columns = df.columns
binary_cols = []

for col in columns:
    if df[col].value_counts().shape[0] == 2:
        binary_cols.append(col) #categorical features with two classes

#Categorical features with multiple classes
multiple_cols_cat = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                     'TechSupport', 'StreamingTV', 'Contract', 'PaymentMethod']

#Binary categorical features
#----------------------------
#six countplots based on counts and datas are gender, seniorcitizen, partner, dependents, phoneservice,paperlessbilling
fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)

sns.countplot(x="gender", data=df, ax=axes[0, 0])
sns.countplot(x="SeniorCitizen", data=df, ax=axes[0, 1])
sns.countplot(x="Partner", data=df, ax=axes[0, 2])
sns.countplot(x="Dependents", data=df, ax=axes[1, 0])
sns.countplot(x="PhoneService", data=df, ax=axes[1, 1])
sns.countplot(x="PaperlessBilling", data=df, ax=axes[1, 2])

churn_numeric = {'Yes':1, 'No':0}
df.Churn.replace(churn_numeric, inplace = True)

#churn rate table
df[['gender','Churn']].groupby(['gender']).mean()
df[['SeniorCitizen','Churn']].groupby(['SeniorCitizen']).mean()
df[['Partner','Churn']].groupby(['Partner']).mean()
df[['Dependents','Churn']].groupby(['Dependents']).mean()
df[['PhoneService','Churn']].groupby(['PhoneService']).mean()
df[['PaperlessBilling','Churn']].groupby(['PaperlessBilling']).mean()

#pivot table with churn values, index gender, columns SeniorCitizen
table = pd.pivot_table(df, values='Churn', index=['gender'], columns = ['SeniorCitizen'], aggfunc=np.mean)
table

#Other categorical features
#----------------------------
#Count plot on Internet Service
sns.countplot(x = "InternetService", data=df)

#table for internet service as group by while it shows churn rate monthly charges
df[['InternetService','Churn','MonthlyCharges']].groupby('InternetService').mean()

#six countplots on streamingtv, streamingmovies, onlinesecurity, onlinebackup, deviceprotection, techsupport
fig, axes = plt.subplots(2,3, figsize=(12,7), sharey = True)

sns.countplot(x="StreamingTV", data=df, ax=axes[0, 0])
sns.countplot(x="StreamingMovies", data=df, ax=axes[0, 1])
sns.countplot(x="OnlineSecurity", data=df, ax=axes[0, 2])
sns.countplot(x="OnlineBackup", data=df, ax=axes[1, 0])
sns.countplot(x="DeviceProtection", data=df, ax=axes[1, 1])
sns.countplot(x="TechSupport", data=df, ax=axes[1, 2])

#churn rate table
display(df[['StreamingTV','Churn']].groupby(['StreamingTV']).mean())
display(df[['StreamingMovies','Churn']].groupby(['StreamingMovies']).mean())
display(df[['OnlineSecurity','Churn']].groupby(['OnlineSecurity']).mean())
display(df[['OnlineBackup','Churn']].groupby(['OnlineBackup']).mean())
display(df[['DeviceProtection','Churn']].groupby(['DeviceProtection']).mean())
display(df[['TechSupport','Churn']].groupby(['TechSupport']).mean())


#Phone Service Data
#-------------------
#data of who has phone service
df.PhoneService.value_counts()

#data of who has multiple lines
df.MultipleLines.value_counts()

#churn rate of multple lines
df[['MultipleLines','Churn']].groupby('MultipleLines').mean()

#Contract and Payment Method Data
#---------------------------------
#plot table for contract
plt.figure(figsize=(10,6))
sns.countplot(x="Contract",data=df)

#churn rate for contract
df[['Contract','Churn']].groupby('Contract').mean()

#plot table for contract
plt.figure(figsize=(10,6))
sns.countplot(x="PaymentMethod",data=df)

#churn rate for contract
df[['PaymentMethod','Churn']].groupby('PaymentMethod').mean()

#Continous Features
#-------------------
#countplots on tenure, monthlycharges
fig, axes = plt.subplots(1,2, figsize = (12,7))
sns.distplot(df["tenure"], ax = axes[0])
sns.distplot(df["MonthlyCharges"], ax = axes[1])

#churn rate changes table related to tenure and monthly charges
df[['tenure','MonthlyCharges','Churn']].groupby("Churn").mean()

#contract table relate to tenure
df[['Contract','tenure']].groupby('Contract').mean()

#drop variables
df.drop(['customerID','gender','PhoneService','Contract','TotalCharges'], axis = 1, inplace = True)

#Data Preprocessing
#-------------------
#importing tools from scikit-learning package
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

#encoding categorical variables
cat_features = ['SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'PaperlessBilling', 'PaymentMethod']
X = pd.get_dummies(df, columns=cat_features, drop_first=True)


#scaling continuous variables
sc = MinMaxScaler()
a = sc.fit_transform(df[['tenure']])
b = sc.fit_transform(df[['MonthlyCharges']])
X['tenure'] = a
X['MonthlyCharges'] = b

#checkpoint on dataset dimensions
X.shape

#Resampling
#----------
#Countplot before Resampling on Class Distribution
sns.countplot(x = 'Churn', data=df).set_title('Class Distribution Before Resampling')

#separating positive class negative class of Churn
X_no = X[X.Churn == 0]
X_yes = X[X.Churn == 1]

#Unsampling the positive class
X_yes_unsampled = X_yes.sample(n = len(X_no), replace = True, random_state=42)
#print out unsampled positive class
print(len(X_yes_unsampled))

#class distribution check when positive and negative class combined
X_unsampled = X_no.append(X_yes_unsampled).reset_index(drop=True)
sns.countplot(x='Churn', data=X_unsampled).set_title('Class Distribution After Resampling')

#Model Creation and Evaluation
#-----------------------------
#Train and Subset Preparation
#import train test split from sklearn model selection
from sklearn.model_selection import train_test_split
#features independent tools
X=X_unsampled.drop(['Churn'],axis=1)
#target dependent variable
y=X_unsampled['Churn']

#Dividing dataset into train and test sublets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)

#Ridge Classifier
#import RidgeClassifier from the sklearn,linear_model and import accuracy_score from sklearn.metrics
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

#create ridge classifier object
clf_ridge = RidgeClassifier()
#train the model
clf_ridge.fit(X_train, y_train)

#make predictions on training set
pred = clf_ridge.predict(X_train)
#evaluate the predictions
accuracy_score(y_train,pred) #achieved 75%

#make predictions on test set
pred_test = clf_ridge.predict(X_test)
#evaluate the predictions
accuracy_score(y_test,pred_test) #achieved 76%

#Random Forest
#import RandomForestClassifier from sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier
#create random forest object
clf_forest = RandomForestClassifier(n_estimators=100, max_depth=10)
#Mentions -- n_estimators: number of trees in the forest, max_depth: maximum depth of the tree
#train the random forest object
clf_forest.fit(X_train, y_train)

#make predictions on training set
pred = clf_forest.predict(X_train)
#evaluate the prediction
accuracy_score(y_train,pred) #achieved 88%

#make predictions on test set
pred_test = clf_forest.predict(X_test)
#evaluate the prediction
accuracy_score(y_test, pred_test) #achieved 84%

#Improving the Model
#--------------------
#import GridSearchV from sklearn.model_selection for easy parameter tuning
from sklearn.model_selection import GridSearchCV

#create GridSearchCV object
parameters = {'n_estimators':[150,200,250,300], 'max_depth': [15,20,25]}
#set RandomForestClassifier() as forest variable for GridSearchCV
forest = RandomForestClassifier()
#setup GridSearchCV as clf variable
clf = GridSearchCV(estimator=forest, param_grid=parameters, n_jobs=-1, cv=5)

#train the parameters
clf.fit(X,y)

# check and print best parameters
print("Best Parameter is", clf.best_params_) #Best Parameter is {'max_depth': 25, 'n_estimators': 150}
# check and print best overall accuracy, best_score_:Mean cross-validated score of the best_estimator
print("Overall Accuracy is", clf.best_score_) #Overall Accuracy is 0.8996908586145143, achieved 90%