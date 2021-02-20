import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import preprocessing, metrics, tree
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler


#to see all data columns in terminal
pd.set_option('display.max_columns',36)

#help methods for data analysis:
#print dataset information - size, variable range etc
#useful to see IQR, data range and various statistics
def printData(dataFrame):
    print(dataFrame.info())
    print(dataFrame.shape)
    print(dataFrame.describe())

#plots the correlation matrix for a given dataframe
def plotCorr(dataFrame):
    plt.figure(figsize=(36,36))
    sns.heatmap(dataFrame.corr(), annot=True)
    plt.show()


def boxplotColumns(df):
    numericalVariablesColumnsList = ["campaign_payout", "clicks", "conversions", "payout", "net", "TR.ROI", "EPC"]
    for column in numericalVariablesColumnsList:
        plt.boxplot(df[column])
        plt.title(column)
        plt.show()




dataFrame = pd.read_csv("cleanerData.csv")

#plotCorr(dataFrame)
#printData(dataFrame_optimised)
#boxplotColumns(dataFrame_optimised)

X = dataFrame[['train_clicks', 'train_npc', 'train_roi', 'train_ctr', 'train_payout', 'lp_clicks',
               'EPC', 'impressions', 'net', 'TR.ROI', 'lpctr']]
y = dataFrame[['prediction_class_id']]

#not balanced
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model_dtree = tree.DecisionTreeClassifier()
X_res, y_res = X_train, y_train




#decision tree model
model_dtree.fit(X_res, y_res)
y_pred = model_dtree.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test,y_pred))
print(y_test)
print(y_pred)
# predictions = pd.DataFrame({'true_y_values':y_test,'predicted_y_values':y_pred})
# print(predictions)



#logistic regression model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# model_lreg = LogisticRegression()
# model_lreg.fit(X_res, y_res)
# y_pred = model_lreg.predict(X_test)
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
