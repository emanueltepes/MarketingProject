import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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

#removes junk data - rows with undefined values
def cleanData(dataFrame):
    cleaned = dataFrame[dataFrame.prob_stop != 'undefined']
    dataFrame.to_csv('cleanedData.csv')
    return cleaned

def boxplotColumns(ColumnsNamesList, df):
    for column in ColumnsNamesList:
        plt.boxplot(df[column])
        plt.title(column)
        plt.show()

#Trying to remove outliers from numerical variables using z-score
def removeOutliers(dataFrame, zScoreTreshold):
    print(dataFrame.shape)
    numericalVariablesColumnsList = ["campaign_payout", "clicks", "conversions", "payout", "net", "TR.ROI", "EPC", "train_clicks"]
    z = np.abs(stats.zscore(dataFrame[numericalVariablesColumnsList]))
    dataFrame_o = dataFrame[(z < zScoreTreshold).all(axis=1)]
    print(dataFrame_o.shape)
    return dataFrame_o

#read dataset with no "undefined values"
dataFrame = pd.read_csv('cleanedData.csv')
printData(dataFrame)

#clean dataset from numerical outliers using z-score and saves to new csv
dataFrame_optimised = removeOutliers(dataFrame,3)
dataFrame_optimised.to_csv('cleanerData.csv')


