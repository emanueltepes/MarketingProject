import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    return cleaned


#read data and clean
dataFrameInit = pd.read_csv('data_cleaned.csv')

dataFrame = cleanData(dataFrameInit)
printData(dataFrame)

#plot advertisers
#dataFrame['traffic_source_name'].value_counts().plot.bar()
#plt.show()

#plot top X campaign by id
#dataFrame['campaign_id'].value_counts().head(30).plot.bar()
#plt.show()
#plot with relative proportions
#(dataFrame['traffic_source_name'].value_counts().head(10) / len(dataFrame)).plot.bar()
#plt.show()
#dataFrame['clicks'].value_counts().sort_index().plot.bar()
#plt.show()

print(dataFrame.shape)
numericalVariablesColumnsList = ["campaign_payout", "clicks", "conversions", "payout", "net", "TR.ROI", "EPC", "train_clicks"]
for column in numericalVariablesColumnsList:
    plt.boxplot(dataFrame[column])
    plt.title(column)
    plt.show()


