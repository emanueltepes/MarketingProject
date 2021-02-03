import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataFrame = pd.read_csv('data_predicted.csv')
#clean undefined values
data = dataFrame[dataFrame.prob_stop != 'undefined']
#we lose almost half of our rows
print(data.shape)

def printData(dataFrame):
    print(dataFrame.info())
    print(dataFrame.shape)
    print(dataFrame.describe())

def plotCorr(dataFrame):
    plt.figure(figsize=(36,36))
    sns.heatmap(dataFrame.corr(), annot=True)
    plt.show()

printData(data)
plotCorr(data)





