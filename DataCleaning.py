import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#removes junk data - rows with undefined values
def cleanData(dataFrame):
    df = dataFrame[(dataFrame.prob_stop != 'undefined') & (dataFrame["predictions_class"] != 'stop')]
    #this condition remove all entries with same campaign_id and different dates and keep only the maximum date
    #df = df.sort_values('datetime').groupby('campaign_id').tail(1)
    df.to_csv("cleaned_duplicates_undefined.csv");
    return df

#Trying to remove outliers from numerical variables using z-score
def removeOutliers(dataFrame, zScoreTreshold):
    print(dataFrame.shape)
    numericalVariablesColumnsList = ["campaign_payout", "clicks", "conversions", "payout", "net", "TR.ROI", "EPC"]
    z = np.abs(stats.zscore(dataFrame[numericalVariablesColumnsList]))
    dataFrame_o = dataFrame[(z < zScoreTreshold).all(axis=1)]
    dataFrame_o = dataFrame_o.loc[dataFrame_o['train_clicks'] > 4]
    dataFrame_o = dataFrame_o.loc[dataFrame_o['impressions'] > 0]
    print(dataFrame_o.shape)
    dataFrame_o.to_csv('cleanerData.csv')
    return dataFrame_o

dataFrame1 = pd.read_csv("data_predicted.csv")
cleanData(dataFrame1)
dataFrame = pd.read_csv('cleaned_duplicates_undefined.csv')
dataFrame_optimised = removeOutliers(dataFrame, 3)
