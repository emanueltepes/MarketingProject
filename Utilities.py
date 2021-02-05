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