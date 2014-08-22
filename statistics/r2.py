import scipy

data = [-3,-1,2,3]
predictions = [-2.19,-1.2,0.73,3.66]
r_squared = scipy.stats.pearsonr(data, predictions)[0]**2
print r_squared
