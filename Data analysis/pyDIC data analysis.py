# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:56:42 2025

@author: 2679571P
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
from sklearn.neighbors import LocalOutlierFactor
from sklearn import linear_model 


n=5 # test number
k=8 # run number
valuesStart=35
valuesEnd=48

#import data
results_file=f'pyDIC tests/tests/T{n}.xlsx' # pix
#results_file=f'T{n}_{k}D.xlsx' # pix

#extract data from excel
results=pd.read_excel(results_file)


print(results)
print(results.columns)



Stress_strain_downsized=pd.DataFrame()
Stress_strain_downsized['Stress']=results['stress']
    
Stress_strain_downsized['Strain']=np.sqrt(results['ave_strain_xx']**2+results['ave_strain_yy']**2)


#remove outlier values #Local Outlier Factor #LOF
'''adapted from 
Title: Anomaly Detection Example with Local Outlier Factor in Python
Author: DataTechNotes
Date: 01/04/2020
Availability: https://www.datatechnotes.com/2020/04/anomaly-detection-with-local-outlier-factor-in-python.html
and
Title: Outlier detection with Local Outlier Factor (LOF)
Author: scikit learn
Availability: https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html
''' 
lof = LocalOutlierFactor(n_neighbors=8, contamination='auto')
X=Stress_strain_downsized[['Strain']].values
y_pred=lof.fit_predict(X)
Stress_strain_downsized['Outlier'] = y_pred
print(Stress_strain_downsized['Outlier'].head)
Stress_strain_downsized = Stress_strain_downsized[Stress_strain_downsized['Outlier'] != -1]

 
# print('print stress strain ',Stress_strain_downsized)
# print(Stress_strain_downsized.columns)
# print('print stress strain ',Stress_strain_downsized['Strain'])



##stress-strain plot
#plotting force and extension with time
print('plotStart')
strain=Stress_strain_downsized[['Strain']].values
stress=Stress_strain_downsized[['Stress']].values
print(strain)
print(stress)


linearstrain=Stress_strain_downsized[['Strain']].values[valuesStart:valuesEnd] # for linear regression 2d array is need so [[]]
linearstress=Stress_strain_downsized[['Stress']].values[valuesStart:valuesEnd]
#plt.plot(linearstrain, linearstress, label='extension')
#plt.plot([4.85E-05,1.43E-03],[10185916.36,300993828.4])
print('plotEnd')
print(linearstrain)
print(linearstress)
print('valuesStart', valuesStart, ' valuesEnd', valuesEnd)


'''adapted from 
Author: Samrat Kishore and WoJ
Date: 14/10/2019
Availability: https://stackoverflow.com/questions/29934083/linear-regression-on-pandas-dataframe-using-sklearn-indexerror-tuple-index-ou 
''' 
lr=linear_model.LinearRegression().fit(linearstrain, linearstress)
print('fit')
#plt.plot(linearstrain, lr.predict(linearstrain), color = 'blue')
print(lr.coef_)
[[E_calculated]]=lr.coef_
print(E_calculated, 'Pa')
print(E_calculated/10**9, 'GPa')
E_expected=130#GPa#copper
print('E expected', E_expected, 'GPa')
E_expected=130*10**9
error=(E_expected-E_calculated)/E_expected*100
print('error', error, '%')

plt.plot()
#plt.subplot(1,2,1)
plt.xlabel('Strain')
plt.ylabel('Stress')
plt.title('Stress-strain graph')
plt.plot(strain, stress, label='Stress-strain curve')
#plt.plot(linearstrain, linearstress, label='extension',color='green')
#plt.plot([4.85E-05,1.43E-03],[10185916.36,300993828.4], color = 'orange', label='Literature value E') #steel
plt.plot([3.92E-05,2.84E-03],[5092958.179,369239468], color = 'orange', label='Expected slope')   #copper
plt.plot(linearstrain, lr.predict(linearstrain), color = 'blue', label = 'Calculated slope')
plt.legend()
#plt.savefig(f'tests/T{n}/copper{n}_curve.png')
plt.show()

plt.plot()
#plt.subplot(1,2,2,)
plt.plot(linearstrain, linearstress, label='Stress-strain curve')
#plt.plot([4.85E-05,1.43E-03],[10185916.36,300993828.4], color = 'orange', label='Literature value E') #steel
plt.plot([3.92E-05,2.84E-03],[5092958.179,369239468], color = 'orange', label='Expected slope') # copper
plt.plot(linearstrain, lr.predict(linearstrain), color = 'blue', label = 'Calculated slope')
plt.legend()
plt.xlabel('Strain')
plt.title('Linear region')
#plt.savefig(f'tests/T{n}/copper{n}_elastic.png')
plt.show()


plt.plot()
plt.subplot(1,2,1)
plt.xlabel('Strain')
plt.ylabel('Stress')
plt.title('Stress-strain graph')
plt.plot(strain, stress, label='Stress-strain curve')
plt.plot(linearstrain, linearstress, label='extension',color='red')
#plt.plot([4.85E-05,1.43E-03],[10185916.36,300993828.4], color = 'orange', label='Literature value E') #steel
plt.plot([3.92E-05,2.84E-03],[5092958.179,369239468], color = 'orange', label='Expected slope')   #copper
plt.plot(linearstrain, lr.predict(linearstrain), color = 'blue', label = 'Calculated slope')
#plt.legend()

plt.subplot(1,2,2,)
plt.plot(linearstrain, linearstress, label='Stress-strain curve')
#plt.plot([4.85E-05,1.43E-03],[10185916.36,300993828.4], color = 'orange', label='Literature value E') #steel
plt.plot([3.92E-05,2.84E-03],[5092958.179,369239468], color = 'orange', label='Expected slope') # copper
plt.plot(linearstrain, lr.predict(linearstrain), color = 'blue', label = 'Calculated slope')
plt.legend()
plt.xlabel('Strain')
plt.title('Linear region')
plt.show()

