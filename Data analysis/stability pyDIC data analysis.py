# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:56:42 2025

@author: 2679571P
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
from sklearn.neighbors import LocalOutlierFactor


n=1 # test number
#k=8 # run number


print(n)
stress=f'st{n}/0304stressDataframe.xlsx'
strainX=f'st{n}/0304ave_strain_xxDataframe.xlsx'
strainY=f'st{n}/0304ave_strain_yyDataframe.xlsx'

stress=pd.read_excel(stress)
strainX=pd.read_excel(strainX)
strainY=pd.read_excel(strainY)
print(stress.columns)
results=pd.DataFrame()
results['stress']=stress[0]
results['ave_strain_xx']=strainX[0]
results['ave_strain_yy']=strainY[0]
print(results.shape)

lenght, b=results.shape
time=[]
for b in range(lenght):
    dt=0+b*0.5
    time.append(dt)

results['time']=time
print(results.head)        

Stress_strain_downsized=pd.DataFrame()
Stress_strain_downsized['Stress']=results['stress']
    
Stress_strain_downsized['Strain']=np.sqrt(results['ave_strain_xx']**2+results['ave_strain_yy']**2)
Stress_strain_downsized['Time']=results['time']

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



##stress-strain plot
#plotting force and extension with time
print('plotStart')
#strain=Stress_strain_downsized[['Strain']].values
strain=Stress_strain_downsized['Strain'].values

stress=Stress_strain_downsized[['Stress']].values
time=Stress_strain_downsized['Time'].values
print(strain)
print(stress)
print(time)

# plot values with time

plt.plot()
print('plot')
plt.xlabel('Time')
plt.ylabel('Strain')
plt.title('Extension with time')
plt.plot(time, strain, label='extension')

plt.savefig(f'ST{n}pydic_time.png')
plt.show()
    