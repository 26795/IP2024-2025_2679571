# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:56:42 2025

@author: 2679571P
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
from scipy import stats

import seaborn as sns
import statsmodels.api as sm
from sklearn.neighbors import LocalOutlierFactor
from sklearn import linear_model 

#Copper tests
diameter=5 #mm

n=4
type='C'

interpolate_seconds=0.5 #s2
ratio_interpolation=0.5/interpolate_seconds
valuesStart=45
valuesEnd=64


extension_file=f'ST{n}{type}.xlsx' # pix
#extension_file=f'CT{n}{type}.xlsx' # pix



force_file='stability force no data.xlsx'   #kN


area= np.pi*1/4*(diameter/1000)**2

#extract data from excel
coordinates=pd.read_excel(extension_file)
force=pd.read_excel(force_file, sheet_name='stab')


print(coordinates)
print('force')
print(force)
print(force.iloc[5])
print(force.columns)

#print force data
print(force['Time'])
print(force['DL1 Load Meter'])
force.index
force.size
force.shape
force['Time'].values
print(force['DL1 Load Meter'].values)

# removing string values 
#Removing nonnumeric values from force data ##convert to numeric to remove non  numeric vlaues from data 
force['DL1 Load Meter'] = pd.to_numeric(force['DL1 Load Meter'], errors="coerce")
# Drop rows containing NaN 
force.dropna(subset=['DL1 Load Meter'], inplace=True)
print(force['DL1 Load Meter'].values)
print(force['DL1 Load Meter'])
print(force['Time'])

print('coordintaes')
#print(coordinates.axes)
coordinates.shape
coordinates.head
print(coordinates.columns)


#calcualte distance between points 
coordinates['Extension (pix)']=coordinates.apply(
    lambda row: np.sqrt((row['obj-0 - X (pix)']-row['obj-1 - X (pix)'])**2
                        +(row['obj-0 - Y (pix)']-row['obj-1 - Y (pix)'])**2),
    axis=1)#row wise so needs to show axis =1


coordinates['Extension (pix)'].dropna()
#cahnge pixels into strain
initial_lenght=coordinates['Extension (pix)'].values[0]
print('innitial lenght', initial_lenght)
coordinates['Strain']=(coordinates['Extension (pix)']-initial_lenght)/initial_lenght



#plotting force and extension with time 
plt.plot()
print('plot')
plt.xlabel('Time')
plt.ylabel('strain')
plt.title('Extension with time')
#plt.plot(coordinates['Time (s)'],coordinates['Extension (pix)'], label='extension')
plt.plot(coordinates['Time (s)'],coordinates['Strain'], label='extension')

plt.savefig(f'ST{n}{type}_time1.png')
#plt.savefig(f'CT{n}{type}_time1.png')

plt.show()

#denoise video data from any significant outliers 
print('denoise and analyse data')

#denoise
#moving average 
'''adapted from 
Author: Sander van den Oord
Date:  25/02/2020
Code version: 2
Availability: https://stackoverflow.com/questions/9621362/how-do-i-compute-a-weighted-moving-average-using-pandas
'''
weigthing=[0.12, 0.12,0.12, 0.12,0.12,0.10, 0.10, 0.1,0.1]
weigthing_sum=1
coordinates['Extension (pix)']=(coordinates['Extension (pix)'].
                                            rolling(window=9,center=True).
                                            apply(lambda x: np.sum(weigthing*x)/weigthing_sum, raw=False))

# print(coordinates.head)
# print(coordinates['Extension (pix)'].head)
# print(coordinatesOLD['Extension (pix)'].head)

coordinates.dropna()
print(coordinates['Extension (pix)'].head)
coordinates=coordinates.dropna()
print(coordinates['Extension (pix)'].head)


initial_lenght=coordinates['Extension (pix)'].values[0]
print('innitial lenght', initial_lenght)
coordinates['Strain']=(coordinates['Extension (pix)']-initial_lenght)/initial_lenght

# plot values with time

plt.plot()
print('plot')
plt.xlabel('Time')
plt.ylabel('Strain')
plt.title('Extension with time')
plt.plot(coordinates['Time (s)'],coordinates['Strain'], label='extension')

plt.savefig(f'ST{n}{type}_time2.png')
#plt.savefig(f'CT{n}{type}_time2.png')
plt.show()

