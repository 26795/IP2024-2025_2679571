# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:56:42 2025

@author: 2679571P
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
from sklearn import linear_model 

# starting values
diameter=5 
n=4 # test number
k=9 # run number
interpolate_seconds=0.5 #s2
ratio_interpolation=0.5/interpolate_seconds
valuesStart=10 # linear region start
valuesEnd=64 # linear region end 
area= np.pi*1/4*(diameter/1000)**2

#import extension adn force data 
#extension_file=f'CSRT for 10/T{n}.xlsx'
extension_file=f'LK for 10/T{n}.xlsx'
#extension_file=f'CSRT for 10/T{n}_{k}.xlsx' # pix
#extension_file=f'LK for 10/T{n}_{k}.xlsx' # pix
#extension_file=f'LK for 10/T{n}.xlsx'
#extension_file='testRun2.xlsx' # pix
force_file='03 18 steel lab tensile SM1002Data.xlsx'   #kN

area= np.pi*1/4*(diameter/1000)**2

#extract data from excel
coordinates=pd.read_excel(extension_file)
force=pd.read_excel(force_file, sheet_name=f'T{n}')


# print(coordinates)
# #print force data
# print(force['Time'])
# print(force['DL1 Load Meter'])
# force.index
# force.size
# force.shape
print('force', force.shape)

# removing string values 
#Removing nonnumeric values from force data ##convert to numeric to remove non  numeric vlaues from data 
force['DL1 Load Meter'] = pd.to_numeric(force['DL1 Load Meter'], errors="coerce")
# Drop rows containing NaN 
force.dropna(subset=['DL1 Load Meter'], inplace=True)
# print(force['DL1 Load Meter'].values)
# print(force['DL1 Load Meter'])
# print(force['Time'])
print(force.shape)

print('coordintaes')
print(coordinates.shape)
print(coordinates.columns)



#calcualte distance between points 
coordinates['Extension (pix)']=coordinates.apply(
    lambda row: np.sqrt((row['obj-0 - X (pix)']-row['obj-1 - X (pix)'])**2
                        +(row['obj-0 - Y (pix)']-row['obj-1 - Y (pix)'])**2),
    axis=1)#row wise so needs to show axis =1


# print(coordinates.head)
# print(coordinates.columns)


#denoise video data from any significant outliers 
print('denoise and analyse data')

#denoise # weighted averagge to minimise any outliers 

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



#tidy force values # remove all zeroes from the end
print(force['DL1 Load Meter'])
print(force['Time'])
print(force['DL1 Load Meter'].values[0:5]) #first five values
force = force.drop([force.index[0]])
force = force.drop([force.index[-1]])
rows_number, columns_number =force.shape
i=0
while i < rows_number:
    arrayOfFive=force['DL1 Load Meter'].values[i:i+5]
    print(arrayOfFive, i)
    #print(arrayOfFive[4])
    print(arrayOfFive[0:-1])
    print(np.all(arrayOfFive[0:-1] !=0))
    print(np.all(arrayOfFive[0:-1]))
    if arrayOfFive.size>0 and arrayOfFive[-1]==0 and  np.all(arrayOfFive[0:-1]):
        print('pass')
        print(force.index[i])
        force = force.drop([force.index[i+4]])
        print(force.shape, 'force shape')
    else:
        print('fail')
        i+=1
    print(i)
    
# print(force['DL1 Load Meter'])
# print(force['Time'])
# print(force)


#downsize moving weighted average to fit with force data, change from 1/fps second to 0.5s
downsizedStrain=coordinates
forceInterpolated=pd.DataFrame()
forceInterpolated['Time']=force['Time']
forceInterpolated['DL1 Load Meter']=force['DL1 Load Meter']
print(forceInterpolated)
force_timestep=force['Time'].values[2]-force['Time'].values[1]
print(force_timestep)
print(coordinates.columns)
extension_timestep=coordinates[['Time (s)']].values[2]-coordinates[['Time (s)']].values[1]
print(extension_timestep)
print(downsizedStrain)
#convert into datetime form
downsizedStrain['Time (s)']=pd.to_datetime(downsizedStrain['Time (s)'],unit='s')
forceInterpolated['Time']=pd.to_datetime(forceInterpolated['Time'], unit='s')
downsizedStrain.index=downsizedStrain['Time (s)']
forceInterpolated.index=forceInterpolated['Time']
#downsizedStrain=downsizedStrain.resample('0.5s', on='Time (s)').mean()
downsizedStrain=downsizedStrain.resample(f'{interpolate_seconds}s', on='Time (s)').mean()
print('starinnnnn', downsizedStrain.head)
print(forceInterpolated)
# force interpoaltion if time step is changed to smth that is not 0.5s
forceInterpolated=forceInterpolated.resample(f'{interpolate_seconds}s').mean()
print('force', forceInterpolated.head)
#forceInterpolated=forceInterpolated.interpolate(method='linear')
forceInterpolated['DL1 Load Meter']=forceInterpolated['DL1 Load Meter'].interpolate(method='polynomial', order=2) # quadratic # specific to the olumn bc of the data frame it doesnt like it # no signifact change in the results 
print('interpl', forceInterpolated.head)


# calculating strain
print('strain')
print(downsizedStrain['Extension (pix)'])
# drop nans
downsizedStrain.dropna()
initial_lenght=downsizedStrain['Extension (pix)'].min()
print('innitial lenght', initial_lenght)
downsizedStrain['Strain']=(downsizedStrain['Extension (pix)']-initial_lenght)/initial_lenght
print('test', downsizedStrain['Strain'])
downsizedStrain.dropna(inplace=True)
print('testii', downsizedStrain.head)


# calcuting stress
print('stress')
#force converted into N
forceInterpolated['Stress']=forceInterpolated['DL1 Load Meter']*1000/area
print(forceInterpolated['Stress'])


#combine into one data frame # has to be the same size 
#,make strain array the same lenght as stress
force_shape, a = forceInterpolated.shape
print(forceInterpolated.shape)
downsizedStrain_shape,b=downsizedStrain.shape
print(downsizedStrain.shape)
Stress_strain_downsized=pd.DataFrame()
#Stress_strain_downsized['Time']=forceInterpolated['Time']
print(force_shape-downsizedStrain_shape)
#if stress is bigger then drop starting values
if force_shape>downsizedStrain_shape:
    print('force_shape>downsizedStrain_shape')
    strain=downsizedStrain['Strain'].values
    for i in range(force_shape-downsizedStrain_shape):
        strain=np.insert(strain,0,0)
        print(i)
    Stress_strain_downsized['Stress']=forceInterpolated[['Stress']]
    Stress_strain_downsized['Strain']=strain

# if stress is smaller drop first values in strain until the same size. end values match    
if force_shape<downsizedStrain_shape:
    print('force_shape<downsizedStrain_shape')
    stress=forceInterpolated['Stress'].values
    print(force)
    for i in range(downsizedStrain_shape-force_shape):
        if stress[0]<0:  # if intilta force value is less than 0, delete
            stress=np.delete(stress, 0)
        else:
            if stress[-1]<0:
                stress=np.delete(stress, -1)
    lenght_of_only_positive_force_val=len(stress)
    print(lenght_of_only_positive_force_val, 'lenght_of_only_positive_force_val')
    downsizedStrainII=pd.DataFrame()
    if  lenght_of_only_positive_force_val<downsizedStrain_shape:  #if lenght is differnet then shorten the strain data 
        strain=downsizedStrain['Strain'].values
        for i in range(downsizedStrain_shape-lenght_of_only_positive_force_val):
            strain=np.delete(strain,0)
        downsizedStrainII['Strain']=strain
    else:
        downsizedStrainII['Strain']=downsizedStrain['Strain']
    print(force)
    Stress_strain_downsized['Stress']=stress
    
    Stress_strain_downsized['Strain']=downsizedStrainII[['Strain']]

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

#plt.plot(strain, stress, label='extension')
valuesStart=int(valuesStart*ratio_interpolation)
valuesEnd=int(valuesEnd*ratio_interpolation)

linearstrain=Stress_strain_downsized[['Strain']].values[valuesStart:valuesEnd] # for linear regression 2d array is need so [[]]
linearstress=Stress_strain_downsized[['Stress']].values[valuesStart:valuesEnd]
#plt.plot(linearstrain, linearstress, label='extension')
#plt.plot([4.85E-05,1.43E-03],[10185916.36,300993828.4])
print('plotEnd')
print(linearstrain)
print(linearstress)
print(f'T{n} test')
print('valuesStart', valuesStart, ' valuesEnd', valuesEnd)

#linear fit 
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
#E_expected=130#GPa#copper
E_expected=200#GPa#steel 
print('E expected', E_expected, 'GPa')
error=(E_expected-E_calculated/10**9)/E_expected*100
print('error', error, '%')


# plots 
#whoel stress strain range
plt.plot()
#plt.subplot(1,2,1)
plt.xlabel('Strain')
plt.ylabel('Stress')
plt.title('Stress-strain graph')
plt.plot(strain, stress, label='Stress-strain curve')
#plt.plot(linearstrain, linearstress, label='extension',color='green')
plt.plot([4.85E-05,400993828.4/E_expected/10**9],[10185916.36,400993828.4], color = 'orange', label='Literature value E') #steel
#plt.plot([3.92E-05,2.84E-03],[5092958.179,369239468], color = 'orange', label='Expected slope')   #copper
plt.plot(linearstrain, lr.predict(linearstrain), color = 'blue', label = 'Calculated slope')
plt.legend()
#plt.savefig(f'CSRT for 10/steel{n}_curve.png')
#plt.savefig(f'LK for 10/steel{n}_curve.png')
# #plt.savefig(f'DIC for 10/copper{n}_curve.png')
plt.show()

# linear stress-strain
plt.plot()
#plt.subplot(1,2,2,)
plt.plot(linearstrain, linearstress, label='Stress-strain curve')
plt.plot([4.85E-05,400993828.4/E_expected/10**9],[10185916.36,400993828.4], color = 'orange', label='Literature value E') #steel
#plt.plot([3.92E-05,2.84E-03],[5092958.179,369239468], color = 'orange', label='Expected slope') # copper
plt.plot(linearstrain, lr.predict(linearstrain), color = 'blue', label = 'Calculated slope')
plt.legend()
plt.xlabel('Strain')
plt.title('Linear region')
#plt.savefig(f'CSRT for 10/steel{n}_elastic.png')
#plt.savefig(f'LK for 10/steel{n}_elastic.png')
# plt.savefig(f'DIC for 10/copper{n}_elastic.png')
plt.show()


# both# subplots
plt.plot()
plt.subplot(1,2,1)
plt.xlabel('Strain')
plt.ylabel('Stress')
plt.title('Stress-strain graph')
plt.plot(strain, stress, label='Stress-strain curve')
plt.plot(linearstrain, linearstress, label='extension',color='red')
#plt.plot([4.85E-05,1.43E-03],[10185916.36,300993828.4], color = 'orange', label='Literature value E') #steel
#plt.plot([3.92E-05,2.84E-03],[5092958.179,369239468], color = 'orange', label='Expected slope')   #copper
#plt.plot(linearstrain, lr.predict(linearstrain), color = 'blue', label = 'Calculated slope')
#plt.legend()

plt.subplot(1,2,2,)
plt.plot(linearstrain, linearstress, label='Stress-strain curve')
plt.plot([4.85E-05,400993828.4/E_expected/10**9],[10185916.36,400993828.4], color = 'orange', label='Literature value E') #steel
#plt.plot([3.92E-05,2.84E-03],[5092958.179,369239468], color = 'orange', label='Expected slope') # copper
plt.plot(linearstrain, lr.predict(linearstrain), color = 'blue', label = 'Calculated slope')
plt.legend()
plt.xlabel('Strain')
plt.title('Linear region')
plt.show()
