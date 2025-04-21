# -*- coding: utf-8 -*-
"""
Created on Tue Mar 4 17:29:48 2025

@author: 2679571P
"""
import numpy as np
import pandas as pd



force_file='03 18 steel lab tensile SM1002Data.xlsx'

#for n in range(1, 11):
for n in range(10,11):
    fileNames_file=f'tests/T{n}/steelFileNames.xlsx'
    fileNamesDF=pd.read_excel(fileNames_file)
    force=pd.read_excel(force_file, sheet_name=f'T{n}')
    
    print(force.columns)
    force=force.iloc[2:]
    
    forceEnd=force['DL1 Load Meter'].values
    print(forceEnd)
    print(len(forceEnd))
    i=0
    #remove all negative force values at the end 
    while forceEnd[-1]<=0:
        forceEnd = np.delete(forceEnd, -1)
        i=i+1
    print(len(forceEnd), i)
    print(forceEnd)
    
    # after neative end values are removed then shorten the force data frame
    force=force[:-i]
     
    # insert forceEnd values into the data frame
    force['DL1 Load Meter']=forceEnd

    
        
    print('forceII/',force['DL1 Load Meter'])  
    
    print('forceII',force['DL1 Load Meter'])
    
    
    
    print(force['DL1 Load Meter'])
    nrofFiles,a=fileNamesDF.shape
    nrofForce,b=force.shape
    
    
    forceValues=force['DL1 Load Meter'].values
    
    
    if nrofFiles>nrofForce:
        for i in range(nrofFiles-nrofForce): # if more pictures then add 0 force at the ebgigng
            forceValues=np.insert(forceValues,0,0)
    if nrofFiles<nrofForce: # remove intiial values a thtebignging 
        for i in range(nrofForce-nrofFiles):
            forceValues=np.delete(forceValues, 0)
    
    print(forceValues)
    
    #time column
    time=[]
    for i in range(len(forceValues)):
        t_s=0+0.5*i
        time=np.append(time, t_s)
    
    print (forceValues, 
           time)         
       
    #make equla lenght 
    meta_data=pd.DataFrame()
    meta_data['file']=fileNamesDF['file']
    meta_data['time(s)']=time
    meta_data['force(N)']=forceValues*1000
    print(meta_data)
    #make syre that the last row is repeated
    #extract last row
    last_row=meta_data.iloc[[-1]]
    # add row 
    meta_data = pd.concat([meta_data, last_row], ignore_index=True)  
    
    meta_data.to_csv(f'tests/T{n}/meta-data.txt',header=True, index=False, sep='\t')
    print (n)
