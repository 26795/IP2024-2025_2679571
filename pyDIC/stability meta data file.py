# -*- coding: utf-8 -*-
"""
Created on Tue Mar 4 17:29:48 2025

@author: 2679571P
"""
import numpy as np
import pandas as pd


force=[]

for n in range(1,6):
    force=[]
    fileNames_file=f'st{n}/steelFileNames.xlsx'
    fileNamesDF=pd.read_excel(fileNames_file)
    nrofFiles,a=fileNamesDF.shape
    for i in range(nrofFiles):  # force array of 0s
        force.append(0)
        
    
    print(force)
        
    forceEnd=force
    print(forceEnd)
    print(len(forceEnd))
        
    forceValues=forceEnd

    print(forceValues)
    
    #time column
    time=[]
    for i in range(nrofFiles):
        t_s=0+0.5*i
        #time=np.append(time, t_s)
        time.append(t_s)
    
    print (forceValues, 
           time)         
       
    #combine into one data frame 
    meta_data=pd.DataFrame()
    meta_data['file']=fileNamesDF['file']
    meta_data['time(s)']=time
    meta_data['force(N)']=forceValues
    print(meta_data)
    #make syre that the last row is repeated
    #extract last row
    last_row=meta_data.iloc[[-1]]
    # add row 
    meta_data = pd.concat([meta_data, last_row], ignore_index=True)  
    
    meta_data.to_csv(f'st{n}/meta-data.txt',header=True, index=False, sep='\t')
    print (n)
