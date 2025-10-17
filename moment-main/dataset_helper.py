# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:37:24 2024

@author: fakbarifar
"""
#%% import libraries

import os
# os.chdir("D:/OneDrive - Queen's University/UNITS")
os.chdir("D:/OneDrive - Queen's University/MOMENT/moment-main")

import numpy as np
import random 
import pandas as pd

import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime
from Create_All_data_3_handedness import read_from_excel3
from Create_All_data_3_handedness import PrepData_CMSA_1_concat, keep_1st_read, keep_1st_read_control, PrepData_FM_1, keep_1st_read_OH, keep_later_reads, keep_later_reads_control
from sklearn.model_selection import train_test_split

#%% initialize random seed

my_seed = 3
np.random.seed(my_seed)
random.seed(my_seed)





#%%
#create z-feature matrices from the csv files

# fileName = "Visually Guided Reaching_stroke_Mar_2025_adv.csv"
fileName = "VGR_Stroke_z_distTest_Mar_2025_unused_filled.csv"
saveName = 'VGR_Stroke_z_distTest_Mar_2025_unused'
data_label = 1

read_from_excel3(fileName, saveName, data_label, PATH="D:\OneDrive - Queen's University\MOMENT\moment-main")


# fileName = "Visually Guided Reaching_control_Mar_2025_adv.csv"
# saveName = 'VGR_Control_z_distTest_Mar_2025'
# data_label = 0

# read_from_excel3(fileName, saveName, data_label, PATH="D:\OneDrive - Queen's University\MOMENT\moment-main")








#%% Remove subjects with multiple stroke dates

date_file_PATH = "D:\OneDrive - Queen's University\Co-teaching\data_time_exploration"
dateFile = os.path.join(date_file_PATH, "lesion_locations.csv")
df = pd.read_csv(dateFile)

# PATH = "D:\OneDrive - Queen's University\Co-teaching\data_time_exploration"
Stroke_All_3 = np.load('VGR_Stroke_z_distTest_Mar_2025_unused.npz')
stData0 = Stroke_All_3['RepFeat']
stDate0 = Stroke_All_3['RepDate']
stTime0 = Stroke_All_3['RepTime']
# PATH = "D:\OneDrive - Queen's University\Co-teaching\data_time_exploration"
# Control_All_3 = np.load('VGR_Control_z_distTest_Mar_2025.npz')
# ctData0 = Control_All_3['RepFeat']
# ctDate0 = Control_All_3['RepDate']
# ctTime0 = Control_All_3['RepTime']
    
all_subs = df['SUBJECTKEY']
unique_subs, unique_counts = np.unique(all_subs, return_counts=True)
   


remove_subs = []
for i in range(len(unique_subs)):
    if unique_counts[i]>1:
        all_dates = []
        nodate = 0
        ind_date = np.where(unique_subs[i]==all_subs)
        for j in ind_date[0]:
            # aa = ind_date[0][0]
            date_st = df['DATEOFSTROKE'][j]
            if date_st==date_st:
                if df['DATEOFSTROKE_UNKNOWN'][j]!='Y':
                    all_dates.append(date_st)
            else:
                nodate = 1
                
        if len(set(all_dates)) != 1 or nodate==1: # input_list has all identical elements.
            remove_subs.append(unique_subs[i])
            
            
# final_remove = []           
# for sb in remove_subs:
#     ind_sb = np.where(stData0[:, 0]==sb)
#     if ind_sb[0].size>0:
#         final_remove.append(sb)
#         stData0 = np.delete(stData0, ind_sb[0], 0)
#         stDate0 = np.delete(stDate0, ind_sb[0], 0)
#         stTime0 = np.delete(stTime0, ind_sb[0], 0)
        
#%% Read data

# PATH = "D:\OneDrive - Queen's University\Co-teaching\data_time_exploration"
# Control_All_3 = np.load('VGR_Control_z_distTest_Mar_2025.npz')
# ctData0 = Control_All_3['RepFeat']
# ctDate0 = Control_All_3['RepDate']
# ctTime0 = Control_All_3['RepTime']
  
#%% Use first recordings 

# stData, stDate, stTime = keep_1st_read(stData0, stDate0, stTime0)
# ctData, ctDate, ctTime = keep_1st_read_control(ctData0, ctDate0, ctTime0)



# keep all recordings
# ctData = ctData0
# ctDate = ctDate0
# ctTime = ctTime0


stData = stData0
stDate = stDate0
stTime = stTime0

#%% Use later recordings

# stData, stDate, stTime = keep_later_reads(stData0, stDate0, stTime0)
# ctData, ctDate, ctTime = keep_later_reads_control(ctData0, ctDate0, ctTime0)

#%% Concatenating features of both arms; Strokes:affected+less-affected, Controls:non-dominant+dominant

PATH = "D:/OneDrive - Queen's University/Current work/data"
dataFile = os.path.join(PATH, 'clinical_scores.csv')
dfc = pd.read_csv(dataFile)
subjects = dfc['SUBJECTKEY']
unq_subjects = np.unique(subjects)
affected = dfc['AFFECTEDARM']

new_stData = []
new_stDate = []
new_stTime = []

do = True
i_st = 0
while do:
    if np.where(subjects==stData[i_st,0])[0].size>0:
        aff = affected[np.where(subjects==stData[i_st,0])[0][0]]
        # if aff=='R':
        #     aff=1
        # elif aff=='L':
        #     aff=2
        # elif aff=='B':
        #     aff=1
        # if stData[i_st, 1]==aff:
        #     new_stData.append(stData[i_st, :])
        #     new_stDate.append(stDate[i_st])
        #     new_stTime.append(stTime[i_st])
        
        if aff=='R':
            new_stData.append( np.concatenate([stData[i_st, :-1],stData[(i_st+1), :]]))
            new_stDate.append( (stDate[i_st], stDate[(i_st+1)]) )
            new_stTime.append( (stTime[i_st], stTime[(i_st+1)]) )
        elif aff=='L':
            new_stData.append( np.concatenate([stData[(i_st+1), :-1],stData[i_st, :]]))
            new_stDate.append( (stDate[(i_st+1)], stDate[i_st]) )
            new_stTime.append( (stTime[(i_st+1)], stTime[i_st]) )
        elif aff=='B':
            if stData[i_st, 2]==1:
                new_stData.append( np.concatenate([stData[(i_st+1), :-1],stData[i_st, :]]))
                new_stDate.append( (stDate[(i_st+1)], stDate[i_st]) )
                new_stTime.append( (stTime[(i_st+1)], stTime[i_st]) )
            elif stData[i_st, 2]==2:
                new_stData.append( np.concatenate([stData[i_st, :-1],stData[(i_st+1), :]]))
                new_stDate.append( (stDate[i_st], stDate[(i_st+1)]) )
                new_stTime.append( (stTime[i_st], stTime[(i_st+1)]) )
            elif stData[i_st, 2]==3:
                new_stData.append( np.concatenate([stData[(i_st+1), :-1],stData[i_st, :]]))
                new_stDate.append( (stDate[(i_st+1)], stDate[i_st]) )
                new_stTime.append( (stTime[(i_st+1)], stTime[i_st]) )
                
            
            
    i_st += 2
    if i_st==len(stData):
        do = False
            
            
            
new_stData = np.asarray(new_stData)
new_stDate = np.asarray(new_stDate)
new_stTime = np.asarray(new_stTime)


# new_ctData = []
# new_ctDate = []
# new_ctTime = []
# # for i_ct in range(len(ctData)):
# #     if ctData[i_ct, 1]==ctData[i_ct, 2]:
# #         new_ctData.append(ctData[i_ct, :]) 
# #         new_ctDate.append(ctDate[i_ct])
# #         new_ctTime.append(ctTime[i_ct])
# #     elif ctData[i_ct, 2]==3 and ctData[i_ct, 1]==1:
# #         new_ctData.append(ctData[i_ct, :])
# #         new_ctDate.append(ctDate[i_ct])
# #         new_ctTime.append(ctTime[i_ct])
# doc = True
# i_ct = 0
# while doc:
#     if ctData[i_ct, 1]==ctData[i_ct, 2]:
#         new_ctData.append( np.concatenate([ctData[(i_ct+1), :-1],ctData[i_ct, :]]))
#         new_ctDate.append( (ctDate[(i_ct+1)], ctDate[i_ct]) )
#         new_ctTime.append( (ctTime[(i_ct+1)], ctTime[i_ct]) )
#     else:
#         new_ctData.append( np.concatenate([ctData[i_ct, :-1],ctData[(i_ct+1), :]]))
#         new_ctDate.append( (ctDate[i_ct], ctDate[(i_ct+1)]) )
#         new_ctTime.append( (ctTime[i_ct], ctTime[(i_ct+1)]) )
        
        
#     i_ct += 2
#     if i_ct==len(ctData):
#         doc = False
    
# new_ctData = np.asarray(new_ctData)
# new_ctDate = np.asarray(new_ctDate)
# new_ctTime = np.asarray(new_ctTime)
    
# np.savez('match_CT', new_ctData=new_ctData, new_ctDate=new_ctDate, new_ctTime=new_ctTime)
# np.savez('match_ST', new_stData=new_stData, new_stDate=new_stDate, new_stTime=new_stTime)

#%% Add CMSA
# new_ctLab = np.zeros(new_ctData.shape[0])
new_stLab = np.ones(new_stData.shape[0])

 
# returm "right arm" + "left arm", as two digits
# ctData_clnc, ctDate_clnc, ctTime_clnc, ctLab_clnc, ct_CMSA, ct_hdr, ct_noIND = PrepData_CMSA_1_concat(new_ctData, new_ctDate, new_ctTime, new_ctLab, method='weak_first') 
stData_clnc, stDate_clnc, stTime_clnc, stLab_clnc, st_CMSA, st_hdr, st_noIND = PrepData_CMSA_1_concat(new_stData, new_stDate, new_stTime, new_stLab, method='weak_first')


# ctLabs = np.empty((ctLab_clnc.shape[0], 2))
# ctLabs[:, 0] = ctLab_clnc
# ctLabs[:, 1] = ct_CMSA

stLabs = np.empty((stLab_clnc.shape[0], 2))
stLabs[:, 0] = stLab_clnc
stLabs[:, 1] = st_CMSA

# np.savez('match_CT__Mar_2025', new_ctData=ctData_clnc, new_ctDate=ctDate_clnc, new_ctTime=ctTime_clnc, new_ctLabs=ctLabs, new_ct_hdr=ct_hdr)
np.savez('match_ST__Mar_2025_unused', new_stData=stData_clnc, new_stDate=stDate_clnc, new_stTime=stTime_clnc, new_stLabs=stLabs, new_st_hdr=st_hdr)


