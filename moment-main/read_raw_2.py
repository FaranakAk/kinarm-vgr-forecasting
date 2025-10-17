# -*- coding: utf-8 -*-
"""
In read_raw_2.py, read_from_mat function is modified to start the trials at (target-on - 200ms) and end the trials at movement offset + offmore, which is calculated from "TARGET_ON + RT + TOTALMT".
"""

#%%

import os
os.chdir("D:/OneDrive - Queen's University/MOMENT/moment-main")

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import scipy.io as spio
import scipy.signal as sps
import numpy as np
import pandas as pd
import fnmatch
import random
import h5py


#%% initialize random seed

my_seed = 3
np.random.seed(my_seed)
random.seed(my_seed)

#%% 

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


# # Example loading
# PATH = "C:/Users/fakbarifar/OneDrive - Queen's University/Faranak/control_mat"
# mat_contents = loadmat(os.path.join(PATH, '11_2012-07-04_14-09-32.mat'))
# r = mat_contents['r']

# PATH = "D:/OneDrive - Queen's University/Faranak/control_control_mat"
# mat_contents = loadmat(os.path.join(PATH, '11_2012-07-04_14-11-38.mat'))
# r = mat_contents['r']


# #hints for using the output
# r[0].analysis.REPORT_FEATURES.min_max_speed_difference_std
# r[0].analysis.REPORT_FEATURES.min

# len(r['c3d'][1]['Right_HandX'])
# Out[22]: 5766
# r['analysis']['REPORT_FEATURES']['speed_maxima_count_std']


#%%

def extract_exam_info(data_folder):
    """ Extracts subject ID and date and time of the exam from just the name of the mat files that can 
        be used to uniquely identify the exam
        input: folder of control/stroke mat files
        output: a (number of subjects) by 7 numpy array. Each row contains sub_ID, exam_year, exam_month,
        exam_day, exam_hour, exam_minute, exam_second """
    
    datalist = []
    id_mat = []
    for file in os.listdir(data_folder):
        if fnmatch.fnmatch(file, '*.mat'):
            datalist.append(file)
            
            file = file.split(".")[0]
            sub_ID = file.split("_")[0]
            exam_year = file.split("_")[1].split("-")[0]
            exam_month = file.split("_")[1].split("-")[1]
            exam_day = file.split("_")[1].split("-")[2]
            exam_hour = file.split("_")[2].split("-")[0]
            exam_minute = file.split("_")[2].split("-")[1]
            exam_second = file.split("_")[2].split("-")[2]
            
            id_mat.append([sub_ID, exam_year, exam_month, exam_day, exam_hour, exam_minute, exam_second])
    
    id_mat = np.asarray(id_mat)
    
    return id_mat


CT_mat_folder = "D:/OneDrive - Queen's University/Faranak/control_mat_Mar_2025_adv" 
CT_id_matrix = extract_exam_info(CT_mat_folder)

ST_mat_folder = "D:/OneDrive - Queen's University/Faranak/stroke_mat_Mar_2025_adv" 
ST_id_matrix = extract_exam_info(ST_mat_folder)

#%%

def match_matData2csvData(new_xData, new_xLabs, new_xDate, new_xTime, x_id_matrix, w_or_s):
    new_datalist = []
    task_scores = []
    features = []
    for i_mat in range(len(x_id_matrix)):
        for i_csv in range(len(new_xData)):
            exam_day = new_xDate[i_csv].split("/")[0]
            if len(exam_day)==1: # months are saved with two digits in x_id_matrix
                exam_day='0'+exam_day
            exam_month = new_xDate[i_csv].split("/")[1]
            if len(exam_month)==1: # months are saved with two digits in x_id_matrix
                exam_month='0'+exam_month
            exam_year = new_xDate[i_csv].split("/")[2]
            
            exam_hour = new_xTime[i_csv].split(":")[0]
            if len(exam_hour)==1: # months are saved with two digits in x_id_matrix
                exam_hour='0'+exam_hour
            exam_minute = new_xTime[i_csv].split(":")[1]
            if len(exam_minute)==1: # months are saved with two digits in x_id_matrix
                exam_minute='0'+exam_minute
            exam_second = new_xTime[i_csv].split(":")[2]
            if len(exam_second)==1: # months are saved with two digits in x_id_matrix
                exam_second='0'+exam_second
            
            if x_id_matrix[i_mat][0]==str(int(new_xData[i_csv][0])) and x_id_matrix[i_mat][1]==exam_year \
            and x_id_matrix[i_mat][2]==exam_month and x_id_matrix[i_mat][3]==exam_day \
            and x_id_matrix[i_mat][4]==exam_hour and x_id_matrix[i_mat][5]==exam_minute \
            and x_id_matrix[i_mat][6]==exam_second:
                file_name = x_id_matrix[i_mat][0]+'_'+x_id_matrix[i_mat][1]+'-' \
                                    +x_id_matrix[i_mat][2]+'-'+x_id_matrix[i_mat][3]+'_' \
                                    +x_id_matrix[i_mat][4]+'-'+x_id_matrix[i_mat][5]+'-' \
                                    +x_id_matrix[i_mat][6]+'.mat'
                new_datalist.append(file_name)
                task_scores.append(list(new_xData[i_csv][0:6]) + list(new_xLabs[i_csv]))
                if w_or_s == 0:
                    features.append(new_xData[i_csv][6:20])
                elif w_or_s == 1:
                    features.append(new_xData[i_csv][26:-1])
                    
    new_datalist = np.asarray(new_datalist)
    task_scores = np.asarray(task_scores)
    features = np.asarray(features)
    return task_scores, features, new_datalist


tt = np.load('match_CT__Mar_2025.npz', allow_pickle=True)
new_ctData = tt['new_ctData']
new_ctDate = tt['new_ctDate']
new_ctTime = tt['new_ctTime']
new_ctLabs = tt['new_ctLabs']


tt = np.load('match_ST__Mar_2025.npz', allow_pickle=True)
new_stData = tt['new_stData']
new_stDate = tt['new_stDate']
new_stTime = tt['new_stTime']
new_stLabs = tt['new_stLabs']



w_or_s = 0 # 0 for weak and 1 for strong arm
CT_ts, CT_features, new_CT_list = match_matData2csvData(new_ctData, new_ctLabs, new_ctDate[:,w_or_s], new_ctTime[:,w_or_s], CT_id_matrix, w_or_s)   
ST_ts, ST_features, new_ST_list = match_matData2csvData(new_stData, new_stLabs, new_stDate[:,w_or_s], new_stTime[:,w_or_s], ST_id_matrix, w_or_s)  
    

#%%


def read_from_mat(data_folder, x_ts, x_features, new_x_list):
    
    offmore = 200

            
    all_header = []
    all_features = []
    all_Vx = []
    
    all_Vy = []
    
    all_Vabs = []
    
    all_Px = []
    
    all_Py = []
    
    all_Pabs = []
    
    new_TP = []
    error_events = []
    
    
    outORback = 0
    
    for k,exam_name in enumerate(new_x_list):
        
        armExp = 0
        
        print('exam=', k, data_folder)
        
        mat_contents = loadmat(os.path.join(data_folder, exam_name))
        exam_contents = mat_contents['r']
        
        sub_ID = int(exam_contents['file_name'].split("_")[0])
        
        armExp = exam_contents['file_label'].split("-")[2]
        if armExp==' RIGHT ':
            armExp = 1
        elif armExp==' LEFT ':
            armExp = 2
            
        # handDom = 
        
        
        TP_no = -1
        for trial_no in range(0, len(exam_contents['c3d'])):
            
            if exam_contents['c3d'][trial_no]['EXPERIMENT']['TASK_PROTOCOL']=='8 target 10cm reach': # Example: 3375 (10:45)
                outORback = 2
                
                if exam_contents['c3d'][trial_no]['TRIAL']['TP']!=1:
                    TP_no +=1 
                
                try:
                    # Try to find "TARGET_ON" event
                    try:
                        ind_target_on = np.where(np.asarray(exam_contents['c3d'][trial_no]['EVENTS']['LABELS']) == 'TARGET_ON')[0][0]
                        # Check the time and adjust units if necessary
                        if exam_contents['c3d'][trial_no]['EVENTS']['TIMES'][ind_target_on] > 500:  # assuming time is in ms if > 500
                            time_target_on = int(exam_contents['c3d'][trial_no]['EVENTS']['TIMES'][ind_target_on])
                        else:
                            time_target_on = int(exam_contents['c3d'][trial_no]['EVENTS']['TIMES'][ind_target_on] * 1000)
                    except IndexError:
                        # "TARGET_ON" does not exist; set default time
                        time_target_on = 200

                    # Try to find "END_OF_REACH" event
                    # try:
                    #     ind_end_of_reach = np.where(np.asarray(exam_contents['c3d'][trial_no]['EVENTS']['LABELS']) == 'END_OF_REACH')[0][0]
                    #     # Check the time and adjust units if necessary
                    #     if exam_contents['c3d'][trial_no]['EVENTS']['TIMES'][ind_end_of_reach] > 500:  # assuming time is in ms if > 500
                    #         time_end_of_reach = int(exam_contents['c3d'][trial_no]['EVENTS']['TIMES'][ind_end_of_reach])
                    #     else:
                    #         time_end_of_reach = int(exam_contents['c3d'][trial_no]['EVENTS']['TIMES'][ind_end_of_reach] * 1000)
                    # except IndexError:
                    #     # "END_OF_REACH" does not exist; set time to the end of the signal
                    #     time_end_of_reach = None  # Assuming the length of 'SIGNAL' represents the end time
                    
                    try:
                        if exam_contents['analysis']['TRIAL_FEATURES']['RT'][TP_no] >= 500:
                            time_offset = time_target_on + int(exam_contents['analysis']['TRIAL_FEATURES']['RT'][TP_no]) + int(exam_contents['analysis']['TRIAL_FEATURES']['TotalMT'][TP_no]) 
                        else:
                            time_offset = time_target_on + int(exam_contents['analysis']['TRIAL_FEATURES']['RT'][TP_no]*1000) + int(exam_contents['analysis']['TRIAL_FEATURES']['TotalMT'][TP_no]*1000)

                        
                    except:
                        time_offset = None # Assuming the length of 'SIGNAL' represents the end time
                        
                        
                    

                    
                except Exception as e:
                    # Append error details for this trial
                    error_events.append([exam_contents['file_name'], trial_no])
                    continue
                
                if exam_contents['c3d'][trial_no]['TRIAL']['TP']!=1 and (exam_contents['c3d'][trial_no]['TRIAL']['IS_ERROR']==0 or exam_contents['c3d'][trial_no]['TRIAL']['IS_ERROR']=='false'): # Detect catch trials and jump over them
                
                    try:
                        if armExp==1:
                            Vx = exam_contents['c3d'][trial_no]['Right_HandXVel'][(time_target_on - 200):(time_offset + offmore)]
                            Vy = exam_contents['c3d'][trial_no]['Right_HandYVel'][(time_target_on - 200):(time_offset + offmore)]
                            Vabs = np.sqrt(np.power(Vx, 2) + np.power(Vy, 2))
                            
                            
                            if exam_contents['c3d'][trial_no]['TRIAL']['TP']==2:
                                th = 0
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][1]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][1]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==3:
                                th = 315
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][2]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][2]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==4:
                                th = 270
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][3]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][3]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==5:
                                th = 225
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][4]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][4]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==6:
                                th = 180
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][5]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][5]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==7:
                                th = 135
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][6]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][6]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==8:
                                th = 90
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][7]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][7]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==9:
                                th = 45
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][8]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][8]/100
                                
                                
                            start_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][0]/100
                            start_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][0]/100
                            
                            if ( (abs(start_x)+abs(end_x))>2 or (abs(start_y)+abs(end_y))>2 ):
                                start_x = start_x/100
                                start_y = start_y/100
                                end_x = end_x/100
                                end_y = end_y/100
                                
                            # calculate the transform matrix
                            Trans_mat = np.asarray([[1, 0, -start_x], [0, 1, -start_y], [0, 0, 1]])
                            th = th*np.pi/180
                            Rot_mat = np.asarray([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
                            
                            # Transform the location signal
                            Px0 = exam_contents['c3d'][trial_no]['Right_HandX']
                            Py0 = exam_contents['c3d'][trial_no]['Right_HandY']
                            if Px0[0]>1:
                                Px0 = Px0/100
                                Py0 = Py0/100
                            Px = Rot_mat.dot(Trans_mat.dot(np.asarray([Px0,Py0, np.ones(len(Px0))])))[0, :][(time_target_on - 200):(time_offset + offmore)]
                            Py = Rot_mat.dot(Trans_mat.dot(np.asarray([Px0,Py0, np.ones(len(Px0))])))[1, :][(time_target_on - 200):(time_offset + offmore)]
                            
                            Pabs = np.sqrt(np.power(Px, 2) + np.power(Py, 2))
                                
                                
                            
                            
                            
                        elif armExp==2:
                            Vx = exam_contents['c3d'][trial_no]['Left_HandXVel'][(time_target_on - 200):(time_offset + offmore)]
                            Vy = exam_contents['c3d'][trial_no]['Left_HandYVel'][(time_target_on - 200):(time_offset + offmore)]
                            Vabs = np.sqrt(np.power(Vx, 2) + np.power(Vy, 2))
                            
                            
                            
                            if exam_contents['c3d'][trial_no]['TRIAL']['TP']==2:
                                th = 0
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][1]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][1]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==3:
                                th = 315
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][2]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][2]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==4:
                                th = 270
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][3]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][3]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==5:
                                th = 225
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][4]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][4]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==6:
                                th = 180
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][5]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][5]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==7:
                                th = 135
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][6]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][6]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==8:
                                th = 90
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][7]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][7]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==9:
                                th = 45
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][8]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][8]/100
                                
                                
                            start_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][0]/100
                            start_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][0]/100
                            
                            if ( (abs(start_x)+abs(end_x))>2 or (abs(start_y)+abs(end_y))>2 ):
                                start_x = start_x/100
                                start_y = start_y/100
                                end_x = end_x/100
                                end_y = end_y/100
                            # calculate the transform matrix
                            Trans_mat = np.asarray([[1, 0, -start_x], [0, 1, -start_y], [0, 0, 1]])
                            th = th*np.pi/180
                            Rot_mat = np.asarray([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
                            
                            # Transform the location signal
                            Px0 = exam_contents['c3d'][trial_no]['Left_HandX']
                            Py0 = exam_contents['c3d'][trial_no]['Left_HandY']
                            if Px0[0]>1:
                                Px0 = Px0/100
                                Py0 = Py0/100
                                
                            Px = Rot_mat.dot(Trans_mat.dot(np.asarray([Px0,Py0, np.ones(len(Px0))])))[0, :][(time_target_on - 200):(time_offset + offmore)]
                            Py = Rot_mat.dot(Trans_mat.dot(np.asarray([Px0,Py0, np.ones(len(Px0))])))[1, :][(time_target_on - 200):(time_offset + offmore)]
                            
                            Pabs = np.sqrt(np.power(Px, 2) + np.power(Py, 2))
                            
                            
                    except:
                        Vx = 0
                        Vy = 0
                        Vabs = 0
                        
                        Px = 0
                        Py = 0
                        Pabs = 0
                        
                        
                        
                        
                    all_Vx.append(Vx)
                    all_Vy.append(Vy)
                    all_Vabs.append(Vabs)
                    
                    
                    all_Px.append(Px)
                    all_Py.append(Py)
                    all_Pabs.append(Pabs)
                    
                    
                    all_header.append([sub_ID, exam_contents['c3d'][trial_no]['TRIAL']['TP'], armExp, x_ts[k], trial_no, outORback, exam_contents['c3d'][trial_no]['EXPERIMENT']['TASK_PROTOCOL'], exam_contents['file_name']])
                    all_features.append(x_features[k])
                        
                        
            elif exam_contents['c3d'][trial_no]['EXPERIMENT']['TASK_PROTOCOL']=='10cm 4target In&Out RT': # example: 3776 (11:23, both)
                
                if exam_contents['c3d'][trial_no]['TRIAL']['TP']!=1: # and TP_no==-1:
                    TP_no +=1
                # elif exam_contents['c3d'][trial_no]['TRIAL']['TP']!=1 and TP_no>-1:
                #     TP_no += 2
                
                
                if exam_contents['c3d'][trial_no]['TRIAL']['TP']!=1:
                    try:
                        ind_target_on_1 = np.where(np.asarray(exam_contents['c3d'][trial_no]['EVENTS']['LABELS'])=='TARGET_ON')[0][0]
                        if exam_contents['c3d'][trial_no]['EVENTS']['TIMES'][ind_target_on_1]>500:
                            time_target_on_1 = int(exam_contents['c3d'][trial_no]['EVENTS']['TIMES'][ind_target_on_1])
                            ind_target_on_2 = np.where(np.asarray(exam_contents['c3d'][trial_no]['EVENTS']['LABELS'])=='TARGET_ON')[0][1]
                            time_target_on_2 = int(exam_contents['c3d'][trial_no]['EVENTS']['TIMES'][ind_target_on_2])
                            
                            # ind_hold_at_target = np.where(np.asarray(exam_contents['c3d'][trial_no]['EVENTS']['LABELS'])=='HOLD_AT_TARGET')[0][0]
                            # time_hold_at_target = int(exam_contents['c3d'][trial_no]['EVENTS']['TIMES'][ind_hold_at_target]*1000)
                            
                            # ind_end_of_reach_1 = np.where(np.asarray(exam_contents['c3d'][trial_no]['EVENTS']['LABELS'])=='END_OF_REACH')[0][0]
                            # time_end_of_reach_1 = int(exam_contents['c3d'][trial_no]['EVENTS']['TIMES'][ind_end_of_reach_1])
                            
                            # ind_end_of_reach_2 = np.where(np.asarray(exam_contents['c3d'][trial_no]['EVENTS']['LABELS'])=='END_OF_REACH')[0][1]
                            # time_end_of_reach_2 = int(exam_contents['c3d'][trial_no]['EVENTS']['TIMES'][ind_end_of_reach_2])
                            
                                                    
                            
                                
                        else:
                            time_target_on_1 = int(exam_contents['c3d'][trial_no]['EVENTS']['TIMES'][ind_target_on_1]*1000)
                            ind_target_on_2 = np.where(np.asarray(exam_contents['c3d'][trial_no]['EVENTS']['LABELS'])=='TARGET_ON')[0][1]
                            time_target_on_2 = int(exam_contents['c3d'][trial_no]['EVENTS']['TIMES'][ind_target_on_2]*1000)
                            
                            # ind_hold_at_target = np.where(np.asarray(exam_contents['c3d'][trial_no]['EVENTS']['LABELS'])=='HOLD_AT_TARGET')[0][0]
                            # time_hold_at_target = int(exam_contents['c3d'][trial_no]['EVENTS']['TIMES'][ind_hold_at_target]*1000)
                            
                            # ind_end_of_reach_1 = np.where(np.asarray(exam_contents['c3d'][trial_no]['EVENTS']['LABELS'])=='END_OF_REACH')[0][0]
                            # time_end_of_reach_1 = int(exam_contents['c3d'][trial_no]['EVENTS']['TIMES'][ind_end_of_reach_1]*1000)
                            
                            # ind_end_of_reach_2 = np.where(np.asarray(exam_contents['c3d'][trial_no]['EVENTS']['LABELS'])=='END_OF_REACH')[0][1]
                            # time_end_of_reach_2 = int(exam_contents['c3d'][trial_no]['EVENTS']['TIMES'][ind_end_of_reach_2]*1000)
                            
                            
                        if exam_contents['analysis']['TRIAL_FEATURES']['RT'][TP_no] >= 500:
                            time_offset_1 = time_target_on_1 + int(exam_contents['analysis']['TRIAL_FEATURES']['RT'][TP_no]) + int(exam_contents['analysis']['TRIAL_FEATURES']['TotalMT'][TP_no]) 
                            TP_no += 1
                            
                            time_offset_2 = time_target_on_2 + int(exam_contents['analysis']['TRIAL_FEATURES']['RT'][TP_no]) + int(exam_contents['analysis']['TRIAL_FEATURES']['TotalMT'][TP_no]) 
                            # TP_no += 2
                        else:
                            time_offset_1 = time_target_on_1 + int(exam_contents['analysis']['TRIAL_FEATURES']['RT'][TP_no]*1000) + int(exam_contents['analysis']['TRIAL_FEATURES']['TotalMT'][TP_no]*1000)
                            TP_no += 1
                            
                            time_offset_2 = time_target_on_2 + int(exam_contents['analysis']['TRIAL_FEATURES']['RT'][TP_no]*1000) + int(exam_contents['analysis']['TRIAL_FEATURES']['TotalMT'][TP_no]*1000)
                            # TP_no += 2
                        
                        
                        
                            
                            
                    except:
                        # time_hold_at_target = 0
                        # time_target_on = 0
                        error_events.append([exam_contents['file_name'], trial_no])
                        continue
                
                if exam_contents['c3d'][trial_no]['TRIAL']['TP']!=1 and (exam_contents['c3d'][trial_no]['TRIAL']['IS_ERROR']==0 or exam_contents['c3d'][trial_no]['TRIAL']['IS_ERROR']=='false'): # Detect catch trials and jump over them
                
                    try:
                        if armExp==1:
                            Vx_out = exam_contents['c3d'][trial_no]['Right_HandXVel'][(time_target_on_1 - 200):(time_offset_1 + offmore)]
                            Vx_back = exam_contents['c3d'][trial_no]['Right_HandXVel'][(time_target_on_2 - 200):(time_offset_2 + offmore)]
                            Vy_out = exam_contents['c3d'][trial_no]['Right_HandYVel'][(time_target_on_1 - 200):(time_offset_1 + offmore)]
                            Vy_back = exam_contents['c3d'][trial_no]['Right_HandYVel'][(time_target_on_2 - 200):(time_offset_2 + offmore)]
                            Vabs_out = np.sqrt(np.power(Vx_out, 2) + np.power(Vy_out, 2))
                            Vabs_back = np.sqrt(np.power(Vx_back, 2) + np.power(Vy_back, 2))
                            
                            
                                                        
                            if exam_contents['c3d'][trial_no]['TRIAL']['TP']==2:
                                th = 0
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][1]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][1]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==3:
                                th1 = 315
                                th2 = 135
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][2]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][2]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==4:
                                th = 270
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][3]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][3]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==5:
                                th1 = 225
                                th2 = 45
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][4]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][4]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==6:
                                th = 180
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][5]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][5]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==7:
                                th1 = 135
                                th2 = 315
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][6]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][6]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==8:
                                th = 90
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][7]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][7]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==9:
                                th1 = 45
                                th2 = 225
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][8]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][8]/100
                                
                            # out    
                            start_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][0]/100
                            start_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][0]/100
                            
                            if ( (abs(start_x)+abs(end_x))>2 or (abs(start_y)+abs(end_y))>2 ):
                                start_x = start_x/100
                                start_y = start_y/100
                                end_x = end_x/100
                                end_y = end_y/100
                                                        
                                
                            # calculate the transform matrix
                            th = th1
                            Trans_mat = np.asarray([[1, 0, -start_x], [0, 1, -start_y], [0, 0, 1]])
                            th = th*np.pi/180
                            Rot_mat = np.asarray([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
                            
                            # Transform the location signal
                            Px0 = exam_contents['c3d'][trial_no]['Right_HandX']
                            Py0 = exam_contents['c3d'][trial_no]['Right_HandY']
                            if Px0[0]>1:
                                Px0 = Px0/100
                                Py0 = Py0/100
    
                            Px_out0 = Px0[(time_target_on_1 - 200):(time_offset_1 + offmore)]
                            Py_out0 = Py0[(time_target_on_1 - 200):(time_offset_1 + offmore)]
                            Px_out = Rot_mat.dot(Trans_mat.dot(np.asarray([Px_out0,Py_out0, np.ones(len(Px_out0))])))[0, :]
                            Py_out = Rot_mat.dot(Trans_mat.dot(np.asarray([Px_out0,Py_out0, np.ones(len(Px_out0))])))[1, :]
                            
                            Pabs_out = np.sqrt(np.power(Px_out, 2) + np.power(Py_out, 2))
                            
                            # back    
                            start_x = end_x
                            start_y = end_y
                            
                            if ( (abs(start_x)+abs(end_x))>2 or (abs(start_y)+abs(end_y))>2 ):
                                start_x = start_x/100
                                start_y = start_y/100
                                end_x = end_x/100
                                end_y = end_y/100
                            
                            # calculate the transform matrix
                            th = th2
                            Trans_mat = np.asarray([[1, 0, -start_x], [0, 1, -start_y], [0, 0, 1]])
                            th = th*np.pi/180
                            Rot_mat = np.asarray([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
                            
                            # Transform the location signal
                            Px0 = exam_contents['c3d'][trial_no]['Right_HandX']
                            Py0 = exam_contents['c3d'][trial_no]['Right_HandY']
                            if Px0[0]>1:
                                Px0 = Px0/100
                                Py0 = Py0/100
                                
                            Px_back0 = Px0[(time_target_on_2 - 200):(time_offset_2 + offmore)]
                            Py_back0 = Py0[(time_target_on_2 - 200):(time_offset_2 + offmore)]
                            Px_back = Rot_mat.dot(Trans_mat.dot(np.asarray([Px_back0,Py_back0, np.ones(len(Px_back0))])))[0, :]
                            Py_back = Rot_mat.dot(Trans_mat.dot(np.asarray([Px_back0,Py_back0, np.ones(len(Px_back0))])))[1, :]
                            
                            Pabs_back = np.sqrt(np.power(Px_back, 2) + np.power(Py_back, 2))
                            
                            
                            
                            
                                                        
                        elif armExp==2:
                            Vx_out = exam_contents['c3d'][trial_no]['Left_HandXVel'][(time_target_on_1 - 200):(time_offset_1 + offmore)]
                            Vx_back = exam_contents['c3d'][trial_no]['Left_HandXVel'][(time_target_on_2 - 200):(time_offset_2 + offmore)]
                            Vy_out = exam_contents['c3d'][trial_no]['Left_HandYVel'][(time_target_on_1 - 200):(time_offset_1 + offmore)]
                            Vy_back = exam_contents['c3d'][trial_no]['Left_HandYVel'][(time_target_on_2 - 200):(time_offset_2 + offmore)]
                            Vabs_out = np.sqrt(np.power(Vx_out, 2) + np.power(Vy_out, 2))
                            Vabs_back = np.sqrt(np.power(Vx_back, 2) + np.power(Vy_back, 2))
                            
                            
                            
                            
                            ##################################Nooshin
                            # # Calculate the full speed signal (before cropping) using the full velocity arrays.
                            # Vx_out_full = exam_contents['c3d'][trial_no]['Left_HandXVel']
                            # Vx_back_full = exam_contents['c3d'][trial_no]['Left_HandXVel']
                            # Vy_out_full = exam_contents['c3d'][trial_no]['Left_HandYVel']
                            # Vy_back_full = exam_contents['c3d'][trial_no]['Left_HandYVel']
                            # Vabs_out_full = np.sqrt(np.power(Vx_out_full, 2) + np.power(Vy_out_full, 2))
                            # Vabs_back_full = np.sqrt(np.power(Vx_back_full, 2) + np.power(Vy_back_full, 2))
                            
                            
                            # # -----------------------------
                            # # Ensure output directory exists
                            # # -----------------------------
                            # output_dir = 'old_vs_new'
                            # if not os.path.exists(output_dir):
                            #     os.makedirs(output_dir)

                            # # -----------------------------
                            # # Plot for "out" speed
                            # # -----------------------------
                            # fig, axs = plt.subplots(2, 1, figsize=(10, 8))

                            # # Subplot 1: Full Out Speed with Cropping Markers
                            # axs[0].plot(Vabs_out_full, label='Full Out Speed', color='blue')
                            # # Mark the cropping boundaries:
                            # axs[0].axvline(x=(time_target_on_1 - 200), color='red', linestyle='--', label='Crop Start')
                            # axs[0].axvline(x=time_offset_1, color='green', linestyle='--', label='Crop End')
                            # axs[0].set_title('Full Out Speed with Cropping Markers')
                            # axs[0].set_xlabel('Sample Index')
                            # axs[0].set_ylabel('Speed')
                            # axs[0].legend()
                            # axs[0].grid(True)

                            # # Subplot 2: Cropped Out Speed
                            # axs[1].plot(Vabs_out, label='Cropped Out Speed', color='orange')
                            # axs[1].set_title('Cropped Out Speed')
                            # axs[1].set_xlabel('Sample Index (Relative)')
                            # axs[1].set_ylabel('Speed')
                            # axs[1].legend()
                            # axs[1].grid(True)

                            # plt.tight_layout()
                            # out_save_path = os.path.join(output_dir, f'trial_{trial_no}_out_speed.png')
                            # plt.savefig(out_save_path)
                            # plt.close(fig)

                            # # -----------------------------
                            # # Plot for "back" speed
                            # # -----------------------------
                            # fig, axs = plt.subplots(2, 1, figsize=(10, 8))

                            # # Subplot 1: Full Back Speed with Cropping Markers
                            # axs[0].plot(Vabs_back_full, label='Full Back Speed', color='blue')
                            # # Mark the cropping boundaries:
                            # axs[0].axvline(x=(time_target_on_2 - 200), color='red', linestyle='--', label='Crop Start')
                            # axs[0].axvline(x=time_offset_2, color='green', linestyle='--', label='Crop End')
                            # axs[0].set_title('Full Back Speed with Cropping Markers')
                            # axs[0].set_xlabel('Sample Index')
                            # axs[0].set_ylabel('Speed')
                            # axs[0].legend()
                            # axs[0].grid(True)

                            # # Subplot 2: Cropped Back Speed
                            # axs[1].plot(Vabs_back, label='Cropped Back Speed', color='orange')
                            # axs[1].set_title('Cropped Back Speed')
                            # axs[1].set_xlabel('Sample Index (Relative)')
                            # axs[1].set_ylabel('Speed')
                            # axs[1].legend()
                            # axs[1].grid(True)

                            # plt.tight_layout()
                            # back_save_path = os.path.join(output_dir, f'trial_{trial_no}_back_speed.png')
                            # plt.savefig(back_save_path)
                            # plt.close(fig)

                            ##################################Nooshin
                                                                                  
                            
                            
                            if exam_contents['c3d'][trial_no]['TRIAL']['TP']==2:
                                th = 0
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][1]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][1]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==3:
                                th1 = 315
                                th2 = 135
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][2]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][2]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==4:
                                th = 270
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][3]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][3]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==5:
                                th1 = 225
                                th2 = 45
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][4]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][4]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==6:
                                th = 180
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][5]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][5]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==7:
                                th1 = 135
                                th2 = 315
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][6]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][6]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==8:
                                th = 90
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][7]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][7]/100
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==9:
                                th1 = 45
                                th2 = 225
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][8]/100 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][8]/100
                            
                            # out    
                            start_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][0]/100
                            start_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][0]/100
                            
                            if ( (abs(start_x)+abs(end_x))>2 or (abs(start_y)+abs(end_y))>2 ):
                                start_x = start_x/100
                                start_y = start_y/100
                                end_x = end_x/100
                                end_y = end_y/100
                                
                            # calculate the transform matrix
                            th = th1
                            Trans_mat = np.asarray([[1, 0, -start_x], [0, 1, -start_y], [0, 0, 1]])
                            th = th*np.pi/180
                            Rot_mat = np.asarray([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
                            
                            # Transform the location signal
                            Px0 = exam_contents['c3d'][trial_no]['Left_HandX']
                            Py0 = exam_contents['c3d'][trial_no]['Left_HandY']
                            if Px0[0]>1:
                                Px0 = Px0/100
                                Py0 = Py0/100
    
                            Px_out0 = Px0[(time_target_on_1 - 200):(time_offset_1 + offmore)]
                            Py_out0 = Py0[(time_target_on_1 - 200):(time_offset_1 + offmore)]
                            Px_out = Rot_mat.dot(Trans_mat.dot(np.asarray([Px_out0,Py_out0, np.ones(len(Px_out0))])))[0, :]
                            Py_out = Rot_mat.dot(Trans_mat.dot(np.asarray([Px_out0,Py_out0, np.ones(len(Px_out0))])))[1, :]
                            
                            Pabs_out = np.sqrt(np.power(Px_out, 2) + np.power(Py_out, 2))
                            
                            # back    
                            start_x = end_x
                            start_y = end_y
                            
                            if ( (abs(start_x)+abs(end_x))>2 or (abs(start_y)+abs(end_y))>2 ):
                                start_x = start_x/100
                                start_y = start_y/100
                                end_x = end_x/100
                                end_y = end_y/100
                            
                            # calculate the transform matrix
                            th = th2
                            Trans_mat = np.asarray([[1, 0, -start_x], [0, 1, -start_y], [0, 0, 1]])
                            th = th*np.pi/180
                            Rot_mat = np.asarray([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
                            
                            # Transform the location signal
                            Px0 = exam_contents['c3d'][trial_no]['Left_HandX']
                            Py0 = exam_contents['c3d'][trial_no]['Left_HandY']
                            if Px0[0]>1:
                                Px0 = Px0/100
                                Py0 = Py0/100
    
                            Px_back0 = Px0[(time_target_on_2 - 200):(time_offset_2 + offmore)]
                            Py_back0 = Py0[(time_target_on_2 - 200):(time_offset_2 + offmore)]
                            Px_back = Rot_mat.dot(Trans_mat.dot(np.asarray([Px_back0,Py_back0, np.ones(len(Px_back0))])))[0, :]
                            Py_back = Rot_mat.dot(Trans_mat.dot(np.asarray([Px_back0,Py_back0, np.ones(len(Px_back0))])))[1, :]
                            
                            Pabs_back = np.sqrt(np.power(Px_back, 2) + np.power(Py_back, 2))
                            
                            
                            
                    except:
                        Vx_out = 0
                        Vx_back = 0
                        Vy_out = 0
                        Vy_back = 0
                        Vabs_out = 0
                        Vabs_back = 0
                        
                        Px_out = 0
                        Px_back = 0
                        Py_out = 0
                        Py_back = 0
                        Pabs_out = 0
                        Pabs_back = 0
                        
            
                    all_Vx.append(Vx_out)
                    all_Vx.append(Vx_back)
                    all_Vy.append(Vy_out)
                    all_Vy.append(Vy_back)
                    all_Vabs.append(Vabs_out)
                    all_Vabs.append(Vabs_back)
                    
                    
                    all_Px.append(Px_out)
                    all_Px.append(Px_back)
                    all_Py.append(Py_out)
                    all_Py.append(Py_back)
                    all_Pabs.append(Pabs_out)
                    all_Pabs.append(Pabs_back)
                    
                    
                    outORback = 0
                    all_header.append([sub_ID, exam_contents['c3d'][trial_no]['TRIAL']['TP'], armExp, x_ts[k], trial_no, outORback, exam_contents['c3d'][trial_no]['EXPERIMENT']['TASK_PROTOCOL'], exam_contents['file_name']])
                    all_features.append(x_features[k])
                    outORback = 1
                    all_header.append([sub_ID, exam_contents['c3d'][trial_no]['TRIAL']['TP'], armExp, x_ts[k], trial_no, outORback, exam_contents['c3d'][trial_no]['EXPERIMENT']['TASK_PROTOCOL'], exam_contents['file_name']])
                    all_features.append(x_features[k])
            
            
            elif exam_contents['c3d'][trial_no]['EXPERIMENT']['TASK_PROTOCOL']=='reachout_std_(8_target)': # Example: 343 632(13:40??)
                outORback = 2
                
                if exam_contents['c3d'][trial_no]['TRIAL']['TP']!=1:
                    TP_no +=1
            
                try:
                    try:
                        ind_target_on = np.where(np.asarray(exam_contents['c3d'][trial_no]['EVENTS']['LABELS']) == 'TARGET_ON')[0][0]
                        # Check the time and adjust units if necessary
                        if exam_contents['c3d'][trial_no]['EVENTS']['TIMES'][ind_target_on] > 500:  # assuming time is in ms if > 500
                            time_target_on = int(exam_contents['c3d'][trial_no]['EVENTS']['TIMES'][ind_target_on])
                        else:
                            time_target_on = int(exam_contents['c3d'][trial_no]['EVENTS']['TIMES'][ind_target_on] * 1000)
                    except IndexError:
                        # "TARGET_ON" does not exist; set default time
                        time_target_on = 200
                        
                        
                        
                        
                    try:
                        if exam_contents['analysis']['TRIAL_FEATURES']['RT'][TP_no] >= 500:
                            time_offset = time_target_on + int(exam_contents['analysis']['TRIAL_FEATURES']['RT'][TP_no]) + int(exam_contents['analysis']['TRIAL_FEATURES']['TotalMT'][TP_no]) 
                        else:
                            time_offset = time_target_on + int(exam_contents['analysis']['TRIAL_FEATURES']['RT'][TP_no]*1000) + int(exam_contents['analysis']['TRIAL_FEATURES']['TotalMT'][TP_no]*1000)

                        
                    except:
                        time_offset = None # Assuming the length of 'SIGNAL' represents the end time
                        
                        
                except Exception as e:
                    # time_hold_at_target = 0
                    # time_target_on = 0
                    error_events.append([exam_contents['file_name'], trial_no])
                    continue
                
                if exam_contents['c3d'][trial_no]['TRIAL']['TP']!=1 and (exam_contents['c3d'][trial_no]['TRIAL']['IS_ERROR']==0 or exam_contents['c3d'][trial_no]['TRIAL']['IS_ERROR']=='false'): # Detect catch trials and jump over them
                
                    try:
                        if armExp==1:
                            Vx = exam_contents['c3d'][trial_no]['Right_HandXVel'][(time_target_on - 200):(time_offset + offmore)]
                            Vy = exam_contents['c3d'][trial_no]['Right_HandYVel'][(time_target_on - 200):(time_offset + offmore)]
                            Vabs = np.sqrt(np.power(Vx, 2) + np.power(Vy, 2))
                            
                            
                            
                            if exam_contents['c3d'][trial_no]['TRIAL']['TP']==2:
                                th = 0
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][1] 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][1]
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==3:
                                th = 315
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][2] 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][2]
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==4:
                                th = 270
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][3] 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][3]
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==5:
                                th = 225
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][4] 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][4]
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==6:
                                th = 180
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][5] 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][5]
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==7:
                                th = 135
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][6] 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][6]
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==8:
                                th = 90
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][7] 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][7]
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==9:
                                th = 45
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][8] 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][8]
                                
                                
                            start_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][0]
                            start_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][0]
                            
                            # check for units (cm/m), assuming if one of the entries is in cm, others are so as well.
                            if ( (abs(start_x)+abs(end_x))>2 or (abs(start_y)+abs(end_y))>2 ):
                                start_x = start_x/100
                                start_y = start_y/100
                                end_x = end_x/100
                                end_y = end_y/100
                            
                            # calculate the transform matrix
                            Trans_mat = np.asarray([[1, 0, -start_x], [0, 1, -start_y], [0, 0, 1]])
                            th = th*np.pi/180
                            Rot_mat = np.asarray([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
                            
                            # Transform the location signal
                            Px0 = exam_contents['c3d'][trial_no]['Right_HandX']
                            Py0 = exam_contents['c3d'][trial_no]['Right_HandY']
                            if Px0[0]>1:
                                Px0 = Px0/100
                                Py0 = Py0/100
    
                            Px = Rot_mat.dot(Trans_mat.dot(np.asarray([Px0,Py0, np.ones(len(Px0))])))[0, :][(time_target_on - 200):(time_offset + offmore)]
                            Py = Rot_mat.dot(Trans_mat.dot(np.asarray([Px0,Py0, np.ones(len(Px0))])))[1, :][(time_target_on - 200):(time_offset + offmore)]
                            
                            Pabs = np.sqrt(np.power(Px, 2) + np.power(Py, 2))
                            
                            
                                                        
                        elif armExp==2:
                            Vx = exam_contents['c3d'][trial_no]['Left_HandXVel'][(time_target_on - 200):(time_offset + offmore)]
                            Vy = exam_contents['c3d'][trial_no]['Left_HandYVel'][(time_target_on - 200):(time_offset + offmore)]
                            Vabs = np.sqrt(np.power(Vx, 2) + np.power(Vy, 2))
                            
                            
                            
                            if exam_contents['c3d'][trial_no]['TRIAL']['TP']==2:
                                th = 0
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][1] 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][1]
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==3:
                                th = 315
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][2] 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][2]
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==4:
                                th = 270
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][3] 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][3]
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==5:
                                th = 225
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][4] 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][4]
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==6:
                                th = 180
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][5] 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][5]
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==7:
                                th = 135
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][6] 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][6]
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==8:
                                th = 90
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][7] 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][7]
                            elif exam_contents['c3d'][trial_no]['TRIAL']['TP']==9:
                                th = 45
                                end_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][8] 
                                end_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][8]
                                
                                
                            start_x = exam_contents['c3d'][trial_no]['TARGET_TABLE']['X_GLOBAL'][0]
                            start_y = exam_contents['c3d'][trial_no]['TARGET_TABLE']['Y_GLOBAL'][0]
                            
                            # check for units (cm/m), assuming if one of the entries is in cm, others are so as well.
                            if ( (abs(start_x)+abs(end_x))>2 or (abs(start_y)+abs(end_y))>2 ):
                                start_x = start_x/100
                                start_y = start_y/100
                                end_x = end_x/100
                                end_y = end_y/100
                            
                                
                                
                            # calculate the transform matrix
                            Trans_mat = np.asarray([[1, 0, -start_x], [0, 1, -start_y], [0, 0, 1]])
                            th = th*np.pi/180
                            Rot_mat = np.asarray([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
                            
                            # Transform the location signal
                            Px0 = exam_contents['c3d'][trial_no]['Left_HandX']
                            Py0 = exam_contents['c3d'][trial_no]['Left_HandY']
                            if Px0[0]>1:
                                Px0 = Px0/100
                                Py0 = Py0/100
                                
                            Px = Rot_mat.dot(Trans_mat.dot(np.asarray([Px0,Py0, np.ones(len(Px0))])))[0, :][(time_target_on - 200):(time_offset + offmore)]
                            Py = Rot_mat.dot(Trans_mat.dot(np.asarray([Px0,Py0, np.ones(len(Px0))])))[1, :][(time_target_on - 200):(time_offset + offmore)]
                            
                            Pabs = np.sqrt(np.power(Px, 2) + np.power(Py, 2))
                            
                            
                    except:
                        Vx = 0
                        Vy = 0
                        Vabs = 0
                        
                        Px = 0
                        Py = 0
                        Pabs = 0
                        
                        
                        
                    all_Vx.append(Vx)
                    all_Vy.append(Vy)
                    all_Vabs.append(Vabs)
                    
                    
                    all_Px.append(Px)
                    all_Py.append(Py)
                    all_Pabs.append(Pabs)
                    
                    
                    all_header.append([sub_ID, exam_contents['c3d'][trial_no]['TRIAL']['TP'], armExp, x_ts[k], trial_no, outORback, exam_contents['c3d'][trial_no]['EXPERIMENT']['TASK_PROTOCOL'],  exam_contents['file_name']])
                    all_features.append(x_features[k])
            
            
            else:
                new_TP.append([exam_contents['file_name'], trial_no])
            
    return new_TP, error_events, all_header, all_features, new_TP, all_Vx, all_Vy, all_Vabs, all_Px, all_Py, \
        all_Pabs
#%%
# FAR: VandP files are the ones with the rotation of the P signals.
# CT_mat_folder = "C:/Users/fakbarifar/OneDrive - Queen's University/Faranak/control_control_mat" 
new_TP_CT, error_events_CT, all_header_CT, all_features_CT, new_TP_CT, all_Vx_CT, all_Vy_CT, all_Vabs_CT, all_Px_CT, all_Py_CT, \
all_Pabs_CT = read_from_mat(CT_mat_folder, CT_ts, CT_features, new_CT_list)
# np.savez('new_control_raw_VandP_Pcorrected_later', all_header_CT=all_header_CT, all_features_CT=all_features_CT, new_TP_CT=new_TP_CT, all_Vx_CT=all_Vx_CT, all_Vy_CT=all_Vy_CT, all_Vabs_CT=all_Vabs_CT, all_Vx_T_ON_CT=all_Vx_T_ON_CT, all_Vy_T_ON_CT=all_Vy_T_ON_CT, all_Vabs_T_ON_CT=all_Vabs_T_ON_CT, all_Px_CT=all_Px_CT, all_Py_CT=all_Py_CT, \
# all_Pabs_CT=all_Pabs_CT, all_Px_T_ON_CT=all_Px_T_ON_CT, all_Py_T_ON_CT=all_Py_T_ON_CT, all_Pabs_T_ON_CT=all_Pabs_T_ON_CT, allow_pickle=True)
       
# ST_mat_folder = "C:/Users/fakbarifar/OneDrive - Queen's University/Faranak/stroke_stroke_mat" 
new_TP_ST, error_events_ST, all_header_ST, all_features_ST, new_TP_ST, all_Vx_ST, all_Vy_ST, all_Vabs_ST, all_Px_ST, all_Py_ST, \
all_Pabs_ST = read_from_mat(ST_mat_folder, ST_ts, ST_features, new_ST_list) 
# np.savez('new_stroke_raw_VandP_Pcorrected_later', all_header_ST=all_header_ST, all_features_ST=all_features_ST, new_TP_ST=new_TP_ST, all_Vx_ST=all_Vx_ST, all_Vy_ST=all_Vy_ST, all_Vabs_ST=all_Vabs_ST, all_Vx_T_ON_ST=all_Vx_T_ON_ST, all_Vy_T_ON_ST=all_Vy_T_ON_ST, all_Vabs_T_ON_ST=all_Vabs_T_ON_ST, all_Px_ST=all_Px_ST, all_Py_ST=all_Py_ST, \
# all_Pabs_ST=all_Pabs_ST, all_Px_T_ON_ST=all_Px_T_ON_ST, all_Py_T_ON_ST=all_Py_T_ON_ST, all_Pabs_T_ON_ST=all_Pabs_T_ON_ST)

#%%
import numpy as np

# ── helper ─────────────────────────────────────────────────────────────────────
def _is_bad_vabs(entry):
    """
    Return True for entries you want to delete.
      • Plain scalar 0   → bad  
      • 0‑length array   → bad  
      • Anything else    → good
    """
    if isinstance(entry, np.ndarray):
        return entry.size == 0                      # empty ndarray
    return entry == 0                               # scalar or list element

# ── main filtering ─────────────────────────────────────────────────────────────
def filter_bad_trials(
        all_header_CT,
        all_features_CT,
        all_Vabs_CT,
        all_Pabs_CT,
):
    keep_idx = [i for i, v in enumerate(all_Vabs_CT) if not _is_bad_vabs(v)]

    # re‑index every parallel list
    all_header_CT   = [all_header_CT[i]   for i in keep_idx]
    all_features_CT = [all_features_CT[i] for i in keep_idx]
    all_Vabs_CT     = [all_Vabs_CT[i]     for i in keep_idx]
    all_Pabs_CT     = [all_Pabs_CT[i]     for i in keep_idx]

    return all_header_CT, all_features_CT, all_Vabs_CT, all_Pabs_CT

# ── usage ──────────────────────────────────────────────────────────────────────
(all_header_CT,
 all_features_CT,
 all_Vabs_CT,
 all_Pabs_CT) = filter_bad_trials(all_header_CT,
                                  all_features_CT,
                                  all_Vabs_CT,
                                  all_Pabs_CT)
                                  
                                  
                                  
(all_header_ST,
 all_features_ST,
 all_Vabs_ST,
 all_Pabs_ST) = filter_bad_trials(all_header_ST,
                                  all_features_ST,
                                  all_Vabs_ST,
                                  all_Pabs_ST)

#%% save to h5py file

def convert_header_format(header_element):
    # Extract the individual components from the header
    subject_id = str(header_element[0])
    trial_tp = str(header_element[1])
    hand_used = str(header_element[2])
    nested_array = header_element[3]  # This is the array we want to expand
    array_elements = [str(x) for x in nested_array]  # Convert each element of the array to a string
    other_values = header_element[4:7]
    description = header_element[7]
    

    # Convert other values to strings
    other_values_strings = [str(x) for x in other_values]

    # Concatenate all components into a single list
    formatted_header = [subject_id, trial_tp, hand_used] + array_elements + other_values_strings + [description]

    return formatted_header


# Convert each element in the list
converted_headers_CT = [convert_header_format(header) for header in all_header_CT]
converted_headers_ST = [convert_header_format(header) for header in all_header_ST]




# Function to save variable length data in HDF5
def save_variable_length_data(h5file, dataset_name, data):
    grp = h5file.create_group(dataset_name)
    for i, item in enumerate(data):
        grp.create_dataset(str(i), data=item, compression="gzip")

# def save_variable_length_data(h5file, dataset_name, data):
#     grp = h5file.create_group(dataset_name)
#     for i, item in enumerate(data):
#         arr_item = np.array(item)
#         # Convert Unicode strings to byte strings if needed
#         if arr_item.dtype.kind == 'U':
#             arr_item = arr_item.astype('S')
#         # If the item is scalar, create the dataset without compression
#         if arr_item.ndim == 0:
#             grp.create_dataset(str(i), data=arr_item)
#         else:
#             grp.create_dataset(str(i), data=arr_item, compression="gzip")



# Create a new HDF5 file and save the data for control
with h5py.File('new_control_raw_VandP_Pcorrected_cropped_Mar_2025_adv_200more.h5', 'w') as f:
    save_variable_length_data(f, 'all_header_CT', converted_headers_CT)
    save_variable_length_data(f, 'all_features_CT', all_features_CT)
    # save_variable_length_data(f, 'new_TP_CT', new_TP_CT)
    # save_variable_length_data(f, 'all_Vx_CT', all_Vx_CT)
    # save_variable_length_data(f, 'all_Vy_CT', all_Vy_CT)
    save_variable_length_data(f, 'all_Vabs_CT', all_Vabs_CT)
    # save_variable_length_data(f, 'all_Vx_T_ON_CT', all_Vx_T_ON_CT)
    # save_variable_length_data(f, 'all_Vy_T_ON_CT', all_Vy_T_ON_CT)
    # save_variable_length_data(f, 'all_Vabs_T_ON_CT', all_Vabs_T_ON_CT)
    # save_variable_length_data(f, 'all_Px_CT', all_Px_CT)
    # save_variable_length_data(f, 'all_Py_CT', all_Py_CT)
    save_variable_length_data(f, 'all_Pabs_CT', all_Pabs_CT)
    # save_variable_length_data(f, 'all_Px_T_ON_CT', all_Px_T_ON_CT)
    # save_variable_length_data(f, 'all_Py_T_ON_CT', all_Py_T_ON_CT)
    # save_variable_length_data(f, 'all_Pabs_T_ON_CT', all_Pabs_T_ON_CT)

# Create a new HDF5 file and save the data for stroke
with h5py.File('new_stroke_raw_VandP_Pcorrected_cropped_Mar_2025_adv_200more.h5', 'w') as f:
    save_variable_length_data(f, 'all_header_ST', converted_headers_ST)
    save_variable_length_data(f, 'all_features_ST', all_features_ST)
    save_variable_length_data(f, 'new_TP_ST', new_TP_ST)
    # save_variable_length_data(f, 'all_Vx_ST', all_Vx_ST)
    # save_variable_length_data(f, 'all_Vy_ST', all_Vy_ST)
    save_variable_length_data(f, 'all_Vabs_ST', all_Vabs_ST)
    # save_variable_length_data(f, 'all_Vx_T_ON_ST', all_Vx_T_ON_ST)
    # save_variable_length_data(f, 'all_Vy_T_ON_ST', all_Vy_T_ON_ST)
    # save_variable_length_data(f, 'all_Vabs_T_ON_ST', all_Vabs_T_ON_ST)
    # save_variable_length_data(f, 'all_Px_ST', all_Px_ST)
    # save_variable_length_data(f, 'all_Py_ST', all_Py_ST)
    save_variable_length_data(f, 'all_Pabs_ST', all_Pabs_ST)
    # save_variable_length_data(f, 'all_Px_T_ON_ST', all_Px_T_ON_ST)
    # save_variable_length_data(f, 'all_Py_T_ON_ST', all_Py_T_ON_ST)
    # save_variable_length_data(f, 'all_Pabs_T_ON_ST', all_Pabs_T_ON_ST)




#%%
















    






























