opencv2_v3_flag = 0
from PIL import Image
import numpy as np
import math
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
if(opencv2_v3_flag==0):
    import cv2.cv as cv
import skimage.io
import warnings
from sklearn.cluster import KMeans
import pandas as pd
import time
warnings.filterwarnings("ignore")
import os
import os.path
import zerorpc
import scipy.interpolate
import sys
import glob
import persomics_image_analysis_function as pia


############ command line function
### Example: python annotation_from_reference_extenal.py 0 1 asdf annotation_Set_1.csv  ### memory free, offline
### Example: python annotation_from_reference_extenal.py 0 0 asdf annotation_Set_1.csv  ### memory free, online

user_input_args = sys.argv[1:]  #### [1]: memory switch [2]: state: 0- online, 1-offline [3]: input_folder [4]: csv_in_filename_ref

local_dir = '/home/karin/server'
dir_local_csv_database = '/home/karin/server/annotation_database'

mem_switch = int(user_input_args[0])   ### memory 1: memory constrained, 0: memory free # 
online_offline_state = int(user_input_args[1])

####################################
print ('=' * 30)
if online_offline_state == 0:
    print '...online mode'
else:
    print '...offline mode'
if mem_switch == 0:
    print '...memory free mode'
else:
    print '...memory constrained mode'

if online_offline_state == 0:
    #====================================================
    #Run spot detection on server
    #====================================================
    tcp_adress = "tcp://172.31.31.65:3000"
    server_obj = pia.RPC_step_3(local_dir, dir_local_csv_database, mem_switch)
    s = zerorpc.Server(server_obj)
    s.bind(tcp_adress)
    print '...server is up'
    print '...project path: ' + local_dir
    print '...path of annotation database: ' + dir_local_csv_database
    print ('=' * 30)
    s.run()

elif online_offline_state == 1:
    #====================================================
    #Run spot detection without server
    #====================================================
    input_folder = user_input_args[2] 
    csv_in_filename_ref = user_input_args[3]

    print '...project path: ' + local_dir
    print '...path of annotation database: ' + dir_local_csv_database
    print '...name of call: ' + input_folder
    print '...name of reference annotation file: ' + csv_in_filename_ref
    print ('=' * 30)

    pia.annotation_from_reference_step_3_local(local_dir, dir_local_csv_database, csv_in_filename_ref, mem_switch, input_folder)
