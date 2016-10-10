#====================================================
#====================================================
#====================================================
#====================================================
#====================================================
#README
#====================================================
#====================================================
#====================================================
#====================================================
#====================================================

#Guide to Python and extension packages installation-----------------------------
#
#The python backend requires the installation of Python and extension packages. It supports 64 bits Python 2.7x. Some related packages are #also required. Please install them following the instruction here. Note that the instruction here is for OS X 64 bits systems. If using 32 #bits systems, similar setting can also be configured. In the following, we list the resources of necessary packages for running the Python #backend.   
#
#The necessary packages are listed in the below Import Section in this file (persomics_image_analysis.py). Here we demostrate the steps of installing Python and related packages. Also, we choose to include, as references, official/unofficial installation guides for packets to be installed.
#
#1. Download and install 64 bits Python 2.7x. Follow instructions at
#https://www.python.org/download/releases/2.7/
#
#2. Install pip
#Download get-pip.py from https://pip.pypa.io/en/latest/installing/#install-pip
#and run: 
#python get-pip.py
#
#3. Install math using pip
#pip install math
#
#4. Install numpy, as described in http://docs.scipy.org/doc/numpy-1.10.1/user/install.html  
#pip install numpy
#
#5. Install pandas, as described in http://pandas.pydata.org/pandas-docs/stable/install.html
#pip install pandas
#
#6. Install warnings using pip
#pip install warnings 
#
#7. Install OpenCV 2.4.9, as decribed in https://jjyap.wordpress.com/2014/05/24/installing-opencv-2-4-9-on-mac-osx-with-python-support/
#brew install opencv
#Make sure to set opencv2_v3_flag=0 in the Import Section below in this file (persomics_image_analysis.py).
#
#Observe that you can install the new version of OpenCV, Opencv 3.x. In that case, you should set opencv2_v3_flag=1 in the Import Section below in this file (persomics_image_analysis.py). This is not needed for the Persomics Folder app.
#
#8. Install Pillow, as described in http://pillow.readthedocs.io/en/3.2.x/installation.html
#pip install Pillow
#
#9. Install scipy, as described in https://penandpants.com/2012/02/24/install-python/ 
#brew install gfortran
#pip install scipy
#
#10. Install scikit-image, as described in http://scikit-image.org/download.html: 
#pip install -U scikit-image
#
#11. Install scikit-learn using pip, as described in http://scikit-learn.org/stable/install.html
#pip install -U scikit-learn
#
#12. Install zerorpc, as described in http://www.zerorpc.io
#sudo pip install zerorpc


# Guide to running the algorithm Python code---------------------
# 1. Add directories named 'uploads', 'processing' and 'downloads' in the same catalogue as the script file persomics_image_analysis.py
# 2. If running the algorithm on a server with the node.js server, edit persomics_image_analysis.py to configure server ip adress and port:
#   tcp_adress="tcp://server_ip:server_port"
# 3. Run the algorithm using the command prompt:
# > python persomics_image_analysis.py [memory_switch] [server_state] [local_input_dir]
# memory_switch
#   0 => Memory free mode
#   1 => Memory constrained
# server_state
#   0 => Run algorithm with node.js on server
#   1 => Run algorithm localy, also state a local_input_dir for input data files
# local_input_dir
#   The directory where the algorithm will find the data to analysis, when using local server state. The input annotation file and montage image, with arbitrary names, must be in the folder dir_local/uploads/input_folder when the program is run.

# Examples:
# online, memory free:         python persomics_image_analysis.py 0 0
# online, memory constrained:  python persomics_image_analysis.py 1 0
# offline, memory free:        python persomics_image_analysis.py 0 1 asdf
# offline, memory constrained: python persomics_image_analysis.py 1 1 asdf


#Guide to the algorithm files--------------------------------------
#persomics_image_analysis.py: run scripts and functions as described above
#
#spot_classification : additional functions









#====================================================
#====================================================
#====================================================
#====================================================
#====================================================
#Import
#====================================================
#====================================================
#====================================================
#====================================================
#====================================================
opencv2_v3_flag = 0
from PIL import Image
import numpy as np
import math
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
if(opencv2_v3_flag==0):
    import cv2.cv as cv
#END if
import skimage.io
import warnings
from sklearn.cluster import KMeans
import pandas as pd
import time
warnings.filterwarnings("ignore")
import zerorpc
import spot_classification
import os
import os.path 
import scipy.interpolate
import sys

#====================================================
#====================================================
#====================================================
#====================================================
#====================================================
#Functions
#====================================================
#====================================================
#====================================================
#====================================================
#====================================================

def tiff_image_read_into_tiles(path_im, image_in_filename, path_im_tiles, tile_size):
    img = Image.open(path_im+'/'+image_in_filename)
    img.seek(0)  ### only use red channel
    im_heigh = img.tag[0x101][0]
    im_width =  img.tag[0x100][0]

    tile_row = int(math.ceil(im_heigh/tile_size))
    tile_col = int(math.ceil(im_width/tile_size))

    print 'im_heigh: ' + str(im_heigh) + ', im_width: '+ str(im_width)
    tile_ind = 0
    for r_i in range(0,tile_row):
        for c_i in range(0,tile_col):
            box = (c_i*tile_size, r_i*tile_size, (c_i+1)*tile_size, (r_i+1)*tile_size)
            img.crop(box).save(path_im_tiles+'/'+'tile_by_tile_'+'r_'+str(r_i)+'_c_'+str(c_i)+'_'+image_in_filename, bits=8)
            tile_ind  = tile_ind + 1

    return tile_row, tile_col

def tiff_image_resize(path_im_tiles, path_im_resize, image_in_filename, tile_size, resize_factor, tile_row, tile_col):
    tile_resize = tile_size*resize_factor
    tile_img_resize = np.zeros([tile_resize,tile_resize])
    img_resize = np.zeros([tile_resize*tile_row,tile_resize*tile_col], dtype="uint8")

    for r_i in range(0,tile_row):
        for c_i in range(0,tile_col):
            tile_img = cv2.imread(path_im_tiles+'/'+'tile_by_tile_'+'r_'+str(r_i)+'_c_'+str(c_i)+'_'+image_in_filename,0)
            tile_img_resize = cv2.resize(tile_img, (0,0), fx=resize_factor, fy=resize_factor)
            img_resize[r_i*tile_resize:(r_i+1)*tile_resize,c_i*tile_resize:(c_i+1)*tile_resize] = tile_img_resize

    img_resize = cv2.equalizeHist(img_resize)
    cv2.imwrite(path_im_resize+'/'+'resized_red_f10_'+image_in_filename,img_resize)

def tiff_image_preprocessing(path_im, image_in_filename, path_im_tiles, tile_size, path_im_resize, resize_factor, mem_switch):
    if mem_switch == 1: ### use the memory constrained method
        tile_row, tile_col = tiff_image_read_into_tiles(path_im, image_in_filename, path_im_tiles, tile_size)
        tiff_image_resize(path_im_tiles, path_im_resize, image_in_filename, tile_size, resize_factor, tile_row, tile_col)
    else: ### memory free
        img = skimage.io.imread(os.path.join(path_im,image_in_filename), plugin='tifffile')
        try: ### for skimage (0.12.x)
            img_single_channel = skimage.img_as_ubyte(img[:,:,0], force_copy=False)
            img_resize = cv2.resize(img_single_channel, (int(img.shape[1]*resize_factor), int(img.shape[0]*resize_factor)))
        except: ### for skimage (0.9.x)
            img_single_channel = skimage.img_as_ubyte(img[0,:,:], force_copy=False)
            img_resize = cv2.resize(img_single_channel, (int(img.shape[2]*resize_factor), int(img.shape[1]*resize_factor)))
        img_resize = cv2.equalizeHist(img_resize)
        cv2.imwrite(path_im_resize+'/'+'resized_red_f10_'+image_in_filename,img_resize)

def median_color_detection(cimg):
    sample_size = 20 
    im_pob = cimg[0:sample_size,0:sample_size].reshape((sample_size * sample_size, 1))
    kmeans_clustering = KMeans(n_clusters = 2)
    idx = kmeans_clustering.fit_predict(im_pob.reshape(-1,1))
    center = kmeans_clustering.cluster_centers_.tolist()
    value_median = int(max(center)[0])
    #print 'value_median: ' + str(value_median)
    return value_median

def circle_detection_preprocessing(cimg):
    hight,width = cimg.shape
    noise_higher_bound = 50
    value_median = median_color_detection(cimg)
    for i in range(0, hight):
        for j in range(0, width):
            im_value = cimg[i,j]
            if im_value < noise_higher_bound:
                cimg[i,j] = value_median
    (_, cimg) = cv2.threshold(cimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cimg

def circle_detection(path_im_resize, image_resized_filename, path_im_visual_detection, target_cir_num, stop_thr, min_rad, max_rad, dp, min_dist, min_para2, max_para2, number_of_column):
    img = cv2.imread(path_im_resize+'/'+image_resized_filename,0)
    cimg = cv2.bilateralFilter(img,min_dist,50,50)
    cimg = circle_detection_preprocessing(cimg)
    #cv2.imwrite(path_im_resize+'/'+'binary_'+image_resized_filename,cimg)
    para2_candidate = range(min_para2, max_para2, 1)
    para2_candidate = para2_candidate[::-1]
    for i in range(len(para2_candidate)):
        para2_i = para2_candidate[i]
        if(opencv2_v3_flag==0):
            circles = cv2.HoughCircles(cimg,cv.CV_HOUGH_GRADIENT,dp, min_dist, param1=10, param2=para2_i, minRadius= min_rad, maxRadius=max_rad)  ### for earlier version (2.x) of cv2
        else:
            circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,dp, min_dist, param1=10, param2=para2_i, minRadius= min_rad, maxRadius=max_rad)  ### for cv2 3.x
        #END
        if circles is not None:
            cir_num = len(circles[0])
        else:
            cir_num = 0
        if abs(cir_num - target_cir_num) <= stop_thr:
            break
    
    print 'number of circles: '+str(cir_num)

    ###### sort circles into columns and rows
    if circles is not None:
        circles_col = circles[0,:,0]
        circles_row = circles[0,:,1]
        cluster_num = number_of_column #### number of column
        kmeans_clustering = KMeans(n_clusters = cluster_num)
        idx = kmeans_clustering.fit_predict(circles_col.reshape(-1,1)) #kmeans label for each circle
        center = kmeans_clustering.cluster_centers_.tolist() #kmeans centers in pixel units
        center_order = sorted(center) #sorted kmeans centers in pixel units
        cir_num = len(circles[0])
        circles_info =  np.zeros([cir_num,5], dtype="int") #### [col_pixel, row_pixel, rad, col_ind, row_ind]
        circles_info_sorted =  np.zeros([cir_num,5], dtype="int") ### sorted by columns and rows

        for c in range(0,cir_num): #go through circles
            indx_circle = idx[c] #kmeans label for circle
            center_circle = center[indx_circle] #kmeans center in pixel units for circle
            col_indx_circle_ordered = center_order.index(center_circle) #sorted kmeans label for circle (the sorted kmeans label is the only thing that is interesting, i.e., the unsorted label is not interesting)
            circles_info[c,0] = circles[0,c][0]
            circles_info[c,1] = circles[0,c][1]
            circles_info[c,2] = circles[0,c][2]
            circles_info[c,3] = col_indx_circle_ordered

        col_order_list = circles_info[:,3].tolist() #vector with kmeans labels
        beg_ind = 0
        for col_ind_N in range(0,cluster_num):
            indices_col_N_list = [i for i, x in enumerate(col_order_list) if x == col_ind_N] #pick out circles_info rows corresponding to temporary kmeans label
            row_buff = circles_info[indices_col_N_list,1].tolist() #pick out "rows in pixel units" from circles_info corresponding to one kmeans label
            ind_sort_row = np.array(row_buff).argsort() #get "sorting index for ascending order" for each row, on format "which row first"            
            row_buff_ind = ind_sort_row.argsort().tolist() #get "sorting index for ascending order" for each row, on format "where should this row be"
            circles_info[indices_col_N_list,4] = row_buff_ind #add "sorting index for ascending order" on format "where should this row be" to circles_info
            info_local = circles_info[indices_col_N_list,:] #take out part of circles_info corresponding to kmeans label             
            info_local_sorted = info_local[ind_sort_row,:] #sort info_local_sorted in row order
            circles_info_sorted[beg_ind:beg_ind+len(indices_col_N_list),:] = info_local_sorted #sorted in lexicographical order: col, row
            beg_ind = beg_ind+len(indices_col_N_list)

        circles_info = circles_info_sorted
        circles_info[:,3] = [1+x for x in circles_info[:,3]] # start enumerations on 1
        circles_info[:,4] = [1+x for x in circles_info[:,4]] 
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        for c in range(0,cir_num):
            col_pixel = circles_info[c,0]
            row_pixel = circles_info[c,1]
            radiux = circles_info[c,2]
            col_index = circles_info[c,3]
            row_index = circles_info[c,4]
            cv2.circle(img,(col_pixel,row_pixel),radiux,(0,255,0),1) # draw the outer circle
            cv2.circle(img,(col_pixel,row_pixel),1,(0,0,255),1) # draw the center of the circle
            cv2.putText(img,str(col_index)+','+str(row_index),(col_pixel,row_pixel), font, 0.5,(0,255,0),1)

        cv2.imwrite(path_im_visual_detection+'/detected_circle_'+image_resized_filename,img)
    else:
        circles_info = []

    return circles_info

def tiff_image_read_out_spots(path_im, image_in_filename, path_im_spots, path_im_spots_rgb, path_im_visual_spots, circles_info, resize_factor, marginal_ext):
    img = Image.open(path_im+'/'+image_in_filename)
    im_heigh = img.tag[0x101][0]
    im_width =  img.tag[0x100][0]
    cir_num = circles_info.shape[0]
    circles_annotation_info_csv = np.zeros([cir_num,9], dtype="int") #### ['spot_center_column_position_in_pixel', 'spot_center_row_position_in_pixel', 'spot_radius_in_pixel', 'spot_column_index', 'spot_row_index', 'patch_image_top_left_in_column_in_pixel', 'patch_image_top_left_in_row_in_pixel', 'patch_image_bottom_right_in_column_in_pixel', 'patch_image_bottom_right_in_row_in_pixel']
    for c in range(0,cir_num):
        col_pixel = circles_info[c,0]
        row_pixel = circles_info[c,1]
        radiux = circles_info[c,2]
        col_index = circles_info[c,3]
        row_index = circles_info[c,4]
        ### calculate path location
        center_x = int(col_pixel/resize_factor)
        center_y = int(row_pixel/resize_factor)
        rad = int(radiux/resize_factor)
        rad_ext = int(rad +rad*marginal_ext)
        left = center_x - rad_ext
        right = center_x + rad_ext
        top = center_y - rad_ext
        bottom = center_y + rad_ext

        ### write annotation result
        circles_annotation_info_csv[c,0] = center_x
        circles_annotation_info_csv[c,1] = center_y
        circles_annotation_info_csv[c,2] = rad
        circles_annotation_info_csv[c,3] = col_index
        circles_annotation_info_csv[c,4] = row_index
        circles_annotation_info_csv[c,5] = left
        circles_annotation_info_csv[c,6] = top
        circles_annotation_info_csv[c,7] = right
        circles_annotation_info_csv[c,8] = bottom

        #-----------------------------------
        #write R,B,G separately 
        #-----------------------------------
        box = (left, top, right, bottom)
        img.seek(0)  ### red channel
        red_name = 'red_spot_'+'col_'+str(col_index)+'_'+'row_'+str(row_index)+'_'+image_in_filename
        img.crop(box).save(path_im_spots_rgb+'/'+red_name, bits=16)
        img.seek(1)  ### green channel
        gree_name = 'green_spot_'+'col_'+str(col_index)+'_'+'row_'+str(row_index)+'_'+image_in_filename
        img.crop(box).save(path_im_spots_rgb+'/'+gree_name, bits=16)
        img.seek(2)  ### blue channel
        blue_name = 'blue_spot_'+'col_'+str(col_index)+'_'+'row_'+str(row_index)+'_'+image_in_filename
        img.crop(box).save(path_im_spots_rgb+'/'+blue_name, bits=16)
        ### read and write with cv2, note that cv2 use BGR order
        img_red = cv2.imread(path_im_spots_rgb+'/'+red_name, -1)
        img_green = cv2.imread(path_im_spots_rgb+'/'+gree_name, -1)
        img_blue = cv2.imread(path_im_spots_rgb+'/'+blue_name, -1)

        img_rgb = np.zeros([rad_ext*2,rad_ext*2,3], dtype="uint16")
        img_rgb[:,:,0] = img_blue
        img_rgb[:,:,1] = img_green
        img_rgb[:,:,2] = img_red

        #-----------------------------------
        #write R,B,G in one file
        #-----------------------------------
        img_spot_name = 'spot_'+'col_'+str(col_index)+'_'+'row_'+str(row_index)+'_'+image_in_filename
        #cv2.imwrite(path_im_spots + img_spot_name, img_rgb) ### with compression LZW, could be used in future
        skimage.io.imsave(path_im_spots + '/'+img_spot_name, img_rgb, plugin='tifffile') ### uncompressed

        #-----------------------------------
        #write visual spot 
        #-----------------------------------
        ### write visual spot
        img_red_visual = cv2.imread(path_im_spots_rgb+'/'+red_name,0)
        spot_img_visual = cv2.equalizeHist(img_red_visual)
        cv2.imwrite(path_im_visual_spots+'/visual_spot_'+'col_'+str(col_index)+'_'+'row_'+str(row_index)+'_'+image_in_filename,spot_img_visual)
        #print 'write spot at ' +'column '+str(col_index)+', '+'row '+str(row_index) + ' into individual tiff file...'

    return circles_annotation_info_csv

def annotation_csv_write(path_annotation, circles_annotation_info_csv):
    annotation_info_csv = pd.DataFrame(circles_annotation_info_csv,columns=['spot_center_column_position_in_pixel', 'spot_center_row_position_in_pixel', 'spot_radius_in_pixel', 'spot_column_index', 'spot_row_index', 'patch_image_top_left_in_column_in_pixel', 'patch_image_top_left_in_row_in_pixel', 'patch_image_bottom_right_in_column_in_pixel', 'patch_image_bottom_right_in_row_in_pixel'])
    annotation_info_csv.to_csv(path_annotation, index=False)

#### wirte the annotation file in processing folder with spot only information

def write_spot_only_annotation(path_annotation_processing, circles_info, resize_factor, marginal_ext):
    cir_num = circles_info.shape[0]
    circles_annotation_info_csv = np.zeros([cir_num,9], dtype="int") #### ['spot_center_column_position_in_pixel', 'spot_center_row_position_in_pixel', 'spot_radius_in_pixel', 'spot_column_index', 'spot_row_index', 'patch_image_top_left_in_column_in_pixel', 'patch_image_top_left_in_row_in_pixel', 'patch_image_bottom_right_in_column_in_pixel', 'patch_image_bottom_right_in_row_in_pixel']
    
    for c in range(0,cir_num):
        col_pixel = circles_info[c,0]
        row_pixel = circles_info[c,1]
        radiux = circles_info[c,2]
        col_index = circles_info[c,3]
        row_index = circles_info[c,4]

        ### calculate path location
        center_x = int(col_pixel/resize_factor)
        center_y = int(row_pixel/resize_factor)
        rad = int(radiux/resize_factor)
        rad_ext = int(rad +rad*marginal_ext)
        left = center_x - rad_ext
        right = center_x + rad_ext
        top = center_y - rad_ext
        bottom = center_y + rad_ext

        ### write annotation result
        circles_annotation_info_csv[c,0] = center_x
        circles_annotation_info_csv[c,1] = center_y
        circles_annotation_info_csv[c,2] = rad
        circles_annotation_info_csv[c,3] = col_index
        circles_annotation_info_csv[c,4] = row_index
        circles_annotation_info_csv[c,5] = left
        circles_annotation_info_csv[c,6] = top
        circles_annotation_info_csv[c,7] = right
        circles_annotation_info_csv[c,8] = bottom

    annotation_info_csv = pd.DataFrame(circles_annotation_info_csv,columns=['spot_center_column_position_in_pixel', 'spot_center_row_position_in_pixel', 'spot_radius_in_pixel', 'spot_column_index', 'spot_row_index', 'patch_image_top_left_in_column_in_pixel', 'patch_image_top_left_in_row_in_pixel', 'patch_image_bottom_right_in_column_in_pixel', 'patch_image_bottom_right_in_row_in_pixel'])
    annotation_info_csv.to_csv(path_annotation_processing, index=False)


def annotation_file_count_spots(annotation_in_file):
    ############# read input annotation file
    df_in = pd.read_csv(annotation_in_file, encoding="ISO-8859-1")
    # drop the first and second columns
    df_in.drop(df_in.columns[[0,1]],axis=1,inplace=True)
    grid_row_num = df_in.shape[0] ## number of rows in grid, 32 in example
    grid_col_num = df_in.shape[1] ## number of cols in grid, 8 in example
    target_cir_num = 0

    for col in range(1,grid_col_num+1):
        col_ind = col
        spot_num_annotation_in_col_i =  len(df_in.loc[df_in[str(col_ind)] != 'BLANK'])    ### number of spots claimed in annotation input file
        target_cir_num = target_cir_num + spot_num_annotation_in_col_i  ### count the target number of spots
    
    return target_cir_num   


def annotation_file_mapping(annotation_in_file, annotation_out_file):
    ############# read input annotation file
    df_in = pd.read_csv(annotation_in_file, encoding="ISO-8859-1")
    # drop the first and second columns
    df_in.drop(df_in.columns[[0,1]],axis=1,inplace=True)
    grid_row_num = df_in.shape[0] ## number of rows in grid, 32 in example
    grid_col_num = df_in.shape[1] ## number of cols in grid, 8 in example

    ############# read output annotation file which produced by spot detection algorithm. The file only contains spots
    df_out = pd.read_csv(annotation_out_file, encoding="ISO-8859-1")
    ########## interpolation between spots for Blanks
    grid_info = np.zeros([grid_row_num,grid_col_num, 12], dtype=object) ### 9-dimension, same as annotation file output. 
    ### read out row information of non-BLANK area, mapping to input annotation format
    for col in range(1,grid_col_num+1):
        col_ind = col
        df_out_col_i = df_out.loc[df_out['spot_column_index'] == col_ind]  ### filter out data for column i
        spot_num_detect_col_i = len(df_out_col_i)    ### number of spots detected by algorithm
        spot_num_annotation_in_col_i =  len(df_in.loc[df_in[str(col_ind)] != 'BLANK'])    ### number of spots claimed in annotation input file
        spot_point = df_out_col_i.index.tolist()[0]  ### row index of first detected spot in column i
        #print 'col: ' + str(col_ind)
        #print 'spot_num_detect_col_i: ' + str(spot_num_detect_col_i)
        #print 'spot_num_annotation_in_col_i: ' + str(spot_num_annotation_in_col_i)
        ### output error message:
        if spot_num_detect_col_i != spot_num_annotation_in_col_i:
            print 'Geometry of input image and annotation file does not match. Please check the column ' + str(col_ind) + ' on image.' 

        ##### the number of detected will be equal to input annotated spots. This is the correct case and directly mapping two files

        for row in range(0,grid_row_num):
            if df_in[str(col_ind)][row] != 'BLANK':
                grid_info[row,col_ind-1,0] =  df_out_col_i['spot_center_column_position_in_pixel'][spot_point]
                grid_info[row,col_ind-1,1] =  df_out_col_i['spot_center_row_position_in_pixel'][spot_point]
                grid_info[row,col_ind-1,2] =  df_out_col_i['spot_radius_in_pixel'][spot_point]
                grid_info[row,col_ind-1,3] =  df_out_col_i['spot_column_index'][spot_point]
                grid_info[row,col_ind-1,4] =  df_out_col_i['spot_row_index'][spot_point]
                grid_info[row,col_ind-1,5] =  df_out_col_i['patch_image_top_left_in_column_in_pixel'][spot_point]
                grid_info[row,col_ind-1,6] =  df_out_col_i['patch_image_top_left_in_row_in_pixel'][spot_point]
                grid_info[row,col_ind-1,7] =  df_out_col_i['patch_image_bottom_right_in_column_in_pixel'][spot_point]
                grid_info[row,col_ind-1,8] =  df_out_col_i['patch_image_bottom_right_in_row_in_pixel'][spot_point]
                grid_info[row,col_ind-1,9] =  col_ind  ### content column index, include spots and blanks
                grid_info[row,col_ind-1,10] = row + 1  ### content row index, include spots and blanks
                grid_info[row,col_ind-1,11] =  df_in[str(col_ind)][row]
                spot_point = spot_point + 1

    ### interpolate the 'BLANK' part in 'spot_center_row_position_in_pixel' and 'spot_center_column_position_in_pixel'
    for col in range(1,grid_col_num+1):
        col_ind = col
        ### extract 'spot_center_row_position_in_pixel' information
        row_info_col_i = grid_info[:,col_ind-1,1]
        nonzero_ind_row = np.nonzero(row_info_col_i)[0]
        nonzero_value_row = row_info_col_i[nonzero_ind_row]
        interp_row = scipy.interpolate.splrep(nonzero_ind_row, nonzero_value_row, k=1, s=0) ### interpolation and extrapolation
        ### extract 'spot_center_column_position_in_pixel' information
        col_info_col_i = grid_info[:,col_ind-1,0]
        nonzero_ind_col = np.nonzero(col_info_col_i)[0]
        nonzero_value_col = col_info_col_i[nonzero_ind_col]
        interp_col = scipy.interpolate.splrep(nonzero_ind_col, nonzero_value_col, k=1, s=0) ### interpolation or extrapolation
        ### extract 'spot_radius_in_pixel' information, take max
        rad_info_col_i = grid_info[:,col_ind-1,2]
        nonzero_ind_rad = np.nonzero(rad_info_col_i)[0]
        nonzero_value_rad = rad_info_col_i[nonzero_ind_rad]
        rad_max = max(nonzero_value_rad)

        ### extract the average information
        for row in range(0,grid_row_num):
            if df_in[str(col_ind)][row] == 'BLANK':
                grid_info[row,col_ind-1,0] = int(scipy.interpolate.splev(row, interp_col))  ### estimated by interpolation or extrapolation
                grid_info[row,col_ind-1,1] = int(scipy.interpolate.splev(row, interp_row))  ### estimated by interpolation or extrapolation
                grid_info[row,col_ind-1,2] = int(rad_max)   ### estimated by max of detected spots
                grid_info[row,col_ind-1,3] =  0  ### BLANK has no index
                grid_info[row,col_ind-1,4] =  0  ### BLANK has no index
                grid_info[row,col_ind-1,5] = grid_info[row,col_ind-1,0] - grid_info[row,col_ind-1,2]
                grid_info[row,col_ind-1,6] = grid_info[row,col_ind-1,1] - grid_info[row,col_ind-1,2] 
                grid_info[row,col_ind-1,7] = grid_info[row,col_ind-1,0] + grid_info[row,col_ind-1,2] 
                grid_info[row,col_ind-1,8] = grid_info[row,col_ind-1,1] + grid_info[row,col_ind-1,2]
                grid_info[row,col_ind-1,9] =  col_ind  ### content column index, include spots and blanks
                grid_info[row,col_ind-1,10] = row + 1  ### content row index, include spots and blanks
                grid_info[row,col_ind-1,11] = 'BLANK'

    ##### convert the grid info matrix into [n x 9] matrix where n is 32x8, compatible to csv
    entry_num = grid_row_num*grid_col_num
    grid_info_csv = np.zeros([entry_num,12], dtype=object)
    entry_pointer = 0

    for col in range(1,grid_col_num+1):
        col_ind = col
        for row in range(0,grid_row_num):
            grid_info_csv[entry_pointer,0] = grid_info[row,col_ind-1,0]
            grid_info_csv[entry_pointer,1] = grid_info[row,col_ind-1,1]
            grid_info_csv[entry_pointer,2] = grid_info[row,col_ind-1,2]
            grid_info_csv[entry_pointer,3] = grid_info[row,col_ind-1,3]
            grid_info_csv[entry_pointer,4] = grid_info[row,col_ind-1,4]
            grid_info_csv[entry_pointer,5] = grid_info[row,col_ind-1,5]
            grid_info_csv[entry_pointer,6] = grid_info[row,col_ind-1,6]
            grid_info_csv[entry_pointer,7] = grid_info[row,col_ind-1,7]
            grid_info_csv[entry_pointer,8] = grid_info[row,col_ind-1,8]
            grid_info_csv[entry_pointer,9] = grid_info[row,col_ind-1,9]
            grid_info_csv[entry_pointer,10] = grid_info[row,col_ind-1,10]
            grid_info_csv[entry_pointer,11] = grid_info[row,col_ind-1,11]
            entry_pointer = entry_pointer + 1

    return grid_info_csv


def tiff_image_read_out_spots_and_blanks(path_im, image_in_filename, path_im_spots, path_im_spots_rgb, path_im_visual_spots, grid_info_csv):

    img = Image.open(path_im+'/'+image_in_filename)
    im_heigh = img.tag[0x101][0]
    im_width =  img.tag[0x100][0]
    cir_num = grid_info_csv.shape[0]

    for c in range(0,cir_num):
        ### read mapped annotation 
        center_x = grid_info_csv[c,0]
        center_y = grid_info_csv[c,1]
        rad = grid_info_csv[c,2]
        col_index = grid_info_csv[c,3]
        row_index = grid_info_csv[c,4]
        left = grid_info_csv[c,5]
        top = grid_info_csv[c,6]
        right = grid_info_csv[c,7]
        bottom = grid_info_csv[c,8]
        content_col_index = grid_info_csv[c,9]
        content_row_index = grid_info_csv[c,10]
        tag = grid_info_csv[c,11]

        #-----------------------------------
        #write R,B,G separately 
        #-----------------------------------
        box = (left, top, right, bottom)
        img.seek(0)  ### red channel
        red_name = 'red_'+'content_'+'col_'+str(content_col_index)+'_'+'row_'+str(content_row_index)+'_spot_'+'col_'+str(col_index)+'_'+'row_'+str(row_index)+'_'+image_in_filename
        img.crop(box).save(path_im_spots_rgb+'/'+red_name, bits=16)
        img.seek(1)  ### green channel
        gree_name = 'green_'+'content_'+'col_'+str(content_col_index)+'_'+'row_'+str(content_row_index)+'_spot_'+'col_'+str(col_index)+'_'+'row_'+str(row_index)+'_'+image_in_filename
        img.crop(box).save(path_im_spots_rgb+'/'+gree_name, bits=16)
        img.seek(2)  ### blue channel
        blue_name = 'blue_'+'content_'+'col_'+str(content_col_index)+'_'+'row_'+str(content_row_index)+'_spot_'+'col_'+str(col_index)+'_'+'row_'+str(row_index)+'_'+image_in_filename
        img.crop(box).save(path_im_spots_rgb+'/'+blue_name, bits=16)

        ### read and write with cv2, note that cv2 use BGR order
        img_red = cv2.imread(path_im_spots_rgb+'/'+red_name, -1)
        img_green = cv2.imread(path_im_spots_rgb+'/'+gree_name, -1)
        img_blue = cv2.imread(path_im_spots_rgb+'/'+blue_name, -1)

        rad_ext = img_red.shape[0]
        img_rgb = np.zeros([rad_ext,rad_ext,3], dtype="uint16")
        img_rgb[:,:,0] = img_blue
        img_rgb[:,:,1] = img_green
        img_rgb[:,:,2] = img_red

        #-----------------------------------
        #write R,B,G in one file
        #-----------------------------------
        image_in_filename_prefix = image_in_filename[0:image_in_filename.index('.')]
        image_in_filename_suffix = image_in_filename[image_in_filename.index('.'):len(image_in_filename)]
        img_spot_name = image_in_filename_prefix+'_'+tag+'_col_'+str(content_col_index)+'_row_'+str(content_row_index)+image_in_filename_suffix
        #cv2.imwrite(path_im_spots + img_spot_name, img_rgb) ### with compression LZW, could be used in future
        skimage.io.imsave(path_im_spots + '/'+img_spot_name, img_rgb, plugin='tifffile') ### uncompressed

        #-----------------------------------
        #write visual spot 
        #-----------------------------------
        img_red_visual = cv2.imread(path_im_spots_rgb+'/'+red_name,0)
        spot_img_visual = cv2.equalizeHist(img_red_visual)
        cv2.imwrite(path_im_visual_spots+'/visual_'+'content_'+'col_'+str(content_col_index)+'_'+'row_'+str(content_row_index)+'_spot_'+'col_'+str(col_index)+'_'+'row_'+str(row_index)+'_'+image_in_filename,spot_img_visual)
        #print 'write spot at ' +'column '+str(col_index)+', '+'row '+str(row_index) + ' into individual tiff file...'


def annotation_mapping_csv_write(path_annotation, grid_info_csv):
    annotation_info_csv = pd.DataFrame(grid_info_csv,columns=['spot_center_column_position_in_pixel', 'spot_center_row_position_in_pixel', 'spot_radius_in_pixel', 'spot_column_index', 'spot_row_index', 'patch_image_top_left_in_column_in_pixel', 'patch_image_top_left_in_row_in_pixel', 'patch_image_bottom_right_in_column_in_pixel', 'patch_image_bottom_right_in_row_in_pixel', 'content_column_index', 'content_row_index', 'tag'])
    annotation_info_csv.to_csv(path_annotation, index=False)

def visualize_contents(path_im_resize, image_resized_filename, path_im_out, im_with_all_content_filename, grid_info_csv, resize_factor):
    img = cv2.imread(path_im_resize+'/'+image_resized_filename,0)
    content_num = grid_info_csv.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for c in range(0,content_num):
        ### read mapped annotation 
        center_x = grid_info_csv[c,0]
        center_y = grid_info_csv[c,1]
        rad = grid_info_csv[c,2]
        col_index = grid_info_csv[c,3]
        row_index = grid_info_csv[c,4]
        left = grid_info_csv[c,5]
        top = grid_info_csv[c,6]
        right = grid_info_csv[c,7]
        bottom = grid_info_csv[c,8]
        content_col_index = grid_info_csv[c,9]
        content_row_index = grid_info_csv[c,10]
        tag = grid_info_csv[c,11]

        ### draw information
        col_pixel = int(center_x*resize_factor)
        row_pixel = int(center_y*resize_factor)
        radiux = int(rad*resize_factor)

        cv2.circle(img,(col_pixel,row_pixel),radiux,(0,255,0),1) # draw the outer circle
        cv2.circle(img,(col_pixel,row_pixel),1,(0,0,255),1) # draw the center of the circle
        cv2.putText(img,str(content_col_index)+','+str(content_row_index),(col_pixel,row_pixel), font, 0.5,(0,255,0),1)
        cv2.putText(img,tag,(col_pixel,row_pixel+15), font, 0.5,(0,255,0),1)
    

    cv2.imwrite(path_im_out+'/'+im_with_all_content_filename,img)

##====================================================
##Without server function
##====================================================

#

def detect_spots_on_local_computer(input_cat_name, dir_local, mem_switch):
    #
    ##====================================================
    ##Parameters
    ##==================================================== 
    print "processing call: "+str(input_cat_name)+"..." 

    dir_image_in=dir_local+'/uploads/'+str(input_cat_name)
    uploaded_files = [f for f in os.listdir(dir_image_in) if os.path.isfile(os.path.join(dir_image_in, f))]        
    nr_of_uploaded_files=len(uploaded_files)

    for file_idx in range(0,nr_of_uploaded_files):       
        if(uploaded_files[file_idx][-4:]==".csv"):
            annotation_in_filename=uploaded_files[file_idx]
        elif(uploaded_files[file_idx][-4:]==".tif" or uploaded_files[file_idx][-5:]==".tiff"):            
            image_in_filename=uploaded_files[file_idx]

    dir_annotation_in=dir_local+'/uploads/'+str(input_cat_name)+"/"+annotation_in_filename
    dir_im_tiles = dir_local+'/processing/'+str(input_cat_name)+'/tiles'
    dir_im_resize = dir_local+'/processing/'+str(input_cat_name)+'/resize'
    dir_im_visual_detection = dir_local+'/processing/'+str(input_cat_name)+'/visual_detection'
    dir_im_visual_spots = dir_local+'/processing/'+str(input_cat_name)+'/visual_spots'
    dir_im_spots_rgb = dir_local+'/processing/'+str(input_cat_name)+'/spots_rgb'
    dir_annotation_processing=dir_local+'/processing/'+str(input_cat_name)+'/annotation_processing'
    annotation_processing_file=dir_annotation_processing+'/'+annotation_in_filename
    dir_im_with_unintended_missed_spots=dir_local+'/downloads/'+str(input_cat_name)
    dir_im_spots = dir_local+'/downloads/'+str(input_cat_name)
    dir_annotation_out=dir_local+'/downloads/'+str(input_cat_name)
    annotation_out_file=dir_annotation_out+'/'+annotation_in_filename
    im_with_unintended_missed_spots_filename=image_in_filename

    #
    tile_size = 1000
    resize_factor = 0.1
    ##====================================================
    ##Algorithm
    ##====================================================     

    ##### make catalogs
    if not os.path.exists(dir_im_tiles):
        os.makedirs(dir_im_tiles)

    if not os.path.exists(dir_im_resize):
        os.makedirs(dir_im_resize)

    if not os.path.exists(dir_im_visual_detection):
        os.makedirs(dir_im_visual_detection)

    if not os.path.exists(dir_im_visual_spots):
        os.makedirs(dir_im_visual_spots)

    if not os.path.exists(dir_im_spots_rgb):
        os.makedirs(dir_im_spots_rgb)

    if not os.path.exists(dir_im_spots):
        os.makedirs(dir_im_spots)

    if not os.path.exists(dir_im_with_unintended_missed_spots):
        os.makedirs(dir_im_with_unintended_missed_spots)

    if not os.path.exists(dir_annotation_processing):
        os.makedirs(dir_annotation_processing)

    ##### FUNCTION 1-2: read in image as tiles, resize image
    tiff_image_preprocessing(dir_image_in, image_in_filename, dir_im_tiles, tile_size, dir_im_resize, resize_factor, mem_switch)
    
    ##### FUNCTION 3: circle detection on resized image
    image_resized_filename = 'resized_red_f10_'+image_in_filename
    target_cir_num = annotation_file_count_spots(dir_annotation_in)  #### number of circles in one image, the target number is reading from annotation input file

    stop_thr = 2          #### stop condition of iteration
    min_rad = 20          #### min rad of one circle
    max_rad = 35          #### max rad of one circle
    dp = 2                #### dp parameter of HoughCircles()
    min_dist = int((min_rad+max_rad)/2.0)  #### min_dist parameter of HoughCircles() and denoise filter
    min_para2 = 12        #### min para2 parameter of HoughCircles() 
    max_para2 = 80        #### max para2 parameter of HoughCircles() 
    number_of_column = 8

    ###### return a list of circles [col_pixel, row_pixel, rad, col_index, row_index]
    circles_info = circle_detection(dir_im_resize, image_resized_filename, dir_im_visual_detection, target_cir_num, stop_thr, min_rad, max_rad, dp, min_dist, min_para2, max_para2, number_of_column)

    ## detect unintentionally missing spots ##############################################       
    ##preliminaries----------------
    (spots_and_blanks_mat,full_cols_vec,nr_of_circle_cols,annotation_circles_per_col_vec)=spot_classification.parse_annotation_file(dir_annotation_in)
    (dist_vertical_median,dist_vertical_median_vec)=spot_classification.get_vertical_median_distance(full_cols_vec,circles_info)
    column_horizontal_positions=spot_classification.get_column_positions(circles_info,nr_of_circle_cols)       

    ##add unintentionally missing circles------
    (err_flag,circles_info)=spot_classification.add_unintentionally_missing_circles(nr_of_circle_cols,annotation_circles_per_col_vec,column_horizontal_positions,dist_vertical_median_vec,circles_info)       
    #print 'err_flag' + str(err_flag)

    ## detect excess spots ##############################################
    (err_flag,circles_info)=spot_classification.remove_extra_circles(nr_of_circle_cols,annotation_circles_per_col_vec,column_horizontal_positions,dist_vertical_median_vec,circles_info)
    
    #print 'err_flag' + str(err_flag)

    ####### FUNCTION 4: mapping the input and output annotation to make them consistent. The grid information inculded both spots and blank tiles
    marginal_ext = 0.4  ### extend each spot board
    write_spot_only_annotation(annotation_processing_file, circles_info, resize_factor, marginal_ext)
    grid_info_csv = annotation_file_mapping(dir_annotation_in, annotation_processing_file)

    ####### FUNCTION 5: write each spot and blank into an individual file
    tiff_image_read_out_spots_and_blanks(dir_image_in, image_in_filename, dir_im_spots, dir_im_spots_rgb, dir_im_visual_spots, grid_info_csv)
    ##save image for visualization---------------------
    visualize_contents(dir_im_resize, image_resized_filename, dir_im_with_unintended_missed_spots, im_with_unintended_missed_spots_filename, grid_info_csv, resize_factor)
    print "Visualization image saved!"

    ####### FUNCTION 6: write the annotation information into the CSV file, add timestamp
    annotation_mapping_csv_write(annotation_out_file, grid_info_csv)
    return_errmess = err_flag       

    print "processed call: "+str(input_cat_name) 


#====================================================
#Server function
#====================================================
class RPCfunctions(object):
    
    
    def __init__(self, local_dir, mem_switch):
        self.local_dir=local_dir
        self.mem_switch = mem_switch
    #END def __init__()
    
    
    
    
    
    def detect_spots(self, input_cat_name):
        #====================================================
        #Parameters
        #==================================================== 
        print "processing call: "+str(input_cat_name)+"..." 
        dir_local = self.local_dir 
        m_switch = self.mem_switch
               
        dir_image_in=dir_local+'/uploads/'+str(input_cat_name)

        uploaded_files = [f for f in os.listdir(dir_image_in) if os.path.isfile(os.path.join(dir_image_in, f))]        
        nr_of_uploaded_files=len(uploaded_files)
        
        for file_idx in range(0,nr_of_uploaded_files):       
            if(uploaded_files[file_idx][-4:]==".csv"):
                annotation_in_filename=uploaded_files[file_idx]
            elif(uploaded_files[file_idx][-4:]==".tif" or uploaded_files[file_idx][-5:]==".tiff"):            
                image_in_filename=uploaded_files[file_idx]


        dir_annotation_in=dir_local+'/uploads/'+str(input_cat_name)+"/"+annotation_in_filename
        dir_im_tiles = dir_local+'/processing/'+str(input_cat_name)+'/tiles'
        dir_im_resize = dir_local+'/processing/'+str(input_cat_name)+'/resize'
        dir_im_visual_detection = dir_local+'/processing/'+str(input_cat_name)+'/visual_detection'
        dir_im_visual_spots = dir_local+'/processing/'+str(input_cat_name)+'/visual_spots'
        dir_im_spots_rgb = dir_local+'/processing/'+str(input_cat_name)+'/spots_rgb'
        dir_annotation_processing=dir_local+'/processing/'+str(input_cat_name)+'/annotation_processing'
        annotation_processing_file=dir_annotation_processing+'/'+annotation_in_filename
        dir_im_with_unintended_missed_spots=dir_local+'/downloads/'+str(input_cat_name)
        dir_im_spots = dir_local+'/downloads/'+str(input_cat_name)
        dir_annotation_out=dir_local+'/downloads/'+str(input_cat_name)
        annotation_out_file=dir_annotation_out+'/'+annotation_in_filename
        im_with_unintended_missed_spots_filename=image_in_filename

        tile_size = 1000
        resize_factor = 0.1

        #====================================================
        #Algorithm
        #====================================================     

        #### make catalogs
        if not os.path.exists(dir_im_tiles):
            os.makedirs(dir_im_tiles)

        if not os.path.exists(dir_im_resize):
            os.makedirs(dir_im_resize)

        if not os.path.exists(dir_im_visual_detection):
            os.makedirs(dir_im_visual_detection)

        if not os.path.exists(dir_im_visual_spots):
            os.makedirs(dir_im_visual_spots)

        if not os.path.exists(dir_im_spots_rgb):
            os.makedirs(dir_im_spots_rgb)

        if not os.path.exists(dir_im_spots):
            os.makedirs(dir_im_spots)

        if not os.path.exists(dir_im_with_unintended_missed_spots):
            os.makedirs(dir_im_with_unintended_missed_spots)

        if not os.path.exists(dir_annotation_processing):
            os.makedirs(dir_annotation_processing)


        ##### FUNCTION 1-2: read in image as tiles, resize image
        tiff_image_preprocessing(dir_image_in, image_in_filename, dir_im_tiles, tile_size, dir_im_resize, resize_factor, m_switch)
    
        #### FUNCTION 3: circle detection on resized image
        image_resized_filename = 'resized_red_f10_'+image_in_filename
        target_cir_num = annotation_file_count_spots(dir_annotation_in)  #### number of circles in one image, the target number is reading from annotation input file
        stop_thr = 2          #### stop condition of iteration
        min_rad = 20          #### min rad of one circle
        max_rad = 35          #### max rad of one circle
        dp = 2                #### dp parameter of HoughCircles()
        min_dist = int((min_rad+max_rad)/2.0)  #### min_dist parameter of HoughCircles() and denoise filter
        min_para2 = 12        #### min para2 parameter of HoughCircles() 
        max_para2 = 80        #### max para2 parameter of HoughCircles() 
        number_of_column = 8

        ##### return a list of circles [col_pixel, row_pixel, rad, col_index, row_index]
        circles_info = circle_detection(dir_im_resize, image_resized_filename, dir_im_visual_detection, target_cir_num, stop_thr, min_rad, max_rad, dp, min_dist, min_para2, max_para2, number_of_column)

        # detect unintentionally missing spots ##############################################       
        #preliminaries----------------
        (spots_and_blanks_mat,full_cols_vec,nr_of_circle_cols,annotation_circles_per_col_vec)=spot_classification.parse_annotation_file(dir_annotation_in)
        (dist_vertical_median,dist_vertical_median_vec)=spot_classification.get_vertical_median_distance(full_cols_vec,circles_info)
        column_horizontal_positions=spot_classification.get_column_positions(circles_info,nr_of_circle_cols)       

        #add unintentionally missing circles------
        (err_flag,circles_info)=spot_classification.add_unintentionally_missing_circles(nr_of_circle_cols,annotation_circles_per_col_vec,column_horizontal_positions,dist_vertical_median_vec,circles_info)       

        # detect excess spots ##############################################
        (err_flag,circles_info)=spot_classification.remove_extra_circles(nr_of_circle_cols,annotation_circles_per_col_vec,column_horizontal_positions,dist_vertical_median_vec,circles_info)

        ####### FUNCTION 4: mapping the input and output annotation to make them consistent. The grid information inculded both spots and blank tiles
        marginal_ext = 0.4  ### extend each spot board
        write_spot_only_annotation(annotation_processing_file, circles_info, resize_factor, marginal_ext)
        grid_info_csv = annotation_file_mapping(dir_annotation_in, annotation_processing_file)

        ####### FUNCTION 5: write each spot and blank into an individual file
        tiff_image_read_out_spots_and_blanks(dir_image_in, image_in_filename, dir_im_spots, dir_im_spots_rgb, dir_im_visual_spots, grid_info_csv)
        visualize_contents(dir_im_resize, image_resized_filename, dir_im_with_unintended_missed_spots, im_with_unintended_missed_spots_filename, grid_info_csv, resize_factor)
        print "Visualization image saved!"

        ####### FUNCTION 6: write the annotation information into the CSV file, add timestamp
        annotation_mapping_csv_write(annotation_out_file, grid_info_csv)
        return_errmess = err_flag       

        print "processed call: "+str(input_cat_name) 
        return return_errmess

################################ user input with arguments
### Example: python persomics_image_analysis_with_argument.py 0 1 asdf ### memory free, offline
### Example: python persomics_image_analysis_with_argument.py 0 0 asdf ### memory free, online


user_input_args = sys.argv[1:]  #### [1]: memory switch [2]: state: 0- online, 1-offline [3]: input_folder 

#### comments on memory: if the image size (0.x GB- 4 GB) is further smaller than the RAM size, recommended to use the memory free version. If the image size is large, use the memory constrained version. 
mem_switch = int(user_input_args[0])   ### memory 1: memory constrained, 0: memory free # 
online_offline_state = int(user_input_args[1])

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
    tcp_adress="tcp://127.0.0.1:4242"
    local_dir='.'
    server_obj=RPCfunctions(local_dir, mem_switch)
    s = zerorpc.Server(server_obj)
    s.bind(tcp_adress)
    print "Server is up..."
    s.run()
elif online_offline_state == 1:
    #====================================================
    #Run spot detection without server
    #====================================================
    local_dir='.'
    input_folder = user_input_args[2] 
    
    detect_spots_on_local_computer(input_folder, local_dir, mem_switch)























































































