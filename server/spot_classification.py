# -*- coding: utf-8 -*-

"""

Created on Wed Jun  8 14:24:04 2016



@author: daniel

"""



#==================================================

#import

#==================================================

import numpy as np

import csv

import cv2

import pandas as pd


#==================================================

#notes

#==================================================

#while the circles_info matrix starts labelling rows and cols

#with 1,2,3,...,

#the functions in this file labels rows and cols with 0,1,2,...



#==================================================

#fcns

#==================================================





def parse_annotation_file(annotation_file):

    #Output:

    #spots_and_blanks_mat : 1 for every spot that is not blank, otherwise 0.   

    #full_cols_vec : vector with 1 for each column that does not contain blanks.

    #nr_of_cols : nr of spot columns.

    #annotation_spots_per_col_vec : vector with the number of dots in each column according to the annotation file.



    #init annotation file reading--------------------------

    file_handler = open(annotation_file, 'rb')

    reader = csv.reader(file_handler, delimiter=',')  
    
    #### Haopeng added 20160722
    df_in = pd.read_csv(annotation_file, encoding="ISO-8859-1")
    # drop the first and second columns
    df_in.drop(df_in.columns[[0,1]],axis=1,inplace=True)
    grid_row_num = df_in.shape[0] ## number of rows in grid, 32 in example
    grid_col_num = df_in.shape[1] ## number of cols in grid, 8 in example
    ###########

    nr_of_rows = grid_row_num

    row=reader.next()

    nr_of_cols= np.int(row[-1])   

    #print nr_of_cols
     

    #get spots_and_blanks_mat---------------------------------

    spots_and_blanks_mat=np.zeros((nr_of_rows,nr_of_cols)) 

    for row_nr in range(0,nr_of_rows):

        row=reader.next()

        for col_nr in range(2,nr_of_cols+2):

            if(row[col_nr]!="BLANK"):

                spots_and_blanks_mat[row_nr,col_nr-2]=1

            #END if(row[col_nr]!="BLANK")

        #END for row_nr in range(0:nr_of_cols)            

    #END for row_nr in range(0:nr_of_rows)      

        

    #get full_cols_vec---------------------------------

    full_cols_vec=np.array(np.zeros((nr_of_cols,1))) #1 for each column that does not contain blanks, otherwise 0.        

    annotation_circles_per_col_vec=np.sum(spots_and_blanks_mat,0)        

    full_cols_vec=np.array(np.zeros((nr_of_cols,1)))    

    for vec_idx in range(0,nr_of_cols):

        if(annotation_circles_per_col_vec[vec_idx]==32):

            full_cols_vec[vec_idx]=1

        else:

            full_cols_vec[vec_idx]=0

        #END    

    #END for vec_idx in range(0,nr_of_cols)

            

    return(spots_and_blanks_mat,full_cols_vec,nr_of_cols,annotation_circles_per_col_vec)           

   

#END def parse_annotation_file

   

   

   







   

   

#--------------------------------------------------------

   

   

   

def get_column_positions(circles_info,nr_of_circle_cols):

    #Output: vector with median horizontal column positions 

 

   [nr_of_circles,dummy]=circles_info.shape 

   column_median_position_vec=[]      

      

   for circle_col_idx in range(0,nr_of_circle_cols):

       column_gather_array=[]  

       for circle_idx in range(0,nr_of_circles):

           if(circles_info[circle_idx,3]-1==circle_col_idx):                     

               column_gather_array.append(circles_info[circle_idx,0])                                

           #END if          

       #END for circle_idx in range(0,nr_of_circles) 

       column_median_position_vec.append(np.median(column_gather_array))     

   #END for circle_col_idx in range(0,nr_of_circle_cols)

   

   return column_median_position_vec

   

#END def get_vertical_median_distance()

 

 

 

#--------------------------------------------------------







def check_if_whole_column_detected(col_nr_to_investigate,annotation_circles_per_col_vec,detected_circles_info):

    #This function checks if column col_nr_to_investigate has as many detected spots as in the annotation file

    #Output: nr of detected dots - nr of annotation file dots, for the column col_nr_to_investigate     

    

    [nr_of_circles,dummy]=detected_circles_info.shape 

    detected_circles_count=0   

    for circle_idx in range(0,nr_of_circles):     

        if(detected_circles_info[circle_idx,3]-1==col_nr_to_investigate):        

            detected_circles_count=detected_circles_count+1       

        #END if                 

    #END for circle_idx in range(0,nr_of_circles)

    missing_spot_flag=np.int(detected_circles_count-annotation_circles_per_col_vec[col_nr_to_investigate])



    return missing_spot_flag



#END def check_if_whole_column_detected

    

  

  

#--------------------------------------------------------

  

  

  

   

def add_unintentionally_missing_circles(nr_of_circle_cols,annotation_circles_per_col_vec,column_horizontal_positions,dist_vertical_median_vec,detected_circles_info):

    #This function adds missing circles that are either unintendedly missed by the dotting machine, or

    #too weak to have been detected by the previous algorithm. The most important purpose of the function 

    #is to avoid error propagation in the spot classification.   

    #

    #Output: err_flag=1 if the nr of spots in the annotation file, and

    #the nr of of detected spots, do not match; and a new spot cannot be found.



    err_flag=0

    for circle_col_idx in range(0,nr_of_circle_cols):

        missing_spot_flag=check_if_whole_column_detected(circle_col_idx,annotation_circles_per_col_vec,detected_circles_info)

        

        while(missing_spot_flag<0 and err_flag==0):

            (err_flag,detected_circles_info)=add_new_circle(circle_col_idx,detected_circles_info,column_horizontal_positions,dist_vertical_median_vec[circle_col_idx])

            missing_spot_flag=check_if_whole_column_detected(circle_col_idx,annotation_circles_per_col_vec,detected_circles_info)

        #END 

    #END for circle_col_idx in range(0,nr_of_circle_cols)        

    

    return (err_flag,detected_circles_info)



#END add_unintentionally_missing_circles









#--------------------------------------------------------

def visualize_detected_circles(path_im_resize, image_name_resized, circles_info, path_im_out,im_with_unintended_missed_spots_filename):

    #Output: saves an image with detected circles in circles_info    

    

    img = cv2.imread(path_im_resize+'/'+image_name_resized,0)

    [cir_num,dummy]=circles_info.shape    

    font = cv2.FONT_HERSHEY_SIMPLEX

    for c in range(0,cir_num):

        col_pixel = circles_info[c,0]

        row_pixel = circles_info[c,1]

        radiux = circles_info[c,2]

        col_index = circles_info[c,3]

        row_index = circles_info[c,4]

        # draw the outer circle

        cv2.circle(img,(col_pixel,row_pixel),radiux,(0,255,0),1)

        # draw the center of the circle

        cv2.circle(img,(col_pixel,row_pixel),1,(0,0,255),1)

        # put text        

        cv2.putText(img,str(col_index)+','+str(row_index),(col_pixel,row_pixel), font, 0.5,(0,255,0),1)    

    #END for c in range(0,cir_num)

    cv2.imwrite(path_im_out+'/'+im_with_unintended_missed_spots_filename,img)



#END def add_new_circle











#--------------------------------------------------------











    

def add_new_circle(circle_col_to_expand,detected_circles_info,column_horizontal_positions,dist_vertical_median):

    #This function adds 1 new circle to detected_circles_info

    #Output: detected_circles_info and error_flag=0/1 if the function can/cannot find a new circle.

    #

    #For now, the function only aims at detecting extra dots inbetween already detected dots. This

    #algorithm can be expanded by adding more search cases, and returning errors more seldomly. 



    #preliminaries------------------------

    err_flag=0

    [nr_of_circles,dummy]=detected_circles_info.shape    

    dist_max=0

    

    #get the maximum vertical distance between two consecutive column spots--------------------------

    for circle_idx in range(0,nr_of_circles-1):      

        if(detected_circles_info[circle_idx,3]-1==circle_col_to_expand and detected_circles_info[circle_idx,4]-1>0):           

           dist_temp=(detected_circles_info[circle_idx,1]-detected_circles_info[circle_idx-1,1])

           if(dist_temp>dist_max):

               dist_max=dist_temp

               insert_idx=circle_idx

           #END if                

        #END if                   

    #END for circle_idx in range(0,nr_of_circles)           

                

    #add a circle if the maximum distance is bigger than 1.5*dist_vertical_median. Otherwise return an error.------------------------             

    if(dist_max>1.5*dist_vertical_median): 

       

        new_circle_col_pos=column_horizontal_positions[circle_col_to_expand]

        #new_circle_row_pos=(detected_circles_info[insert_idx,1]+detected_circles_info[insert_idx-1,1])/2.0 #interpolation             

        new_circle_row_pos=detected_circles_info[insert_idx-1,1]+dist_vertical_median #add at the side to accomodate more dots           

        new_circle_radius=np.max(detected_circles_info[:,2])

        new_circle_col_label=detected_circles_info[insert_idx,3]

        new_circle_row_label=detected_circles_info[insert_idx,4]

        new_row=[new_circle_col_pos, new_circle_row_pos, new_circle_radius, new_circle_col_label, new_circle_row_label]                        

        detected_circles_info_list=detected_circles_info.tolist()

        detected_circles_info_list.insert( insert_idx, new_row)

        detected_circles_info=np.asarray(detected_circles_info_list,dtype=int)                                   

        for circle_idx_relabel in range(insert_idx+1,nr_of_circles-1):

              if(detected_circles_info[circle_idx_relabel,3]-1==circle_col_to_expand):

                detected_circles_info[circle_idx_relabel,4]=detected_circles_info[circle_idx_relabel,4]+1

            #END 

        #END for circle_idx_relabel in range(insert_idx,nr_of_circles-1)

      

    else:

        err_flag=1        

    #END if(dist_max>1.5*dist_vertical_median)

  

    return (err_flag,detected_circles_info)  

  

#END def add_new_circle()

    

 

 

 

#----------------------------------------------------------------------







   

def remove_extra_circles(nr_of_circle_cols,annotation_circles_per_col_vec,column_horizontal_positions,dist_vertical_median_vec,detected_circles_info):

    #This function removes extra circles in each column. Columns with dots excess are identified by comparing to 

    #the annotation file.     



    err_flag=0

    for circle_col_idx in range(0,nr_of_circle_cols):

           

        missing_spot_flag=check_if_whole_column_detected(circle_col_idx,annotation_circles_per_col_vec,detected_circles_info)    

        while(missing_spot_flag>0 and err_flag==0):

            (err_flag,detected_circles_info)=remove_circle(circle_col_idx,detected_circles_info,column_horizontal_positions,dist_vertical_median_vec[circle_col_idx],annotation_circles_per_col_vec)

            missing_spot_flag=check_if_whole_column_detected(circle_col_idx,annotation_circles_per_col_vec,detected_circles_info)

        #END

            

    #END for circle_col_idx in range(0,nr_of_circle_cols)           

    return (err_flag,detected_circles_info)



#END add_unintentionally_missing_circles



 

 

#----------------------------------------------------------------------







  

def remove_circle(circle_col_to_compress,detected_circles_info,column_horizontal_positions,dist_vertical_median,annotation_circles_per_col_vec):

    #This function removes 1 circle from detected_circles_info.

    #Output: Detected_circles_info and error_flag=0/1 if the function can/cannot find a circle to remove.

    #        Currently, error_flag=0 always.

 



    #preliminaries------------------------

    err_flag=0

    [nr_of_circles,dummy]=detected_circles_info.shape    

    dist_vec=[]
    

    #get the minimum vertical distance between two consecutive column spots--------------------------

    for circle_idx in range(0,nr_of_circles-1):      

        if(detected_circles_info[circle_idx,3]-1==circle_col_to_compress and detected_circles_info[circle_idx,4]-1>0):           

           dist_temp=(detected_circles_info[circle_idx,1]-detected_circles_info[circle_idx-1,1])

           dist_vec.append(dist_temp)    

        #END if                   

    #END for circle_idx in range(0,nr_of_circles)                           

    min_idx=np.argmin(dist_vec)         

    dist_vec_length=len(dist_vec)

    

    #----------------------------------------------------------------------------------------------  

    #The closest circle pair is identified. Thereafter, the one circle of the pair, which has its other neighbor circle at

    #the closest distance, is removed. If the pair is on one edge of the column, 

    #the circle closest to the edge is removed.        

    if(min_idx==0):            

        remove_idx=0

    elif(min_idx==dist_vec_length-1):

        remove_idx=annotation_circles_per_col_vec[circle_col_to_compress]-1

    else:

        dist_before=dist_vec[min_idx-1]

        dist_after=dist_vec[min_idx+1]

        if(dist_before<=dist_after):

            remove_idx=min_idx

        else:    

            remove_idx=min_idx+1

        #END if(dist_before<=dist_after)                 

    #END if(min_idx==0)

    remove_idx=np.int(remove_idx+np.sum(annotation_circles_per_col_vec[0:circle_col_to_compress]))

  

    #remove a circle and relabel rows accordingly---------------------------------

    detected_circles_info_list=detected_circles_info.tolist()

    del detected_circles_info_list[remove_idx]

    detected_circles_info=np.asarray(detected_circles_info_list,dtype=int)                                   

    for circle_idx_relabel in range(remove_idx,nr_of_circles-1):

          if(detected_circles_info[circle_idx_relabel,3]-1==circle_col_to_compress):

            detected_circles_info[circle_idx_relabel,4]=detected_circles_info[circle_idx_relabel,4]-1

        #END 

    #END for circle_idx_relabel in range(insert_idx,nr_of_circles-1)





    #currently, err_flag=0 always----------------------      

    #else:

    #    err_flag=1        

    ##END if(dist_max>1.5*dist_vertical_median)



  

    return (err_flag,detected_circles_info)  

  

#END def add_new_circle()

    

 

 



#-------------------------------------------------------------------------

   

   

   

   

def get_vertical_median_distance(full_cols_vec,circles_info):

   #Output: dist_median:            spot median vertical distance for columns without blanks

   #        vertical_median_vec:    median distance for each column

 

   #preliminaries----------------------------------

   [nr_of_circles,dummy]=circles_info.shape 

   dist_array=[]    

   nr_of_circle_cols=len(full_cols_vec)

   vertical_median_vec=[]



   #spot median vertical distance for columns without blanks-----------------------------

   for circle_idx in range(0,nr_of_circles): #calculate median vertical distance

       if((full_cols_vec[circles_info[circle_idx,3]-1]==1) and (circles_info[circle_idx,4]<32)):                     

            dist_array.append(circles_info[circle_idx+1,1]-circles_info[circle_idx,1])                                

       #END if    

   #END for circle_idx in range(0,nr_of_circles)

   dist_median=np.median(dist_array)   



   #median distance for each column------------------------------------

   for circle_col_idx in range(0,nr_of_circle_cols): 

       dist_array=[]

       for circle_idx in range(0,nr_of_circles-1): 

           

           if(circles_info[circle_idx,3]-1==circle_col_idx and circles_info[circle_idx+1,3]-1==circle_col_idx):

               dist_array.append(circles_info[circle_idx+1,1]-circles_info[circle_idx,1])

           #END

        

       #END for circle_idx in range(0,nr_of_circles)

       vertical_median_vec.append(np.median(dist_array))

        

   #END for circle_col_idx in range(0,nr_of_circle_cols)

   

   return (dist_median,vertical_median_vec)

   

#END def get_vertical_median_distance()

   

   





