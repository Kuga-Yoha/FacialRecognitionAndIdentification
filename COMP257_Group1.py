# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:37:36 2023

@author: Kugavathanan  Yohanathan
         Sayanthana  Yoganathan
         
         COMP-257
"""



import scipy.io as io
import os


#Get the current file location
current_file_location = os.path.abspath(__file__)


print("Current file location:", current_file_location)


mat_file_path = os.path.join(os.path.dirname(current_file_location), 'umist_cropped.mat')

#Load the .mat file
mat_data = io.loadmat(mat_file_path)


facedat_data = mat_data['facedat']

print(type(facedat_data))

print(facedat_data)
#(1, 20)



print(facedat_data[0][19].size)


dataset = []

for i in range(facedat_data[0].size):
    for j  in range (facedat_data[0][i].size):
        dataset.append(facedat_data[0][i])
    
print(len(dataset))   


print(dataset[0].dtypes.names) 


print(facedat_data[0][])
    
    
    
