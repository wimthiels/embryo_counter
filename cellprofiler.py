#!/usr/bin/env python
# coding: utf-8

# In[9]:


from pathlib import Path
import os, re, sys
import pandas as pd
p_root_folder = Path("/home/wth/Downloads/testinfra/ISOLATED/labkit_Roberto/")


# # set up the image file

# In[61]:


l_ImageNumber = []
l_ImageFileName = []
l_ImagePathName = []
l_plate = []
d_lookup = {}
p_input_img = p_root_folder / "INPUT" 
p_out = p_root_folder / "CellProfilerAnalyst"

for ix_file, file_i in enumerate(p_input_img.glob('*.tif')):
	image_number = ix_file + 1
	l_ImageNumber.append(image_number)
	l_ImageFileName.append(file_i.name)
	l_ImagePathName.append(file_i.parent)
	plate = file_i.name[0:2]
	l_plate.append(plate)
	d_lookup[plate] =  image_number
	
df_image = pd.DataFrame({'ImageNumber' : l_ImageNumber, 'ImageFileName' : l_ImageFileName , 'ImagePathName' : l_ImagePathName, 'plate': l_plate, 'well' :1})
df_image.to_csv(p_out / "image.csv")


# # Set up the objects file

# In[62]:


p_input_object = p_root_folder / "OUTPUT"

l_df_regions = []

for file_i  in  p_input_objects.glob("*/regions.csv"):
	df_region = pd.read_csv(file_i)
	input_file_link = file_i.parts[8][0:2]
	df_region['plate'] = input_file_link
	df_region['well'] = 1
	df_region['ImageNumber'] = d_lookup[input_file_link]
	l_df_regions.append(df_region)
df_regions = pd.concat(l_df_regions)
df_regions.columns = [i.replace("-","").replace("_", "") for i in df_regions.columns]
df_regions.to_csv(p_out / "object.csv")
df_regions


# In[ ]:




