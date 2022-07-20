# install 

* `git clone https://wimthiels@bitbucket.org/pgmsembryogenesis/embryo_counter.git`
* cd into embryo_counter
* `pip install -r requirements.txt`

# one-time preparatory action
change the root folder to your local drive in 
* embryo_counter_run.py
* cellprofiler.py
* ./CellProfilerAnalyst/propertiesFile.txt

# run embryo_counter
*  collect input images
    * put all the input tiff files into the INPUT folder.  
    * Preferably use the '#2' files : they are not too large (+- 50MB)
    * The XY-resolution is set at 0.454 micron. You can change this by setting prm['res_XY']  in embryo_counter_run
    
    
*  `python embryo_counter_run`

*  check OUTPUT folder for result (report.tif, regions.csv).  The total cell count is the number of rows with column lbl_pred == 'P'

# training via CellProfilerAnalyst
   
## installation and manual
* https://cellprofileranalyst.org/releases
* https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-Analyst-3.0.2/index.html


## compose image.csv and objects.csv
`python cellprofiler.py`

## run
Launch CellProfilerAnalyst and load the properties file (./CellProfilerAnalyst/propertiesFile.txt).
From then on see instruction in the manual

