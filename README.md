## pyShore

pyShore is an open source toolkit for mapping shoreline structures with a U-Net framework pre-trained on high-resolution orthoimagery. The following flowchart outlines the use of this tool in ArcGIS Pro.

![alt text](images/ArcGIS_flow.jpg)

### How to add the toolkit to ArcGIS Pro
- Download the pyShore github repository.
- In ArcGIS Pro, turn on Catalog Pane under the View tab (section A. in below chart).
- Go to Catalog, and add pyShore toolbox by clicking the "Add Toolbox" (section B.). 
- The "CoastalStructurePrediction" toolkit is added to your environment and ready to use (section C and D.).

![alt text](images/add_pyShore.png)

### How to run the toolbox
There are four inputs required to run the toolkit: 

**ProjectDir** - a working folder in which to save all processing and result data.

**ImageDir** - a user defined folder with all source imagery in a georeferenced (geoTiff) format. 

**GeoPath** - a user defined a single geometry file (i.e., shapefile) that defines the geographic locations of shoreline within the provided imagery.

**BufferDist** - a user defined distance to buffer shorelines, the distance from the shoreline that should be extracted from the imagery to serve as the target for the classification.

## Script description
### pyShoreArcGIS
- ImageCropping.py 
- ImagePrediction.py
- ImagePostprocessing.py


### Requirements for model training
- torch: 1.12.0
- CUDA: 11.2
- python: 3.7.10

### Requirements for ArcGIS prediction
ArcGIS Pro 2.3.0

Your ArcGIS Pro environment should have the deep learning framework set up. To set up your environment, please refer: https://pro.arcgis.com/en/pro-app/2.6/tool-reference/image-analyst/pdf/deep_learning_install.pdf

You will also need geopandas and rasterio library. For some open-source python libraries, they are not automatically installed in ArcGIS Pro. To install geopandas, activate the virtual environment you created from above tutorial, and run conda install geopandas libtiff=4.0.10. The rasterio can be installed through conda or pip.

