# pyShore

This post is used to show how to use pyShore in ArcGIS.

[comment]: <> (Data preprocessing before Deep Learning:)

[comment]: <> (- Need to have a list of images with annotations indicating the bounding box of the class in the image and the associated class name)

[comment]: <> (- Digitize all the animals in the given .tiff file in the raw data folder, and save it as a geojson or shapefile format)

[comment]: <> (Working on Models:)

[comment]: <> (Step1. Data processing and creating the training and validation dataset)

[comment]: <> (- Create a folder under dataset and name it "processing_data")

[comment]: <> (- In the above created folder, create two folders and name them "clipped" and "annotation" respectively.)

[comment]: <> (- Run the get_data.py in the script to generate the training dataset and testing dataset, and annotations for each image.)

[comment]: <> (- The results will be saved in the folder created above.)

[comment]: <> (Step2. Get pre-trained vgg16 weights, other weights can be found on the same repo, based on the cnn architecture chosen to use.)

[comment]: <> (- https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5)

[comment]: <> (Step3. Running on local computer.)

[comment]: <> (The train_fcnn.py is used to train the model based on the customed dataset.Open a terminal, and training the model with: )

[comment]: <> (- python3 train_frcnn.py -o simple -p directory/to/in/processing_data/train_annotation.txt --num_epochs 1000 --input_weight_path directory/to/the/weights/downloaded/above/vgg16_weights_tf_dim_ordering_tf_kernels.h5 --output_weight_path model_frcnn_vgg.hdf5 --hf True --vf True --rot True --network vgg)

[comment]: <> (- Check on the train_frcnn.py to get detailed calling requirement.)

[comment]: <> (Step 4. Running on sciclone or use GPU to boost the training.)

[comment]: <> (- modify the jobscript file in the script folder, copy above function call in step3 to this file.)

[comment]: <> (- set the gpu call on the top of the file, if gpu is not available, CPUs will be used to run the training.)

[comment]: <> (- submit a jobscript to sciclone by using: qsub jobscript)

[comment]: <> (Step 5. Running on cdsw using GPUs)

[comment]: <> (- Open a terminal, and copy the function call in step 3 to run training on cdsw.)

[comment]: <> (Step 6. Model validation)

[comment]: <> (- Running on testing and validation )

[comment]: <> (- python3 test_frcnn.py -o simple -p directory/to/in/processing_data/test_annotation.txt --network vgg)

[comment]: <> (- Producing 1&#41;. bounding_box_coordinates.csv that hosts the new predicted bounding box of testing data; 2&#41;. test_mAPs.csv file that saves the mAP value for each testing image.)


## Script description
### pyShoreArcGIS
- DataProcessing.py includes a series of support functions to preprocess the raw image and generate the image patches and annotations for training and validation.
- get_data.py is a call script used to generate the image patches and annotations. The dimension of image patches can be assigned in here.
- data_aggregation.py include script to split the raw image into image patches, these image patches will be used later on as input for model prediction.
- jobscript: a script to submit jobs on sciclone

### Model training with processed NAIP images
The faster_rcnn folder includes all model related scripts, they are called on training and validation process

## Requirements for model training
- torch: 1.12.0
- CUDA: 11.2
- python: 3.7.10

## Requirements for ArcGIS prediction
ArcGIS Pro 2.3.0


