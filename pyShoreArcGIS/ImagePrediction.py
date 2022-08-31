
import arcpy
import numpy as np
import os
from pathlib import Path
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
import rasterio
import torch

def logits2conf(img):

    Xmax = np.amax(img)
    Xmin = np.amin(img)

    newX = (img - Xmin) / (Xmax - Xmin)

    return newX

# image_path = arcpy.GetParameterAsText(0)
# threshold = float(arcpy.GetParameterAsText(1))

image_path = r"C:\\Users\mlv\Documents\projects\ShorelineArmoring_ArcGIS"
# threshold = 0.5

model_path = os.path.join(image_path, "weights","model.pth")
output_path = os.path.join(image_path, "outputmask")
Path(output_path).mkdir(exist_ok=True, parents=True)

allfiles = [file for file in os.listdir(os.path.join(image_path, "crop_temp")) if file.endswith(".tif")]
arcpy.AddMessage("Processing: {}".format(allfiles))

model = smp.Unet('resnet18',
                encoder_weights="imagenet",
                in_channels=4,
                classes=4)

checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

transform_test = transforms.Compose([transforms.ToTensor()])

impute_nan = np.tile([0, 0, 0, 0], (256,256,1))

with torch.no_grad():
    model.eval()

    for roi in allfiles:

        arcpy.AddMessage("Working on {}".format(roi))

        roi_file = os.path.join(image_path, "crop_temp", roi)

        output_image = os.path.join(output_path,os.path.basename(roi_file).split('.tif')[0] + '_unet.tif')
        output_conf = os.path.join(output_path, os.path.basename(roi_file).split('.tif')[0] + '_unet_conf.tif')

        # Read metadata of the initial image
        with rasterio.open(roi_file, mode ='r') as src:
            tags = src.tags().copy()
            meta = src.meta
            image = src.read()
            image = np.moveaxis(image, (0, 1, 2), (2, 0, 1)).astype('float32')
            dtype = src.read(1).dtype

        # Update meta to reflect the number of layers
        meta.update(count = 1)

        # Write prediction class
        with rasterio.open(output_image, 'w', **meta) as dst:

            nan_mask = np.isnan(image)
            image[nan_mask] = impute_nan[nan_mask]

            image = transform_test(image)
            logits = model(image.unsqueeze(0))

            # conf = np.amax(logits.numpy(), axis=1).squeeze()
            conf = np.amax(logits.numpy(), axis=1).squeeze()
            conf = logits2conf(conf).squeeze()

            probs = logits.softmax(dim=1).numpy().argmax(1).squeeze()

            dst.write_band(1, probs.astype(dtype).copy())

        # Write confidence (update the export file data type)
        meta.update(dtype='float32')
        with rasterio.open(output_conf, 'w', **meta) as dst:
            dst.write_band(1, conf)





