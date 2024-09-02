
# Project path for exporting-----------
EXPORT_PATH = 'projects/your_projectpath/assets/'
print(EXPORT_PATH)
# ------------------------------------------------
import ee

# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
ee.Initialize(project='ee-project)

print("hello world")
# Import other packages used in the tutorial
import canty # IRMAD>>this module needs to be saved in the same folder as the Jupyter
import geemap
import numpy as np
#####################################################################################
# Functions for handling data and masking
def collect(aoi, date, date2): # Collects the first image within the specified time range for the first and second period, filters by aoi
    try:
        # Collect the first image within the specified time range for the first period
        im1 = ee.Image( ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                               .filterBounds(aoi)
                               .filterDate(ee.Date(date), ee.Date(date).advance(1, 'day'))
                               .filter(ee.Filter.contains(rightValue=aoi,leftField='.geo'))
                               .sort('CLOUDY_PIXEL_PERCENTAGE')
                               .first()
                               .clip(aoi) )
       
        # Collect the first image within the specified time range for the second period
        im2 = ee.Image( ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                               .filterBounds(aoi)
                               .filterDate(ee.Date(date2), ee.Date(date2).advance(1, 'day'))
                               .filter(ee.Filter.contains(rightValue=aoi,leftField='.geo'))
                               .sort('CLOUDY_PIXEL_PERCENTAGE')
                               .first()
                               .clip(aoi) )
        # Get the timestamps of the collected images
        timestamp1 = im1.date().format('E MMM dd HH:mm:ss YYYY')
        print(timestamp1.getInfo())
        timestamp2 = im2.date().format('E MMM dd HH:mm:ss YYYY')
        print(timestamp2.getInfo())
        # Get the image IDs
        image1_id = im1.id().getInfo()
        image2_id = im2.id().getInfo()

        # Print the image IDs
        print("Image 1 ID:", image1_id)
        print("Image 2 ID:", image2_id)
       
        # Return the collected images
        return (im1, im2)
   
    except Exception as e:
        print('Error: %s'%e)
def apply_mask(image, mask):
        """
        Apply a mask to a Sentinel image.

        Args:
            image (ee.Image): Sentinel image.
            mask (ee.Image): Mask image.

        Returns:
            ee.Image: Sentinel image with the mask applied.
        """
        masked_image = image.updateMask(mask)

        return masked_image
    # Load ESA WorldCover dataset
def export_image(image, folder_path, image_name, scale=20):
    """
    Export an image from Google Earth Engine using geemap.

    Args:
        image (ee.image.Image): The image to export.
        folder_path (str): The path to the folder where the image will be exported.
        image_name (str): The name to use for the exported file.
        scale (int, optional): The scale of the exported image. Defaults to 30.
    """
    # Create the full export path
    export_path = f"{folder_path}\\{image_name}.tif"

    # Create a geemap Map instance
    Map = geemap.Map()

    # Add the image to the map
    Map.addLayer(image, {}, 'Image')

    # Export the image
    geemap.ee_export_image(image, export_path, scale=scale)

    # Display the map
    Map
def process_images(aoi, dates, visirbands, city_mask):
    """
    Processes images based on the provided area of interest (AOI), dates, visible/infrared bands, and city mask.

    Parameters:
    aoi (ee.Geometry): The area of interest for image processing.
    dates (list): A list of date strings in the format 'YYYY-MM-DD' to filter the image collection.
    visirbands (list): A list of band names to select from the image collection.
    city_mask (ee.Image): An image mask to apply to the processed images.

    Returns:
    None
    """
   
    for i in range(len(dates) - 1):
        date1 = dates[i]
        date2 = dates[i + 1]

        # Collect the two Sentinel-2 images for the specified dates
        im1, im2 = collect(aoi, date1, date2)
       
        # Apply the city mask to the images
        im1 = apply_mask(im1, city_mask)
        im2 = apply_mask(im2, city_mask)

        # Generate the output name based on the date range
        output_name = f'FINAL20MAcrosstherange_{date1}_{date2}'
        file_names.append(output_name)  # Append the output name to the list

        # Run the IMAD function on the masked images
        canty.run_imad(aoi, im1.select(visirbands), im2.select(visirbands), output_name)

# Folder path for exporting images to the local machine
folder_path = r'C:\Users\jachy\Desktop\iMAD\Outputs'
#####################################################################################
# Areo of interest (AOI) for the study as a featureCollection derived from a shapefile located as an asset in the Earth Engine
aoi = ee.FeatureCollection(
    'projects/ee-thesiswar/assets/Gaza_AOI').geometry() #North Gaza
#####################################################################################
# Masking using the Dynamic World
START = ee.Date('2023-09-27') # select desired day
# Define the end date by advancing the start date by one day
END = START.advance(1, 'day')

# Create a composite filter that combines spatial and temporal filters
col_filter = ee.Filter.And(
    ee.Filter.bounds(ee.Geometry(aoi)),  # Filter to include only images intersecting the area of interest (AOI)
    ee.Filter.date(START, END),  # Filter to include only images within the specified date range
)

# Apply the filter to the Dynamic World Image Collection
dw_col = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filter(col_filter)

# Define the class names present in the Dynamic World dataset
CLASS_NAMES = [
    'water',
    'trees',
    'grass',
    'flooded_vegetation',
    'crops',
    'shrub_and_scrub',
    'built',
    'bare',
    'snow_and_ice',
]

# Extract the first image from the filtered collection
dw_image = ee.Image(dw_col.first())
# Clip the Dynamic World image to the area of interest
dw_image_clipped = dw_image.clip(aoi)
built_mask = dw_image_clipped.select('label').eq(CLASS_NAMES.index('built'))

# Generate a binary layer where 'built' areas are 1 and others are 0, then apply selfMask to keep only 'built' areas
binary_layer = built_mask.gt(0).selfMask()
# Update the mask of the built_mask layer with the binary_layer to isolate 'built' areas
city_mask = built_mask.updateMask(binary_layer)
#############################################################################
# execution of Pre-procesing
# Define the bands to be used in the analysis
visirbands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
# Define the dates for the two periods
file_names = [] # empty list to save the names of the processed images
# Batch Processing of dates
dates = [
    "2023-09-27",
    "2023-11-01",
    "2023-11-26",
    "2023-12-26",
    "2024-01-20",
    "2024-03-05",
    "2024-04-04",
    "2024-05-09",
    "2024-06-18",
    "2024-07-18",
]
dates =  ["2023-09-27","2024-07-23"]# for individual processing, in case of not needing batch
process_images(aoi, dates, visirbands, city_mask) # Function to process images in batch and send tasks to GEE for processing
# Run the IRMAD >> parameters: (aoi, image1, image2, output_name) individdually
#canty.run_imad(aoi, im1.select(visirbands), im2.select(visirbands),'ExampleIRMAD') # Function to run IRMAD on two images
#######################################################################################
alpha_values = [0.00005]
# Iterate through all the p values in the list (option for more)
# Iterate through each file name.and export the binary masks locally into the a local machine
for file_name in file_names:
    try:
        # Load the image from GEE.
        im_z = ee.Image(EXPORT_PATH + file_name).select(6).rename('Z') # assuming that the 6th band is the Z-score and we have 6 bands for the analysis

        # p-values image by canty
        pval = canty.chi2cdf(im_z, 6).subtract(1).multiply(-1).rename('pval')
        # Iterate through all the p values in the list
        for p_value in alpha_values:
            # Create a binary mask: 1 where pval is less than p_value (indicating change), 0 otherwise (indicating no change)
            binaryMask = pval.lt(p_value).rename('binaryMask')

            # Define the export name by removing the dot from the p_value
            export_name = f'MASKS{str(p_value).replace(".", "_")}_{file_name}'

            # Export the binaryMask with the unique name
            #export_image(binaryMask, folder_path, export_name, scale=20) # Export the binary mask
            print(f"Exported mask with p-value: {p_value} for {file_name}")

    except Exception as e:
        print(f"Error processing file {file_name}: {e}")

print("Export is finished.")
# Iterate through each file name.for clustering
region = aoi
for file_name in file_names:
    try:
        # Load the image from GEE.
        input = ee.Image(EXPORT_PATH + file_name).select(0, 1, 2, 3, 4, 5)

        # Make the training dataset.and set scales
        training = input.sample(region=region, scale=20, numPixels=50000)

        # Instantiate the clusterer and train it.based on the training dataset and the mean
        clusterer = ee.Clusterer.wekaKMeans(10).train(training)

        # Cluster the input using the trained clusterer.
        result = input.cluster(clusterer)
        # Export the clustered image with a unique name.
        export_name = f'cluster20_6classes10ss{file_name}'
        #export_image(result, folder_path, export_name, scale=20)
        print(f"Exported cluster for {file_name}")

    except Exception as e:
        print(f"Error processing file {file_name}: {e}")

print("Export is finished.")

