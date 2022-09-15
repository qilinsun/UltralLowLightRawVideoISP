import logging
import numpy as np
import os

import colour
from colour_demosaicing import (
    demosaicing_CFA_Bayer_DDFAPD,
    masks_CFA_Bayer)

from colour_hdri import (
    EXAMPLES_RESOURCES_DIRECTORY,
    Image,
    ImageStack,
    camera_space_to_sRGB,
    convert_dng_files_to_intermediate_files,
    convert_raw_files_to_dng_files,
    filter_files,
    read_exif_tag,
    image_stack_to_radiance_image,
    update_exif_tags,
    weighting_function_Debevec1997)
from colour_hdri.plotting import plot_radiance_image_strip

from colour_hdri import (
    EXAMPLES_RESOURCES_DIRECTORY,
    tonemapping_operator_simple,
    tonemapping_operator_normalisation,
    tonemapping_operator_gamma,
    tonemapping_operator_logarithmic,
    tonemapping_operator_exponential,
    tonemapping_operator_logarithmic_mapping,
    tonemapping_operator_exponentiation_mapping,
    tonemapping_operator_Schlick1994,
    tonemapping_operator_Tumblin1999,
    tonemapping_operator_Reinhard2004,
    tonemapping_operator_filmic)
from colour_hdri.plotting import plot_tonemapping_operator_image

logging.basicConfig(level=logging.INFO)
colour.plotting.colour_style()
colour.utilities.describe_environment()

# RAW_FILES = filter_files(RESOURCES_DIRECTORY, ('CR2',))
# DNG_FILES = convert_raw_files_to_dng_files(RAW_FILES, RESOURCES_DIRECTORY)
RESOURCES_DIRECTORY = 'images'
DNG_FILES = filter_files(RESOURCES_DIRECTORY, ('dng',))  #['images/5k.dng']


INTERMEDIATE_FILES = convert_dng_files_to_intermediate_files(DNG_FILES,RESOURCES_DIRECTORY,demosaicing=False) # False
update_exif_tags(zip(DNG_FILES, INTERMEDIATE_FILES))
tiff_img = colour.cctf_encoding(colour.read_image(str(INTERMEDIATE_FILES[-1]))) # (2014, 3040, 3)

colour.plotting.plot_image(tiff_img,text_kwargs={'text': os.path.basename(INTERMEDIATE_FILES[-1])})
print('>>> INTERMEDIATE_FILES  ',INTERMEDIATE_FILES)
print('>>> tiff_img.shape  ',tiff_img.shape) # (2014, 3040)

XYZ_TO_CAMERA_SPACE_MATRIX = colour.utilities.as_float_array(
    [float(M_c) for M_c in read_exif_tag(
        DNG_FILES[-1], 'ColorMatrix2').split()]).reshape((3, 3))
print('>>> XYZ_TO_CAMERA_SPACE_MATRIX  ',XYZ_TO_CAMERA_SPACE_MATRIX)

batch_size = 5
black_level = None
white_level = None
white_balance_multipliers = None
weighting_function = weighting_function_Debevec1997
CFA_pattern = 'RGGB'
output_directory = "images"

paths = []
# for dng_files in colour.utilities.batch(DNG_FILES, batch_size):
image_stack = ImageStack()
for dng_file in DNG_FILES:
    image = Image(dng_file)
    image.read_metadata()
    image.path = str(dng_file.replace('dng', 'tiff'))
    image.read_data()
    image_stack.append(image)

path = os.path.join(output_directory,'{0}_{1}_MRFPD.{2}'.
        format(os.path.splitext(os.path.basename(image_stack.path[0]))[0],
        batch_size,'exr'))
        # batch_size,'png'))
paths.append(path)

print('>>> image.metadata  ',image.metadata)

logging.info('Scaling "{0}"...'.format(', '.join(image_stack.path)))
black_level_e = (0 if
                    image_stack.black_level[0] is None else
                    np.max(image_stack.black_level[0]))
white_level_e = (1 if
                    image_stack.white_level[0] is None else
                    np.min(image_stack.white_level[0]))
logging.info('\tBlack Level (Exif): {0}'.format(
    image_stack.black_level))
logging.info('\tWhite Level (Exif): {0}'.format(
    image_stack.white_level))
black_level = black_level if black_level is not None else black_level_e
white_level = white_level if white_level is not None else white_level_e
logging.info('\tBlack Level (Used): {0}'.format(black_level))
logging.info('\tWhite Level (Used): {0}'.format(white_level))
print('>>> paths  ',paths)
# Scaling should be performed on individual planes, for convenience
# and simplicity the maximum of the black level and the minimum of
# the white level are used for all planes.
# print('>>> np.max(image_stack.data)  ',np.max(image_stack.data))   0.0642856
# print('>>> np.min(image_stack.data)  ',np.min(image_stack.data))   0
image_stack.data = (image_stack.data - black_level) * (1 / (white_level_e - black_level_e))
# print('>>> np.max min(image_stack.data)  ',np.max(image_stack.data),np.max(image_stack.data))   0.93308914405 0.000488400488287
logging.info('Merging "{0}"...'.format(path))
logging.info('\tImage stack "F Number" (Exif): {0}'.format(image_stack.f_number))
logging.info('\tImage stack "Exposure Time" (Exif): {0}'.format(image_stack.exposure_time))
logging.info('\tImage stack "ISO" (Exif): {0}'.format(image_stack.iso))
# print('>>> image_stack.data.shape  ',image_stack.data.shape) # (2014, 3040, 1)
# image = image_stack_to_radiance_image(image_stack, weighting_function)
# print('>>> image.shape  ',image.shape) # (2014, 3040, 1)
# image[np.isnan(image)] = 0
image = image_stack.data[:,:,0]

logging.info('White Balancing "{0}"...'.format(path))
white_balance_multipliers_e = np.power(image_stack.white_balance_multipliers[0], -1)
logging.info('\tWhite Balance Multipliers (Exif): {0}'.format(
    white_balance_multipliers_e))
white_balance_multipliers = (white_balance_multipliers
                                if white_balance_multipliers is not None
                                else white_balance_multipliers_e)
logging.info('\tWhite Balance Multipliers (Used): {0}'.format(white_balance_multipliers))
# For consistency and comparison ease with 
# *Colour - HDRI - Example: Merge from Raw Files* example, the 
# white balance multipliers are not normalised here too.
# white_balance_multipliers /= np.max(white_balance_multipliers)
R_m, G_m, B_m = masks_CFA_Bayer(image.shape, CFA_pattern)
print('>>> R_m, G_m, B_m  ',R_m, G_m, B_m)
logging.info('Demosaicing "{0}"...'.format(path))
image[R_m] *= white_balance_multipliers[0]
image[G_m] *= white_balance_multipliers[1]
image[B_m] *= white_balance_multipliers[2]
# print('>>> np.max min(image.data)  ',np.max(image.data),np.min(image.data))  # 1.34284784616 0.00103975834432
image = demosaicing_CFA_Bayer_DDFAPD(image, CFA_pattern)
print('>>> np.max min(image.data)  ',np.max(image.data),np.min(image.data))  # 1.34284784616 0.00103975834432
logging.info('Writing "{0}"...'.format(path))

# IMAGE = colour.read_image(PATHS[0])
IMAGE_sRGB = camera_space_to_sRGB(image, XYZ_TO_CAMERA_SPACE_MATRIX) # (2014, 3040, 3)
# print('>>> np.max min(IMAGE_sRGB.data)  ',np.max(IMAGE_sRGB.data),np.min(IMAGE_sRGB.data))  # 1.34284784616 0.00103975834432
IMAGE_sRGB = (IMAGE_sRGB - np.min(IMAGE_sRGB))/(np.max(IMAGE_sRGB)-np.min(IMAGE_sRGB))
print('>>> np.max min(IMAGE_sRGB.data)  ',np.max(IMAGE_sRGB.data),np.min(IMAGE_sRGB.data))  # 1.34284784616 0.00103975834432

# colour.write_image(IMAGE_sRGB, path.replace('.exr', '_sRGB.jpg'))
# colour.write_image(IMAGE_sRGB[:,:,::-1], path.replace('.exr', '_sRGB_inv.jpg'))
colour.write_image(IMAGE_sRGB[:,:,::-1]**0.8, path.replace('.exr', '_sRGB_gamma08.jpg'))
colour.write_image(IMAGE_sRGB[:,:,::-1]**0.85, path.replace('.exr', '_sRGB_gamma085.jpg'))
