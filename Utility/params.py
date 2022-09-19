"""
Adapted from Antoine Monod
Link: https://github.com/amonod/hdrplus-python/blob/main/package/algorithm/params.py
"""

import rawpy
import datetime
import os
import time
import argparse


def getParams(mode):
	'''Returns a dictionary of parameters that corresponds to a specific algorithm mode.'''
	params = {}
	if mode == 'full':
		params = {
			'time': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),

			'alignment': {
				'mode': 'bayer',  # images are single image Bayer / Color Filter Arrays
				'tuning': {
					# WARNING: these parameters are defined fine-to-coarse!
					'factors': [1, 2, 4, 4],
					'tileSizes': [16, 16, 16, 8],
					'searchRadia': [1, 4, 4, 4],
					'distances': ['L1', 'L2', 'L2', 'L2'],
					'subpixels': [False, True, True, True]  # if you want to compute subpixel tile alignment at each pyramid level
				},
				# rawpy parameters for images with motion fields
				'rawpyArgs': {
					'demosaic_algorithm' : rawpy.DemosaicAlgorithm.AHD,  # used in HDR+ supplement
					'half_size' : False,
					'use_camera_wb' : True,
					'use_auto_wb' : False,
					'no_auto_bright': True,
					'output_color' : rawpy.ColorSpace.sRGB,  # sRGB
					'output_bps' : 8
				},
				'writeMotionFields': False,
				'writeAlignedTiles': False
			},

			'merging': {
				'tuning': {
					'patchSize': 16,
					'method': 'DFTWiener',  # 'keepAlternate' / 'pairAverage' / 'DFTWiener'
					'noiseCurve': 'exifNoiseProfile'  # 'exifNoiseProfile' / 'exifISO' / tuple of (lambdaS, lambdaR) values (sigma = lambdaS*I + lambdaR)
				},
				'rawpyArgs': {
					'demosaic_algorithm' : rawpy.DemosaicAlgorithm.AHD,  # used in HDR+ supplement
					'half_size' : False,
					'use_camera_wb' : True,
					'use_auto_wb' : False,
					'no_auto_bright': True,
					'output_color' : rawpy.ColorSpace.sRGB,  # maybe try rawpy.colorSpace.raw and apply a color matrix in .txt?
					'gamma': (1, 1),  # gamma correction not applied by rawpy
					'output_bps' : 16
				},
				'writeReferenceImage': False,
				'writeGammaReference': False,
				'writeMergedBayer': False,
				'writeMergedImage': False,
				'writeGammaMerged': False
			},

			'finishing': {
				'tuning': {
					'ltmGain': 'auto',
					'gtmContrast': 0.075,
					'sharpenAmount': [1, 0.5, 0.5],
					'sharpenSigma': [1, 2, 4],
					'sharpenThreshold': [0.02, 0.04, 0.06]
				},
				# rawpy parameters for reference and final image
				'rawpyArgs': {
					'demosaic_algorithm' : rawpy.DemosaicAlgorithm.AHD,  # used in HDR+ supplement
					'half_size' : False,
					'use_camera_wb' : True,
					'use_auto_wb' : False,
					'no_auto_bright': True,
					'output_color' : rawpy.ColorSpace.sRGB,
					'gamma': (1, 1),  # gamma correction not applied by rawpy
					'output_bps' : 16
				},
				'writeReferenceImage': False,
				'writeGammaReference': True,
				'writeMergedImage': False,
				'writeGammaMerged': True,
				'writeShortExposure': False,
				'writeLongExposure': False,
				'writeFusedExposure': False,
				'writeLTMImage': False,
				'writeLTMGamma': False,
				'writeGTMImage': False,
				'writeReferenceFinal': True,
				'writeFinalImage': True
			}
		}

	elif mode == 'align':
		params = {
			'time': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),

			'alignment': {
				'mode': 'bayer',  # images are single image Bayer / Color Filter Arrays
				'tuning': {
					# WARNING: these parameters are defined fine-to-coarse!
					'factors': [1, 2, 4, 4],
					'tileSizes': [16, 16, 16, 8],
					'searchRadia': [1, 4, 4, 4],
					'distances': ['L1', 'L2', 'L2', 'L2'],
					'subpixels': [False, True, True, True]  # if you want to compute subpixel tile alignment at each pyramid level
				},
				# rawpy parameters for images with motion fields
				'rawpyArgs': {
					'demosaic_algorithm' : rawpy.DemosaicAlgorithm.AHD,  # used in HDR+ supplement
					'half_size' : False,
					'use_camera_wb' : True,
					'use_auto_wb' : False,
					'no_auto_bright': True,
					'output_color' : rawpy.ColorSpace.sRGB,  # sRGB
					'output_bps' : 8
				},
				'writeMotionFields': False,
				'writeAlignedTiles': True
			},

			'merging': None,

			'finishing': None
		}

	elif mode == 'merge':
		params = {
			'time': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),

			'alignment': None,  # assumes burst already aligned

			'merging': {
				'tuning': {
					'patchSize': 16,
					'method': 'DFTWiener',  # 'keepAlternate' / 'pairAverage' / 'DFTWiener'
					'noiseCurve': 'exifNoiseProfile'  # 'exifNoiseProfile' / 'exifISO' / tuple of (lambdaS, lambdaR) values (sigma = lambdaS*I + lambdaR)
				},
				'rawpyArgs': {
					'demosaic_algorithm' : rawpy.DemosaicAlgorithm.AHD,  # used in HDR+ supplement
					'half_size' : False,
					'use_camera_wb' : True,
					'use_auto_wb' : False,
					'no_auto_bright': True,
					'output_color' : rawpy.ColorSpace.sRGB,  # maybe try rawpy.colorSpace.raw and apply a color matrix in .txt?
					'gamma': (1, 1),  # gamma correction not applied by rawpy
					'output_bps' : 16
				},
				'writeReferenceImage': False,
				'writeGammaReference': False,
				'writeMergedBayer': True,
				'writeMergedImage': False,
				'writeGammaMerged': False
			},

			'finishing': None
		}

	elif mode == 'finish':
		params = {
			'time': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),

			'alignment': None,  # assumes burst already aligned

			'merging': None,  # assumes burst already merged

			'finishing': {
				'tuning': {
					'ltmGain': 'auto',
					'gtmContrast': 0.075,
					'sharpenAmount': [1, 0.5, 0.5],
					'sharpenSigma': [1, 2, 4],
					'sharpenThreshold': [0.02, 0.04, 0.06]
				},
				# rawpy parameters for reference and final image
				'rawpyArgs': {
					'demosaic_algorithm' : rawpy.DemosaicAlgorithm.AHD,  # used in HDR+ supplement
					'half_size' : False,
					'use_camera_wb' : True,
					'use_auto_wb' : False,
					'no_auto_bright': True,
					'output_color' : rawpy.ColorSpace.sRGB,
					'gamma': (1, 1),  # gamma correction not applied by rawpy
					'output_bps' : 16
				},
				'writeReferenceImage': False,
				'writeGammaReference': False,
				'writeMergedImage': False,
				'writeGammaMerged': False,
				'writeShortExposure': False,
				'writeLongExposure': False,
				'writeFusedExposure': False,
				'writeLTMImage': False,
				'writeLTMGamma': False,
				'writeGTMImage': False,
				'writeReferenceFinal': False,
				'writeFinalImage': True
			}
		}
	return params
