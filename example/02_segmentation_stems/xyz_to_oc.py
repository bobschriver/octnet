import sys
import csv
import numpy as np
import glob
import os

sys.path.append('../00_create_data/')
import vis

sys.path.append('../../py/')
import pyoctnet

def xyz_to_oc(input_filepath, output_prefix, output_oc_path, output_ply_path):
	
	other_classification = 0.01
	stem_classification = 1.0

	print 'Reading from %s' % input_filepath	
	with open(input_filepath, 'rb') as csv_file:
		reader = csv.reader(csv_file)
		version = reader.next()
		projection = reader.next()
		header = reader.next()

		x_index = header.index('X')
		y_index = header.index('Y')
		z_index = header.index('Z')

		r_index = header.index('Red')
		g_index = header.index('Green')
		b_index = header.index('Blue')

		intensity_index = header.index('Intensity')
	
		classification_index = header.index('Classification')

		xyzs = []
		features = []
		classifications = []
		classifications_other = []

		for row in reader:
			x = float(row[x_index])
			y = float(row[y_index])
			z = float(row[z_index])

			intensity = float(row[intensity_index])
		
			r = (float(row[r_index]) - intensity) / 255
			g = (float(row[g_index]) - intensity) / 255
			b = (float(row[b_index]) - intensity) / 255
		
			classification = int(float(row[classification_index]))
		
			xyzs.append((x, y, z))
			features.append((r, g, b))

			if classification == 20:
				classifications.append((stem_classification, 1.0))
			else:
				classifications.append((other_classification, 1.0))
		vx_res = 64

		oc_from_xyz_rgb = pyoctnet.Octree.create_from_pc(np.asarray(xyzs, dtype=np.float32), np.asarray(features, dtype=np.float32), vx_res, vx_res, vx_res, normalize=True)
		
		oc_rgb_filepath = os.path.join(output_oc_path, '%s_rgb.oc' % output_prefix)
		print 'Writing to %s' % oc_rgb_filepath
		oc_from_xyz_rgb.write_bin(oc_rgb_filepath)

		ply_rgb_filepath = os.path.join(output_ply_path, '%s_rgb.ply' % output_prefix)
		print 'Writing to %s' % ply_rgb_filepath
		vis.write_ply_voxels(ply_rgb_filepath, oc_from_xyz_rgb.to_cdhw()[0])

		oc_from_xyz_classification = pyoctnet.Octree.create_from_pc(np.asarray(xyzs, dtype=np.float32), np.asarray(classifications, dtype=np.float32), vx_res, vx_res, vx_res, normalize=True)
		
		oc_classification_filepath = os.path.join(output_oc_path, '%s_classification.oc' % output_prefix)
		print 'Writing to %s' % oc_classification_filepath
		oc_from_xyz_classification.write_bin(oc_classification_filepath)

		ply_classification_filepath = os.path.join(output_ply_path, '%s_classification.ply' % output_prefix)
		print 'Writing to %s' % ply_classification_filepath
		vis.write_ply_voxels(ply_classification_filepath, oc_from_xyz_classification.to_cdhw()[0])

train_path = 'train'
test_path = 'test'

input_path = 'xyz'
output_oc_path = 'oc'
output_ply_path = 'ply'

xyz_ext = '.xyz'
ply_ext = '.ply'
oc_ext = '.oc'

for input_path in glob.glob(os.path.join(test_path, input_path, '*' + xyz_ext)):
	head, tail = os.path.split(input_path)
	filename, ext = os.path.splitext(tail)

	xyz_to_oc(input_path, filename, os.path.join(test_path, output_oc_path), os.path.join(test_path, output_ply_path))
