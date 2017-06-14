import sys
import csv
import numpy as np
import glob

sys.path.append('../00_create_data/')
import vis

sys.path.append('../../py/')
import pyoctnet

def xyz_to_oc(input_filename, output_rgb_filename, output_classification_filename):
	
	other_classification = 0.01
	stem_classification = 1.0
	
	with open(input_filename , 'rb') as csv_file:
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

		vis.write_ply_voxels(output_rgb_filename, oc_from_xyz_rgb.to_cdhw()[0])

		oc_from_xyz_classification = pyoctnet.Octree.create_from_pc(np.asarray(xyzs, dtype=np.float32), np.asarray(classifications, dtype=np.float32), vx_res, vx_res, vx_res, normalize=True)

		vis.write_ply_voxels(output_classification_filename, oc_from_xyz_classification.to_cdhw()[0])

train_path = 'train/xyz'
test_path = 'test/xyz'

xyz_ext = '.xyz'
ply_ext = '.ply'



