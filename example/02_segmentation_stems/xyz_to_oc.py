import sys
import csv
import numpy as np

sys.path.append('../00_create_data/')
import vis

sys.path.append('../../py/')
import pyoctnet

filename = 'points/points_2_36_tr1.xyz'

with open(filename , 'rb') as csv_file:
	reader = csv.reader(csv_file)
	version = reader.next()
	projection = reader.next()
	header = reader.next()

	print "its chu"

	x_index = header.index('X')
	y_index = header.index('Y')
	z_index = header.index('Z')

	r_index = header.index('Red')
	g_index = header.index('Green')
	b_index = header.index('Blue')

	intensity_index = header.index('Intensity')
	
	classification_index = header.index('Classification')

	min_x = 0.0
	min_y = 0.0
	min_z = 0.0

	max_x = 0.0
	max_y = 0.0
	max_z = 0.0

	xyzs = []
	features = []
	classifications_stem = []
	classifications_other = []

	for row in reader:
		x = float(row[x_index])
		y = float(row[y_index])
		z = float(row[z_index])

		intensity = float(row[intensity_index])
		
		r = int(float(row[r_index]) - intensity)
		g = int(float(row[g_index]) - intensity)
		b = int(float(row[b_index]) - intensity)
		
		classification = int(float(row[classification_index]))
		
		xyzs.append((x, y, z))
		features.append((r, g, b))

		if classification == 20:
			classifications_stem.append((1, 1))
			classifications_other.append((0, 0))
		else:
			classifications_other.append((1, 1))
			classifications_stem.append((0, 0))

		if x < min_x:
			min_x = x

		if y < min_y:
			min_y = y

		if z < min_z:
			min_z = z


		if x > max_x:
			max_x = x

		if y > max_y:
			max_y = y
		
		if z > max_z:
			max_z = z

	print (min_x, min_y, min_z)
	print (max_x, max_y, max_z)

	shifted_min_x = 0.0
	shifted_max_x = max_x - min_x

	shifted_min_y = 0.0
	shifted_max_y = max_y - min_y

	shifted_min_z = 0.0
	shifted_max_z = max_z - min_z

	print (shifted_max_x, shifted_max_y, shifted_max_z)

	vx_res = 64

	oc_from_xyz_rgb = pyoctnet.Octree.create_from_pc(np.asarray(xyzs, dtype=np.float32), np.asarray(features, dtype=np.float32), vx_res, vx_res, vx_res, normalize=True)

	vis.write_ply_voxels('oc_from_xyz_rgb_64.ply', oc_from_xyz_rgb.to_cdhw()[0])

	oc_from_xyz_stem = pyoctnet.Octree.create_from_pc(np.asarray(xyzs, dtype=np.float32), np.asarray(classifications_stem, dtype=np.float32), vx_res, vx_res, vx_res, normalize=True)

	vis.write_ply_voxels('oc_from_xyz_stem_64.ply', oc_from_xyz_stem.to_cdhw()[0], color=[255, 255, 255])

	oc_from_xyz_other = pyoctnet.Octree.create_from_pc(np.asarray(xyzs, dtype=np.float32), np.asarray(classifications_other, dtype=np.float32), vx_res, vx_res, vx_res, normalize=True)

	vis.write_ply_voxels('oc_from_xyz_other_64.ply', oc_from_xyz_other.to_cdhw()[0], color=[255, 255, 255])
