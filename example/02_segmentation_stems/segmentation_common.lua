local common = dofile('../common.lua')
local dataloader = dofile('dataloader.lua')

function common.segmentation_worker(opt)
	print(string.format('out_root: %s', opt.out_root))
  	-- create out root dir
  	paths.mkdir(opt.out_root)

	print ('[INFO] load train data paths')
	local train_data_paths = common.walk_paths_cached(opt.train_data_root, opt.ext)
	
	print ('[INFO] load train label paths')
	local train_label_paths = common.walk_paths_cached(opt.train_label_root, opt.ext)
	
	print ('[INFO] load test data paths')
	local test_data_paths = common.walk_paths_cached(opt.test_data_root, opt.ext)

	print ('[INFO] load test label paths')
	local test_label_paths = common.walk_paths_cached(opt.test_label_root, opt.ext)

	opt.train_data_paths = train_data_paths
	opt.train_label_paths = train_label_paths

	opt.test_data_paths = test_data_paths
	opt.test_label_paths = test_label_paths

	local train_data_loader = dataloader.DataLoader(opt.train_data_paths, opt.train_label_paths, opt.batch_size)
	local test_data_loader = dataloader.DataLoader(opt.test_data_paths, opt.test_label_paths, opt.batch_size)

	common.worker(opt, train_data_loader, test_data_loader)
end

return common
