dataloader = {}

local DataLoader = torch.class('dataloader.DataLoader')

function DataLoader:__init(data_paths, label_paths, batch_size)
	self.data_paths = data_paths or error('')
	self.label_paths = label_paths or error('')
	self.batch_size = batch_size or error('')

	self.data_idx = 0
	self.label_idx = 0
end

function DataLoader:getBatch()
	local bs = math.min(self.batch_size, #self.data_paths - self.data_idx)

	local used_data_paths = {}
	local used_label_paths = {}

	for batch_idx = 1, bs do
		self.data_idx = self.data_idx + 1

		local used_data_path = self.data_paths[self.data_idx]
		local used_label_path = self.label_paths[self.data_idx]

		print(used_data_path)
		print(used_label_path)

		table.insert(used_data_paths, used_data_path)
		table.insert(used_label_paths, used_label_path)	
	end

	self.data_cpu = oc.FloatOctree()
	self.data_cpu:read_from_bin_batch(used_data_paths)
	self.data_gpu = self.data_cpu:cuda(self.data_gpu)

	self.label_cpu = oc.FloatOctree()
	self.label_cpu:read_from_bin_batch(used_label_paths)
	self.label_gpu = self.label_cpu:cuda(self.label_gpu)


	if (#self.data_paths - self.data_idx) < self.batch_size then
		self.data_idx = 0
	end

	collectgarbage(); collectgarbage()

	return self.data_gpu, self.label_gpu
end

return dataloader
