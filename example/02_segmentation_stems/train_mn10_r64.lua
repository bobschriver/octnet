#!/usr/bin/env th

local common = dofile('segmentation_common.lua')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('oc')

local opt = {}

opt.train_data_root = 'train/oc/data'
opt.train_label_root = 'train/oc/label'
opt.test_data_root = 'test/oc/data'
opt.test_label_root = 'test/oc/label'
opt.ext = 'oc'
opt.out_root = 'results'

opt.vx_size = 64
opt.batch_size = 4

opt.weightDecay = 0.0001
opt.learningRate = 1e-3
opt.n_epochs = 20
opt.learningRate_steps = {}
opt.learningRate_steps[15] = 0.1
opt.optimizer = optim['adam']

local n_grids = 4096
opt.net = nn.Sequential()
  :add( oc.OctreeConvolutionMM(3,6, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(6,6, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreePool2x2x2('max') )
  
  :add( oc.OctreeConvolutionMM(6,6, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(6,12, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeGridPool2x2x2('max') ) 
  
  :add( oc.OctreeConvolutionMM(12,12, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(12,24, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeGridPool2x2x2('max') )
  
  :add( oc.OctreeConvolutionMM(24,48, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(48,96, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeGridPool2x2x2('max') ) 

  :add( oc.OctreeConvolutionMM(96,96, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(96,96, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(96,96, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeGridUnpool2x2x2('max') ) 

  :add( oc.OctreeConvolutionMM(96,48, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(48,24, n_grids) ) 
  :add( oc.OctreeReLU(true) )  
  :add( oc.OctreeGridUnpool2x2x2('max') )   

  :add( oc.OctreeConvolutionMM(24,24, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(24,12, n_grids) )
  :add( oc.OctreeReLU(true) )  
  :add( oc.OctreeGridUnpool2x2x2('max') )   
    
  :add( oc.OctreeConvolutionMM(12,12, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(12,1, n_grids) ) 
  :add( oc.OctreeLogSoftMax() )   
  :add( oc.OctreeToCDHW() )

common.net_he_init(opt.net)
opt.net:cuda()
opt.criterion = nn.CrossEntropyCriterion()
opt.criterion:cuda()

common.segmentation_worker(opt)
