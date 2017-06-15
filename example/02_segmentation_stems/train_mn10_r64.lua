#!/usr/bin/env th

local common = dofile('classification_common.lua')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('oc')

local opt = {}
opt.ex_data_root = 'preprocessed'
opt.ex_data_ext = 'oc'
opt.out_root = 'results'
opt.vx_size = 64
opt.n_classes = 10
opt.batch_size = 4

opt.weightDecay = 0.0001
opt.learningRate = 1e-3
opt.n_epochs = 20
opt.learningRate_steps = {}
opt.learningRate_steps[15] = 0.1
opt.optimizer = optim['adam']

local n_grids = 4096
opt.net = nn.Sequential()
  :add( oc.OctreeConvolutionMM(8,8, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(8,16, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeGridPool2x2x2('max') )
  
  :add( oc.OctreeConvolutionMM(16,16, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(16,32, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeGridPool2x2x2('max') ) 
  
  :add( oc.OctreeConvolutionMM(32,32, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(32,64, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeGridPool2x2x2('max') )
  
  :add( oc.OctreeConvolutionMM(64,64, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(64,128, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeGridPool2x2x2('max') ) 

  :add( oc.OctreeConvolutionMM(128,128, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(128,128, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(128,128, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeGridUnPool2x2x2('max') ) 

  :add( oc.OctreeConcat() )
  :add( oc.OctreeConvolutionMM(128,64, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(64,32, n_grids) ) 
  :add( oc.OctreeReLU(true) )  
  :add( oc.OctreeGridUnPool2x2x2('max') )   

  :add( oc.OctreeConcat() )
  :add( oc.OctreeConvolutionMM(64,32, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(32,16, n_grids) )
  :add( oc.OctreeReLU(true) )  
  :add( oc.OctreeGridUnPool2x2x2('max') )   
    
  :add( oc.OctreeConcat() )
  :add( oc.OctreeConvolutionMM(32,32, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(32,8, n_grids) ) 
  :add( oc.OctreeReLU(true) )  
  :add( oc.OctreeGridUnPool2x2x2('max') )   
  :add( oc.OctreeLogSoftMax() )   

  
common.net_he_init(opt.net)
opt.net:cuda()
opt.criterion = nn.CrossEntropyCriterion()
opt.criterion:cuda()

common.classification_worker(opt)
