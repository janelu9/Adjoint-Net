# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:37:42 2020

@author: kfzx-lujian
"""

# Copyright (c) 2018 ICBC Authors. All Rights Reserved

import os
#from flask import Flask, request, jsonify
from flask import jsonify
import json

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import h5py
import models
import faiss

import time
from logging.handlers import TimedRotatingFileHandler,RotatingFileHandler
import logging

from threading import Thread

env = os.environ
args = type('args',(),{})
args.port = int(env.get("PORT",6520))
args.saved_model = env.get("saved_model","saved_model")
args.index_file = env.get("index","index")
args.vec_file = env.get("vector","vector")
#args.topk = int(env.get("topk",1024))
args.nprob = 4

class mylogger(logging.Logger):
    """
    mylogger
    """
    def __init__(self, name='log', level=logging.INFO, time = False,fmt=None):
        self.level = level
        self.time = time
        self.LOG_PATH = env.get('logdir')
        if not os.path.exists(self.LOG_PATH):
            os.makedirs(self.LOG_PATH)
        self.file_name = os.path.join(self.LOG_PATH, '{name}.log'.format(name=name))
        
        self.fmt= '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s' if fmt is None else fmt
 
        super(mylogger, self).__init__(name=name, level=level)        
        self.__setFileHandler__()                
                
    def __setFileHandler__(self, level=None):
        file_handler = TimedRotatingFileHandler(filename=self.file_name, when='D', interval=7, backupCount=5) if self.time\
                        else RotatingFileHandler(self.file_name, maxBytes=1024*1024*100, backupCount=5)
        if not level:
            file_handler.setLevel(self.level)
        else:
            file_handler.setLevel(level)
        formatter = logging.Formatter(self.fmt)
        file_handler.setFormatter(formatter)
        self.file_handler = file_handler
        self.file_handler.suffix = "%Y%m%d"
        self.addHandler(self.file_handler)

logger=mylogger()  
 
model=None
VECDIM=None
num_feat_num=None
index=None
tags=None

def backstage_prepare():
    while not (os.path.isfile("oaas_download_done") and os.path.isfile("gpu_test_done")):
        if not os.path.isfile("gpu_test_done"):
            logger.info("wait for gpu testing ...")
        if not os.path.isfile("oaas_download_done"):
            logger.info("wait for oaas downloading ...")        
        time.sleep(3)        
    st=time.time()
    def copy_param(m,saved_param):
        model_param=m.state_dict()
        for k,v in model_param.items():
            assert k in saved_param,f'parameter: {k} not found '
            model_param[k]=saved_param[k]
        return model_param

    global model,VECDIM,num_feat_num,index,tags
    logger.info("loading model...")
    check_point_dict=torch.load(os.path.join(args.saved_model,os.listdir(args.saved_model)[0]))
    VECDIM=check_point_dict['Query.kwargs']['vec_dim']
    num_feat_num=check_point_dict['Query.kwargs']['num_feat_num']
    model=models.QueryModel(
                            points_total_num=check_point_dict['shared_kwargs']['points_total_num'],
                            emb_dim=check_point_dict['shared_kwargs']['emb_dim'],
                            query_param=check_point_dict['Query.kwargs']
                            ).cuda()
                            
    saved_param=check_point_dict['model_param']
    model.load_state_dict(copy_param(model,saved_param))
    model.eval()
    #load faiss index
    logger.info("loading index...")
    index= faiss.read_index(os.path.join(args.index_file,os.listdir(args.index_file)[0]))
    res=faiss.StandardGpuResources()
    #res.setTempMemory(2*1024*1024*1024)
    index = faiss.index_cpu_to_gpu(res,0,index)
    index.nprobe=args.nprob
    #load tags
    logger.info("loading tags...")
    with  h5py.File(os.path.join(args.vec_file,os.listdir(args.vec_file)[0]),'r') as f:
        tags=f['tags'][()]
    #tags=np.array(["%015d"%i for i in tags])
    dury=time.time()-st
    logger.info(f"initialization completed in {dury} seconds, enjoin the service!")
    
#load model
Thread(target=backstage_prepare,daemon=True).start()


def preprocess(x):
    num,cat=torch.split(torch.from_numpy(np.array(x,'float32')).cuda().reshape((1,1,-1)),num_feat_num,dim=-1)
    return None,num,cat.long(),None
    
def process(x):
    with torch.no_grad():
        y=model(*x)        
    return y.reshape(1,VECDIM).cpu().numpy()

def reprocess(x,k):
    _,I = index.search(x,k)
    return list(tags[I.reshape(-1)]) 

def query(request):         
    try:
        st1=time.time()
        js=json.loads(request.data.decode('utf-8') )
        x=js.get('query')
        k=int(js.get('topk'))
        st2=time.time()
        x=preprocess(x)
        x=process(x)
        y=reprocess(x,k)
        ed=time.time()
        dury1=ed-st1
        dury2=ed-st2
        logger.info(f"{dury1},{dury2}")
        return jsonify({"success":"0","errorMsg":None,"items":y})
    except Exception as e:
        logger.error(e)
        if tags is None:
            return jsonify({"success":"1","errorMsg":"Initialization hasn't been completed, wait for minutes please! ","items":None})
        return jsonify({"success":"1","errorMsg":str(e),"items":None})
       
