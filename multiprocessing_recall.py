# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:37:42 2020

@author: kfzx-lujian
"""

# Copyright (c) 2018 ICBC Authors. All Rights Reserved

import os
import json

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import h5py
import models
import faiss

import time
import logging
import csv

import multiprocessing 
from functools import partial


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("torch")
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser("twins")
    parser.add_argument("--input_file", type=str, help="infer file")   
    parser.add_argument("--saved_model", type=str, help="model file to restore ")    
    parser.add_argument("--model_type", type=str, default='query', help="which side to infer")
    parser.add_argument("--local_yield", type=bool, default=False, help="feed data from memory or local")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for infer")
    parser.add_argument("--recall_num", type=int, default=128, help="recall num")
    parser.add_argument("--cuda", type=bool, default=True, help="if using cuda")
    parser.add_argument("--index_input_dir", type=str,default='index_files', help="index input dir")    
    parser.add_argument("--tag_input_dir", type=str,default='tag_input_dir', help="tag input dir")      
    parser.add_argument("--result_output_dir", type=str,default='result_files', help="result output dir") 
    return parser.parse_args()


#load tags
logger.info("loading tags...")
with  h5py.File(os.path.join(args.tag_input_dir,os.listdir(args.tag_input_dir)[0]),'r') as f:
    Tags=f['tags'][()]
Tags=np.array(["%015d"%i for i in Tags])


def barch_infer(file,block):      
    check_point_dict=torch.load(args.saved_model+'/model.tar')    
    VECDIM=check_point_dict['Query.kwargs']['vec_dim']
    sub_model=args.model_type.lower()
    if sub_model=='item':        
        model=models.ItemModel(
            points_total_num=check_point_dict['shared_kwargs']['points_total_num'],
            emb_dim=check_point_dict['shared_kwargs']['emb_dim'],
            prod_param=check_point_dict['Item.kwargs']
            )
        if args.local_yield:
            dl=data_loader.MemoryInferFromLocal(file
            ,num_num=check_point_dict['Item.kwargs']['num_feat_num']
            ,cat_num=check_point_dict['Item.kwargs']['cat_feat_num']
            ,seq_num=check_point_dict['Item.kwargs']['seq_feat_num']
            ,batch_size=args.batch_size
            )
            #dl=torch.utils.data.DataLoader(ds,batch_size=args.batch_size,num_workers=0,shuffle=False,drop_last=False)
        else:
            ds=data_loader.InferFromMemery(args.input_file
            ,num_num=check_point_dict['Item.kwargs']['num_feat_num']
            ,cat_num=check_point_dict['Item.kwargs']['cat_feat_num']
            ,seq_num=check_point_dict['Item.kwargs']['seq_feat_num']
            )
            dl=torch.utils.data.DataLoader(ds,batch_size=args.batch_size,num_workers=2,shuffle=False,drop_last=False)                
    else:        
        model=models.QueryModel(
            points_total_num=check_point_dict['shared_kwargs']['points_total_num'],
            emb_dim=check_point_dict['shared_kwargs']['emb_dim'],
            query_param=check_point_dict['Query.kwargs']
            )
        if args.local_yield:
            dl=data_loader.MemoryInferFromLocal(file
            ,num_num=check_point_dict['Query.kwargs']['num_feat_num']
            ,cat_num=check_point_dict['Query.kwargs']['cat_feat_num']
            ,seq_num=check_point_dict['Query.kwargs']['seq_feat_num']
            ,batch_size=args.batch_size
            )
            #dl=torch.utils.data.DataLoader(ds,batch_size=args.batch_size,num_workers=0,shuffle=False,drop_last=False)
        else:
            ds=data_loader.InferFromMemery(args.input_file
            ,num_num=check_point_dict['Query.kwargs']['num_feat_num']
            ,cat_num=check_point_dict['Query.kwargs']['cat_feat_num']
            ,seq_num=check_point_dict['Query.kwargs']['seq_feat_num']
            )
            dl=torch.utils.data.DataLoader(ds,batch_size=args.batch_size,num_workers=2,shuffle=False,drop_last=False)   
    
    def copy_param(m,saved_param):
        model_param=m.state_dict()
        for k,v in model_param.items():
            assert k in saved_param,f'parameter: {k} not found '
            model_param[k]=saved_param[k]
        return model_param

    if args.cuda:
        model=model.cuda()

    logger.info('Load model ...')        
    epoch_dict=check_point_dict['model_param']
    model.load_state_dict(copy_param(model,epoch_dict))
    model.eval()
    
    #load faiss index
    logger.info("loading index...")
    index= faiss.read_index(os.path.join(args.index_file,os.listdir(args.index_file)[0]))
    res=faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res,0,index)
    index.nprobe=args.nprob
    dury=time.time()-st
    logger.info(f"initialization completed in {dury} seconds, enjoin the service!")
    
    logger.info('Recalling ...')
    tags=[]
    recalls=[]
    cnt=0
    lv=0
    with open('part-%08d.csv' % block,'w',encoding='utf-8') as f:
        csv_writer = csv.writer(f,delimiter=chr(27))
        with torch.no_grad():
            for kv in dl:
                tag=kv['ids']
                if args.cuda:
                    vec=model(kv['ids'],kv['num'].cuda(),kv['cat'].cuda(),kv['seq'].cuda()).reshape(-1,VECDIM).cpu().numpy().astype('float32')
                else:
                    vec=model(kv['ids'],kv['num'],kv['cat'],kv['seq']).reshape(-1,VECDIM).cpu().numpy().astype('float32')
                _,I = index.search(vec,args.recall_num)
                for i,j in zip(Tags[I.reshape(-1)].tolist(),tag.tolist()):
                    csv_writer.writelines(i+j)
                
                
           
    


       
