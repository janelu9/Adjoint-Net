# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:24:30 2020

@author: kfzx-lujian
"""

# Copyright (c) 2018 ICBC Authors. All Rights Reserved


import os
import time

import numpy as np
import argparse
import logging

import h5py

import torch

import data_loader
import models 

import faisslib


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("torch")
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser("twins")
    parser.add_argument("--input_file", type=str, help="infer file")   
    parser.add_argument("--saved_model", type=str, help="model file to restore ")    
    parser.add_argument("--model_type", type=str, default='item', help="which side to infer")
    parser.add_argument("--local_yield", type=bool, default=False, help="feed data from memory or local")
    parser.add_argument("--batch_size", type=int, default=2000, help="Batch size for training")
    parser.add_argument("--cuda", type=bool, default=True, help="if using cuda")
    parser.add_argument("--index", type=bool, default=True, help="if output index")
    parser.add_argument("--fc_param", type=str, default="IVF4096,SQ4", help="faiss factory function param")
    parser.add_argument("--index_output_dir", type=str,default='index_files', help="index output dir")     
    parser.add_argument("--vector_output_dir", type=str,default='vector_files', help="vector output dir") 
    return parser.parse_args()

def building_index(args,vecs,block):
    logger.info("Building indexes...")
    if args.cuda:
        index=faisslib.StandardGpuFC(vecs,vecs.shape[1],args.fc_param)
    else:
        index=faisslib.Fc(vecs,vecs.shape[1],args.fc_param)
    logger.info("Indexes have been built, writing to disk...")
    faisslib.save(index,os.path.join(args.index_output_dir,args.fc_param.replace(',','_'))+f'_{block}.idx')  

def transfer2vector(args):
    
    check_point_dict=torch.load(args.saved_model+'/model.tar')
    
    VECDIM=check_point_dict['Query.kwargs']['vec_dim']
    
    sub_model=args.model_type.lower()
   
    if sub_model=='item':        
        model=models.ItemModel(
            points_total_num=check_point_dict['shared_kwargs']['points_total_num'],
            emb_dim=check_point_dict['shared_kwargs']['emb_dim'],
            item_param=check_point_dict['Item.kwargs']
            )
        if args.local_yield:
            ds=data_loader.InferFromLocal(args.input_file
            ,num_num=check_point_dict['Item.kwargs']['num_feat_num']
            ,cat_num=check_point_dict['Item.kwargs']['cat_feat_num']
            ,seq_num=check_point_dict['Item.kwargs']['seq_feat_num']
            )
            dl=torch.utils.data.DataLoader(ds,batch_size=args.batch_size,num_workers=0,shuffle=False,drop_last=False)
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
            ds=data_loader.InferFromLocal(args.input_file
            ,num_num=check_point_dict['Query.kwargs']['num_feat_num']
            ,cat_num=check_point_dict['Query.kwargs']['cat_feat_num']
            ,seq_num=check_point_dict['Query.kwargs']['seq_feat_num']
            )
            dl=torch.utils.data.DataLoader(ds,batch_size=args.batch_size,num_workers=0,shuffle=False,drop_last=False)
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
    logger.info('Transfer to vector ...')
    tags=[]
    vecs=[]
    cnt=0
    lv=0
    with torch.no_grad():
        for kv in dl:
            tag=kv['ids'].reshape(-1).numpy()
            if args.cuda:
                kv={k:v.cuda() for k,v in kv.items()}
            vec=model(kv['ids'],kv['num'],kv['cat'],kv['seq']).reshape(-1,VECDIM).cpu().numpy().astype('float32')
            vecs.append(vec)
            tags.append(tag)
            lv+=1
            if lv%100==0:
                lvb=args.batch_size*lv
                logger.info(f'{lvb} vectors have been transfered ...')
            if len(vecs)>1e8:
                vecs=np.vstack(vecs)
                tags=np.hstack(tag)
                if args.index :
                    building_index(args,vecs,cnt)
                with h5py.File(os.path.join(args.vector_output_dir,f'{sub_model}_{cnt}.vec'),'w') as f:
                    f.create_dataset(shape=tags.shape,dtype='int64',name='tags',compression='gzip',data=tags)
                    if not args.index:
                        f.create_dataset(shape=vecs.shape,dtype='float32',name='vectors',compression='gzip',data=vecs)                
                cnt+=1
                vecs=[]
                tags=[]
        if len(vecs)>0:
            vecs=np.vstack(vecs)
            tags=np.hstack(tags)
            if args.index :
                building_index(args,vecs,cnt)
            with h5py.File(os.path.join(args.vector_output_dir,f'{sub_model}_{cnt}.vec'),'w') as f:
                f.create_dataset(shape=tags.shape,dtype='int64',name='tags',compression='gzip',data=tags)
                if not args.index:
                    f.create_dataset(shape=vecs.shape,dtype='float32',name='vectors',compression='gzip',data=vecs)
        logger.info('Done, Congratulations!')

def main():
    args = parse_args()
    transfer2vector(args)


if __name__ == "__main__":
    main()
