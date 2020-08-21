# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:24:30 2020

@author: kfzx-lujian

Copyright (c) 2018 ICBC kfzx-lujian. All Rights Reserved

"""


import os
import time

import numpy as np
import argparse
import logging

import pandas as pd

import torch

import data_loader
import models 




logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("torch")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser("twins")
    parser.add_argument("--query_file", type=str, help="query file loaded to memery")
    parser.add_argument("--item_file", type=str, help="item file tuple: file1,file2,file3")
    parser.add_argument("--label_file", type=str, default=None, help="label file ")
    
    parser.add_argument("--test_query_file", type=str, default=None, help="test query")
    parser.add_argument("--test_label_file", type=str, default=None, help="test label file ")
    
    parser.add_argument("--query_num_num", type=int, default=0, help="num num of query")    
    parser.add_argument("--query_cat_num", type=int, default=0, help="cat num of query")
    parser.add_argument("--query_seq_num", type=int, default=0, help="seq num of query") 
    parser.add_argument("--query_cat_dim", type=int, default=1000, help="cat dim of query")
    
    parser.add_argument("--item_num_num", type=int, default=0, help="num num of item")    
    parser.add_argument("--item_cat_num", type=int, default=0, help="cat num of item")
    parser.add_argument("--item_seq_num", type=int, default=0, help="seq num of item") 
    parser.add_argument("--item_cat_dim", type=int, default=1000, help="cat dim of item") 
    
    parser.add_argument( "--epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument( "--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument( "--cuda", type=bool, default=True, help="if using cuda")
    parser.add_argument( "--lr", type=float, default=0.5, help="learning rate")
    parser.add_argument( "--lr_scheduler", type=int, default=20, help="learning rate decrey scheduler")
    parser.add_argument( "--model_output_dir", type=str, default='model_output', help="model output folder")
    parser.add_argument( "--neg_size", type=int, default=32, help="length of listwise")
    parser.add_argument( "--item_ids_flag", type=int, default=0, help="if embedding items index")
    parser.add_argument( "--query_hidden_dim", type=int, default=128, help="query hidden dim")
    parser.add_argument( "--item_hidden_dim", type=int, default=64, help="item hidden dim")
    parser.add_argument( "--vec_dim", type=int, default=32, help="Dimension of encoder output")
    parser.add_argument( "--embedding_dim", type=int, default=32, help="Default Dimension of Embedding")

    return parser.parse_args()


def start_train(args):
    
    if not args.label_file:
        ds=data_loader.MyIterDataset(args.query_file,args.item_file,neg_size=args.neg_size,item_ids_flag=args.item_ids_flag)
        dl=torch.utils.data.DataLoader(ds,batch_size=ds.batch_size,num_workers=len(ds.item_start_rows),drop_last=True)
    else:
        ds=data_loader.MyDataset(args.query_file,args.item_file,args.label_file
                                     ,query_num_num=args.query_num_num
                                     ,query_cat_num=args.query_cat_num
                                     ,query_seq_num=args.query_seq_num
                                     ,query_cat_dim=args.query_cat_dim
                                     ,item_num_num=args.item_num_num
                                     ,item_cat_num=args.item_cat_num
                                     ,item_seq_num=args.item_seq_num
                                     ,item_cat_dim=args.item_cat_dim                                    
                                     ,neg_size=args.neg_size
                                     ,item_ids_flag=args.item_ids_flag)
        dl=torch.utils.data.DataLoader(ds,batch_size=ds.batch_size,num_workers=2,shuffle=True)

    model=models.Adjoint(
        points_total_num=ds.item_ids_flag*ds.item_ids_dim,
        emb_dim=args.embedding_dim,
        query_param={'num_feat_num':ds.query_num_num,'cat_feat_num':ds.query_cat_num,'seq_feat_num':ds.query_seq_num,
                    'cat_total_num':ds.query_cat_dim,'hidden_dim':args.query_hidden_dim,'vec_dim':args.vec_dim},
        item_param={'num_feat_num':ds.item_num_num,'cat_feat_num':ds.item_cat_num,'seq_feat_num':ds.item_seq_num,
                    'cat_total_num':ds.item_cat_dim,'hidden_dim':args.item_hidden_dim,'vec_dim':args.vec_dim,'id_flag':ds.item_ids_flag}
               )
    logger.info(model)
    check_point_dict={'shared_kwargs':model.shared_kwargs,
                'Item.kwargs':model.Item.kwargs,
                'Query.kwargs':model.Query.kwargs
                     }
    
    if args.test_query_file and args.test_label_file:
        
        import faiss
        test_query_ds=data_loader.InferFromMemery(args.test_query_file
            ,num_num=check_point_dict['Query.kwargs']['num_feat_num']
            ,cat_num=check_point_dict['Query.kwargs']['cat_feat_num']
            ,seq_num=check_point_dict['Query.kwargs']['seq_feat_num']
            )
        
        test_query_dl=torch.utils.data.DataLoader(test_query_ds,batch_size=2000,num_workers=2,shuffle=False,drop_last=False)  

        item_ds=data_loader.InferFromMemery(args.item_file
            ,num_num=check_point_dict['Item.kwargs']['num_feat_num']
            ,cat_num=check_point_dict['Item.kwargs']['cat_feat_num']
            ,seq_num=check_point_dict['Item.kwargs']['seq_feat_num']
            )
        item_dl=torch.utils.data.DataLoader(item_ds,batch_size=2000,num_workers=2,shuffle=False,drop_last=False) 
        
        label_data=pd.read_csv(args.test_label_file,sep=chr(27),header=None,dtype='float32').astype('int64')
        label_cols=label_data.columns
        y=label_data.groupby(label_cols[1]).agg(lambda x :set(x.values)).sort_index().values
        
        
        query_model=models.QueryModel(
            points_total_num=check_point_dict['shared_kwargs']['points_total_num'],
            emb_dim=check_point_dict['shared_kwargs']['emb_dim'],
            query_param=check_point_dict['Query.kwargs']
            )                
        query_model.eval()
        
        item_model=models.ItemModel(
            points_total_num=check_point_dict['shared_kwargs']['points_total_num'],
            emb_dim=check_point_dict['shared_kwargs']['emb_dim'],
            item_param=check_point_dict['Item.kwargs']
            )
        item_model.eval()
        
        def copy_param(m,saved_param):
            model_param=m.state_dict()
            for k,v in model_param.items():
                assert k in saved_param,f'parameter: {k} not found '
                model_param[k]=saved_param[k]
            return model_param
        
        if args.cuda:
            query_model=query_model.cuda()
            item_model=item_model.cuda()

    if args.cuda:
        model=model.cuda()
        
    
    criterion=models.HingeLoss()
                  
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_scheduler, gamma=0.1)

    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    recall_init=0.0
    # learning rate
    for epoch in range(args.epochs):
        '''
        lr = args.lr * (0.1 ** (epoch // args.lr_scheduler))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        '''
        for batch,kwargs in enumerate(dl):
            optimizer.zero_grad()
            if args.cuda:
                kwargs={k:v.cuda() for k,v in kwargs.items()}
            output = model(kwargs)
            loss = criterion(output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            log_interval = 200
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                logger.info('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.6f} | ppl {:8.2f}'.format(
                        epoch, batch*args.batch_size, 1, scheduler.get_last_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, np.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
                
        scheduler.step()
                
        if args.test_query_file and args.test_label_file:
                
            if epoch % 2 ==0:
                logger.info('testing model ...')
                index=faiss.IndexFlatIP(args.vec_dim)            
                epoch_dict=model.state_dict()
                item_model.load_state_dict(copy_param(item_model,epoch_dict))            
                with torch.no_grad():
                    for kv in item_dl:
                        if args.cuda:
                            kv={k:v.cuda() for k,v in kv.items()}
                        p=item_model(kv['ids'],kv['num'],kv['cat'],kv['seq']).reshape(-1,args.vec_dim).cpu().numpy()
                        index.add(p)
                query_model.load_state_dict(copy_param(query_model,epoch_dict))
                y_=[]
                with torch.no_grad():
                    for kv in test_query_dl:
                        if args.cuda:
                            kv={k:v.cuda() for k,v in kv.items()}
                        u=query_model(kv['ids'],kv['num'],kv['cat'],kv['seq']).reshape(-1,args.vec_dim).cpu().numpy()
                        _,I=index.search(u,40)
                        y_.extend(list(I))
                assert len(y)==len(y_),"y's length does not equal to y_'s"
                ct=0
                s=0
                for i,j in zip(y,y_):
                    for k in i[0] :
                        s+=1
                        if k in j:
                            ct+=1 
                            
                recall=ct/(s+0.0001)
                logger.info(f"epoch:{epoch} ,top40 recall:{recall}")
                if recall>recall_init:
                    logger.info('saving model ...')
                    check_point_dict.update({'model_param':epoch_dict})
                    torch.save(check_point_dict,os.path.join(args.model_output_dir,'model.tar'))
                    recall_init=recall
                    
    if not (args.test_query_file and args.test_label_file):
        logger.info('saving model ...')
        check_point_dict.update({'model_param':model.state_dict()})
        torch.save(check_point_dict,os.path.join(args.model_output_dir,'model.tar'))


def main():
    args = parse_args()
    start_train(args)


if __name__ == "__main__":
    main()
