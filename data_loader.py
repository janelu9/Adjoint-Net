# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:47:54 2020

@author: kfzx-lujian
"""


import torch
import csv
from itertools import chain
import pandas as pd
import numpy as np
import random

class MyIterDataset(torch.utils.data.IterableDataset):
    '''
    MyIterDataset
    param: 
        query_data_path: str
        item_data_path : list[str,]
    '''    
    def __init__(self,query_data_path,item_data_path,neg_size=32,max_len=20,item_ids_flag=0):
        super(MyIterDataset).__init__()
        
        self.query_data_path=query_data_path
        self.item_data_path=item_data_path.split(',') 
        self.batch_size=neg_size
        
        self.query_ids_flag=0
        self.item_ids_flag=item_ids_flag
        
        self.max_len=max_len        
        self._paddingSeq=lambda x:list(chain(*[i.split(',')[:max_len]+[-1]*max(0,max_len-1-i.count(','))
                                              if isinstance(i,str) and i!='' else [-1]*max_len for i in x]))      
        self._parse_query_data()
       
        self._parse_item_feature_param()
        self.item_start_rows=[0]+[self._parse_item_start_row(i) for i in self.item_data_path[1:]]
        
    def _parse_query_data(self):
        query_data=pd.read_csv(self.query_data_path,sep=chr(27),header=0)
        query_cols=query_data.columns
        query_cols_length=len(query_cols)
        num,cat,seq=[],[],[]
        for idx,col in enumerate(query_cols):
            col0=col[0].lower()
            if col0=='n':
                num.append(idx)
            elif col0=='c':
                cat.append(idx)
            elif col0=='s':
                seq.append(idx)
            elif col0=='i':
                self.query_ids_flag=1
                
        self.query_num_num=len(num)
        self.query_cat_num=len(cat)
        self.query_seq_num=len(seq)
        self.query_cat_dim=int(query_cols[cat[-1]].split('_')[1]) if cat else 0
        
        self.query_num=np.array(query_data.iloc[:,num],dtype='float32')       
        self.query_cat=np.array(query_data.iloc[:,cat],'float32').astype('int64')          
        self.query_seq=np.array([self._paddingSeq(i) for i in query_data.iloc[:,seq].values],'float32').astype('int64')+1
            
        self.query_ids=np.array(query_data.index)[:,np.newaxis]+1
        self.query_ids_dim=len(self.query_ids)
        if self.query_ids_flag==0:
            self.query_ids=self.query_ids[:,0:0]
        
    def _parse_item_feature_param(self):
        with open(self.item_data_path[0]) as f:
            rows=csv.reader(f,delimiter=chr(27))
            item_cols=next(rows)
        num,cat,seq=[],[],[]                
        for idx,col in enumerate(item_cols):
            col0=col[0].lower()
            if col0=='n':
                num.append(idx)
            elif col0=='c':
                cat.append(idx)
            elif col0=='s':
                seq.append(idx)
            elif col0=='i':
                self.item_ids_flag=1
                
        self.item_num_num=len(num)
        self.item_cat_num=len(cat)
        self.item_seq_num=len(seq)
        self.item_cat_dim=int(item_cols[cat[-1]].split('_')[1]) if cat else 0
        
        self.id_pivot=1+self.item_ids_flag
        self.num_pivot=self.id_pivot+self.item_num_num
        self.cat_pivot=self.num_pivot+self.item_cat_num

        

    def _parse_item_start_row(self,datadir):
        with open(datadir) as f:
            rows=csv.reader(f,delimiter=chr(27))
            item_cols=next(rows)
            start_id=next(rows)[0]
            cnt=1
            for row in rows:
                if row[0]==start_id:
                    cnt+=1
                else:
                    return cnt%self.batch_size
    
    def _yield_items(self,data_path,start_row):
        with open(data_path) as f:
            rows=csv.reader(f,delimiter=chr(27))
            next(rows)
            for i in range(start_row):
                next(rows)
            item_ids,item_num,item_cat,item_seq=[],[],[],[]
            item_size=0
            for row in rows:
                item_ids.append(row[1:self.id_pivot])
                item_num.append(row[self.id_pivot:self.num_pivot])
                item_cat.append(row[self.num_pivot:self.cat_pivot])
                if self.item_seq_num>0:
                    item_seq.append(self._paddingSeq(row[self.cat_pivot:]))   
                item_size+=1
                if item_size==self.batch_size:
                    query=[int(float(row[0]))]
                    feed={
                        'item_ids':np.array(item_ids,dtype='float32').astype('int64')+1
                        ,'item_num':np.array(item_num,dtype='float32')
                        ,'item_cat':np.array(item_cat,dtype='float32').astype('int64')
                        ,'item_seq':np.array(item_seq,dtype='float32').astype('int64')+1
                        ,'query_ids':self.query_ids[query]
                        ,'query_num':self.query_num[query]
                        ,'query_cat':self.query_cat[query]
                        ,'query_seq':self.query_seq[query]
                          }
                    item_ids,item_num,item_cat,item_seq=[],[],[],[]
                    item_size=0
                    yield feed
        
    def __iter__(self):
        work_info=torch.utils.data.get_worker_info()
        if work_info is None:
            data_dir=self.item_data_path[0]
            start_id=0
        else:           
            data_dir=self.item_data_path[work_info.id]
            start_id=self.item_start_rows[work_info.id]
        return iter(self._yield_items(data_dir,start_id))  
    
    


class MyDataset(torch.utils.data.Dataset):
    '''
    MyIterDataset
    param: 
        query_data_path: str
        item_data_path : str,..
    '''    
    def __init__(self,query_data_path,item_data_path,label_data_path,
                 query_num_num=0,
                 query_cat_num=0,
                 query_seq_num=0,
                 query_cat_dim=1000,
                 item_num_num=0,
                 item_cat_num=0,
                 item_seq_num=0,
                 item_cat_dim=1000,
                 neg_size=32,
                 max_len=20,
                 item_ids_flag=0,
                 sample_type=0
                ):
        super(MyDataset).__init__()
        
        self.query_data_path=query_data_path
        self.item_data_path=item_data_path
        self.label_data_path=label_data_path
        self.batch_size=neg_size
        
        self.query_num_num=query_num_num
        self.query_cat_num=query_cat_num
        self.query_seq_num=query_seq_num
        self.query_cat_dim=query_cat_dim
        
        self.item_num_num=item_num_num
        self.item_cat_num=item_cat_num
        self.item_seq_num=item_seq_num
        self.item_cat_dim=item_cat_dim        
        
        self.query_ids_flag=0
        self.item_ids_flag=item_ids_flag
        self.sample_type=sample_type
        
        self.max_len=max_len
        
        self._paddingSeq=lambda x:list(chain(*[i.split(',')[:max_len]+[-1]*max(0,max_len-1-i.count(','))
                                              if isinstance(i,str) and i!='' else [-1]*max_len for i in x]))      
        self._parse_query_data()
        self._parse_item_data()
        self._parse_label_data()
        self._getitem=self._sample_type0 if self.sample_type==0 else self._sample_type1
            
        
        
    def _parse_query_data(self):
        query_data=pd.read_csv(self.query_data_path,sep=chr(27),header=None)
        query_data.set_index(0,inplace=True)
        query_data.sort_index(inplace=True)
        
        self.query_ids=np.array(query_data.index)[:,np.newaxis]+1
        self.query_ids_dim=len(self.query_ids)
        assert self.query_ids_dim==self.query_ids[-1,0],"query's row number is not unique"
        if self.query_ids_flag==0:
            self.query_ids=self.query_ids[:,0:0]
            
        id_pivot= 1
        num_pivot=id_pivot+self.query_num_num
        cat_pivot=num_pivot+self.query_cat_num
        
        self.query_num=np.array(query_data.iloc[:,id_pivot:num_pivot],dtype='float32')       
        self.query_cat=np.array(query_data.iloc[:,num_pivot:cat_pivot],'float32').astype('int64')          
        self.query_seq=np.array([self._paddingSeq(i) for i in query_data.iloc[:,cat_pivot:].values],'float32').astype('int64')+1
            
            
    def _parse_item_data(self):
        item_data=pd.read_csv(self.item_data_path,sep=chr(27),header=None)
        item_data.set_index(0,inplace=True)
        item_data.sort_index(inplace=True)
        
        self.item_ids=np.array(item_data.index)[:,np.newaxis]+1
        self.item_ids_dim=len(self.item_ids)
        assert self.item_ids_dim==self.item_ids[-1,0],"item's row number is not unique"
        if self.item_ids_flag==0:
            self.item_ids=self.item_ids[:,0:0]
        
        id_pivot=1
        num_pivot=id_pivot+self.item_num_num
        cat_pivot=num_pivot+self.item_cat_num

        self.item_num=np.array(item_data.iloc[:,id_pivot:num_pivot],dtype='float32')       
        self.item_cat=np.array(item_data.iloc[:,num_pivot:cat_pivot],'float32').astype('int64')          
        self.item_seq=np.array([self._paddingSeq(i) for i in item_data.iloc[:,cat_pivot:].values],'float32').astype('int64')+1
            
                
    def _parse_label_data(self):
        label_data=pd.read_csv(self.label_data_path,sep=chr(27),header=None,dtype='float32').astype('int64')
        self.label_data=label_data.values
        self.data_length=len(self.label_data)
        
        label_cols=label_data.columns
        label_group=label_data.groupby(label_cols[1]).agg(lambda x :set(x.values))
        assert len(label_group)<=self.query_ids_dim,"label's query not in query_data"
        buy_set=set(self.label_data[:,0])
        buy_set_len=len(buy_set)
        #self.sample_num=min(int(buy_set_len/self.item_ids_dim*(self.batch_size-1)),1)
        self.sample_num=min(int(0.5*(self.batch_size-1)),1)
        
        if self.sample_type==0:
            def f(x):
                len_x=len(x)
                negs=list(buy_set-x)
                num=buy_set_len-len_x
                total_num=len_x*60
                if num<=total_num:return negs
                return random.sample(negs,total_num)

            self.buy_negs_dict=label_group.applymap(f).to_dict()[0]
            self.buy_negs_len_dict={k:len(v) for k,v in self.buy_negs_dict.items()}
        else:
            self.buy_list=list(buy_set)
            self.buy_poss_dict=label_group.to_dict()[0]
                       
        self.no_buy_list=list(set(range(self.item_ids_dim))-buy_set)
        
        
    def __len__(self,):
        return self.data_length
    
    def _sample_type0(self,index):
        item_id,query_id=self.label_data[index,:]
        
        sample_num=min(self.sample_num,self.buy_negs_len_dict[query_id])        
        sample_items_id1=random.sample(self.buy_negs_dict[query_id],sample_num)
        sample_items_id2=random.sample(self.no_buy_list,self.batch_size-sample_num-1)
        item=[item_id]
        item.extend(sample_items_id1)
        item.extend(sample_items_id2)
        query=[query_id]       
        feed={
             'item_ids':self.item_ids[item]
            ,'item_num':self.item_num[item]
            ,'item_cat':self.item_cat[item]
            ,'item_seq':self.item_seq[item]
            ,'query_ids':self.query_ids[query]
            ,'query_num':self.query_num[query]
            ,'query_cat':self.query_cat[query]
            ,'query_seq':self.query_seq[query]
              }
        return feed
    
    def _sample_type1(self,index):
        item_id,query_id=self.label_data[index,:]
        
        sample_items_id_step1=random.choices(self.buy_list,k=(self.batch_size-1)*10)
        sample_items_id_step2=list(set(sample_items_id_step1)-self.buy_poss_dict[query_id])
        sample_items_id_step3_num=min(self.sample_num,len(sample_items_id_step2))
        
        sample_items_id1=random.sample(sample_items_id_step2,sample_items_id_step3_num)
        sample_items_id2=random.sample(self.no_buy_list,self.batch_size-sample_items_id_step3_num-1)
        item=[item_id]
        item.extend(sample_items_id1)
        item.extend(sample_items_id2)
        query=[query_id]       
        feed={
             'item_ids':self.item_ids[item]
            ,'item_num':self.item_num[item]
            ,'item_cat':self.item_cat[item]
            ,'item_seq':self.item_seq[item]
            ,'query_ids':self.query_ids[query]
            ,'query_num':self.query_num[query]
            ,'query_cat':self.query_cat[query]
            ,'query_seq':self.query_seq[query]
              }
        return feed
        
    def __getitem__(self,index):        
        return self._getitem(index)
    
    
class InferFromLocal(torch.utils.data.IterableDataset):
    '''
    param: 
        query_data_path: str
    '''    
    def __init__(self,data_path,
                 num_num=0,
                 cat_num=0,
                 seq_num=0,                 
                 max_len=20,
                 batch_size=32):
        super(InferFromLocal).__init__()
        
        self.data_path=data_path.strip().split(',')
        self.batch_size=batch_size
        
        self.num_num=num_num
        self.cat_num=cat_num
        self.seq_num=seq_num
        
        self.ids_flag=0
        
        self.max_len=max_len        
        self._paddingSeq=lambda x:list(chain(*[i.split(',')[:max_len]+[-1]*max(0,max_len-1-i.count(',')) if isinstance(i,str) and i!='' else [-1]*max_len for i in x])) 
        
        self._parse_feature_param()
        
    def _parse_feature_param(self):
        
        self.id_pivot=2
        self.num_pivot=self.id_pivot+self.num_num
        self.cat_pivot=self.num_pivot+self.cat_num

    def _yield(self,data_path):
        seq=[]
        with open(data_path) as f:
            rows=csv.reader(f,delimiter=chr(27))
            for row in rows:
                ids=[row[1:self.id_pivot]]
                num=[row[self.id_pivot:self.num_pivot]]
                cat=[row[self.num_pivot:self.cat_pivot]]
                if self.seq_num>0:
                    seq=[self._paddingSeq(row[self.cat_pivot:])]  
                feed={
                    'ids':np.array(ids,dtype='float32').astype('int64')
                    ,'num':np.array(num,dtype='float32')
                    ,'cat':np.array(cat,dtype='float32').astype('int64')
                    ,'seq':np.array(seq,dtype='float32').astype('int64')+1
                      }
                yield feed
        
    def __iter__(self):
        work_info=torch.utils.data.get_worker_info()
        if work_info is None:
            data_dir=self.data_path[0]
        else:           
            data_dir=self.data_path[work_info.id]
        return iter(self._yield(data_dir)) 
        
        
class InferFromMemery(torch.utils.data.Dataset):
    '''
    param: 
        query_data_path: str
    '''    
    def __init__(self,data_path,
                 num_num=0,
                 cat_num=0,
                 seq_num=0,
                 max_len=20):
        super(InferFromMemery).__init__()
        
        self.data_path=data_path        
        self.max_len=max_len

        self.num_num=num_num
        self.cat_num=cat_num
        self.seq_num=seq_num        
        
        self._paddingSeq=lambda x:list(chain(*[i.split(',')[:max_len]+[-1]*max(0,max_len-1-i.count(','))
                                              if isinstance(i,str) and i!='' else [-1]*max_len for i in x]))             
        self._parse_data()
        
    def _parse_data(self):
        data=pd.read_csv(self.data_path,sep=chr(27),header=None,index_col=0)
        data.sort_index(inplace=True)
        
        self.ids=np.array(data.index)[:,np.newaxis]
        self.ids_dim=len(self.ids)
        #assert self.ids_dim==self.ids[-1,0],"row number is not unique"
        
        id_pivot=1
        num_pivot=id_pivot+self.num_num
        cat_pivot=num_pivot+self.cat_num

        self.num=np.array(data.iloc[:,id_pivot:num_pivot],dtype='float32')       
        self.cat=np.array(data.iloc[:,num_pivot:cat_pivot],'float32').astype('int64')          
        self.seq=np.array([self._paddingSeq(i) for i in data.iloc[:,cat_pivot:].values],'float32').astype('int64')+1

    def __len__(self,):
        return self.ids_dim
    
    def __getitem__(self,index):
        index=[index]    
        feed={
        'ids':self.ids[index]
        ,'num':self.num[index]
        ,'cat':self.cat[index]
        ,'seq':self.seq[index]
          }
        return feed

class MemoryInferFromLocal(object):
    '''
    param: 
        query_data_path: str
    '''    
    def __init__(self,data_path,
                 num_num=0,
                 cat_num=0,
                 seq_num=0,                 
                 max_len=20,
                 batch_size=32):
        super(MemoryInferFromLocal).__init__()
        
        self.data_path=data_path.strip()
        self.batch_size=batch_size
        
        self.num_num=num_num
        self.cat_num=cat_num
        self.seq_num=seq_num
        
        self.ids_flag=0
        
        self.max_len=max_len        
        self._paddingSeq=lambda x:list(chain(*[i.split(',')[:max_len]+[-1]*max(0,max_len-1-i.count(',')) if isinstance(i,str) and i!='' else [-1]*max_len for i in x])) 
        
        self._parse_feature_param()
        
    def _parse_feature_param(self):
        
        self.id_pivot=2
        self.num_pivot=self.id_pivot+self.num_num
        self.cat_pivot=self.num_pivot+self.cat_num

    def _yield(self,data_path):
        cnt=0
        tag,ids,num,cat,seq=[],[],[],[],[]
        with open(data_path) as f:
            rows=csv.reader(f,delimiter=chr(27))
            for row in rows:
                if self.ids_flag>0:
                    ids.append([row[0:1]])                
                tag.extend(row[1:self.id_pivot])
                num.append([row[self.id_pivot:self.num_pivot]])
                cat.append([row[self.num_pivot:self.cat_pivot]])
                if self.seq_num>0:
                    seq.append([self._paddingSeq(row[self.cat_pivot:])])
                cnt+=1
                if cnt==self.batch_size:
                    feed={
                     'ids':torch.from_numpy(np.array(ids,dtype='float32').astype('int64'))
                    ,'num':torch.from_numpy(np.array(num,dtype='float32'))
                    ,'cat':torch.from_numpy(np.array(cat,dtype='float32').astype('int64'))
                    ,'seq':torch.from_numpy(np.array(seq,dtype='float32').astype('int64')+1)
                      }
                    yield tag,feed
                    tag,ids,num,cat,seq=[],[],[],[],[]
                    cnt=0
            if cnt>0:
                    feed={
                     'ids':torch.from_numpy(np.array(ids,dtype='float32').astype('int64'))
                    ,'num':torch.from_numpy(np.array(num,dtype='float32'))
                    ,'cat':torch.from_numpy(np.array(cat,dtype='float32').astype('int64'))
                    ,'seq':torch.from_numpy(np.array(seq,dtype='float32').astype('int64')+1)
                      }
                    yield tag,feed
            
        
    def __iter__(self):
        return iter(self._yield(self.data_path)) 