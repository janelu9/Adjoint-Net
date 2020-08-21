#!/usr/bin/env python
# coding: utf-8
# @author Jane 2020.7.7

import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import numpy as np

VECDIM=128

class PointEmbding(nn.Module):
    def __init__(self,emb_size,is_sparse=False):
        super(PointEmbding, self).__init__()
        self.emb_size=emb_size
        self.Embeding=nn.Embedding(*emb_size,sparse=is_sparse,padding_idx=0)

    def forward(self,x):
        return self.Embeding(x)

class sinb(nn.Module):
    def __init__(self,alpha=1.0):
        super(sinb, self).__init__()
        self.alpha=alpha

    def forward(self,x):
        x=self.alpha*x
        x[x>pi/2]=pi/2
        x[x<-pi/2]=-pi/2
        return torch.sin(x)

class Single(nn.Module):    
    def __init__(self
                 ,num_feat_num=0
                 ,cat_feat_num=0
                 ,cat_total_num=0
                 ,seq_feat_num=0
                 ,V_dim=32
                 ,hidden_dim=128
                 ,index_emb=None
                 ,id_flag=False
                 ,max_len=20
                 ,vec_dim=VECDIM
                ):
        super(Single,self).__init__()
        self.num_feat_num=num_feat_num
        self.cat_feat_num=cat_feat_num
        self.seq_feat_num=seq_feat_num
        self.V_dim=V_dim
        self.hidden_dim=hidden_dim
        self.id_flag=id_flag
        self.max_len=max_len
        
        self.index_emb=index_emb  
        
        if self.num_feat_num>0:  
            self.V=nn.parameter.Parameter(torch.Tensor(num_feat_num,V_dim))
            
        if self.cat_feat_num>0:
            self.cat_emb = nn.Embedding(cat_total_num,V_dim)        
        
        if self.seq_feat_num>0:                      
            self.seq_operation=nn.ModuleList([nn.Linear(V_dim,V_dim) for _ in range(self.seq_feat_num)])
            
           
        self.fc1=nn.Linear(V_dim*(self.num_feat_num+self.cat_feat_num+self.seq_feat_num+self.id_flag),hidden_dim)
        self.fc2=nn.Linear(hidden_dim,V_dim)
              
        self.vec1=nn.Linear(V_dim+V_dim+hidden_dim,vec_dim*4)
        
        self.vec=nn.Linear(vec_dim*4,vec_dim)
        
        self.reset_param()
        
        self.kwargs=dict(
         num_feat_num=num_feat_num
        ,cat_feat_num=cat_feat_num
        ,cat_total_num=cat_total_num
        ,seq_feat_num=seq_feat_num
        ,hidden_dim=hidden_dim
        ,id_flag=id_flag
        ,max_len=max_len
        ,vec_dim=vec_dim
        )
        
    def reset_param(self):
        torch.nn.init.kaiming_uniform_(self.V,a=np.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.cat_emb.weight,a=np.sqrt(5))
        
    def forward(self,i,n,c,s):
        combine=[]        
        if self.seq_feat_num:
            seq_index=torch.split(s,self.max_len,-1)         
            combine.extend([
                torch.relu(self.seq_operation[i](F.dropout(self.index_emb(seq),0.1))).sum(2,keepdim=True).\
                div(torch.unsqueeze((seq>0).sum(2,keepdim=True),3).float()+1e-5)
                for i,seq in enumerate(seq_index)
            ])
            
        if self.id_flag:
            combine.append(self.index_emb(i))

        if self.cat_feat_num:
            combine.append(self.cat_emb(c))        
        
        if self.num_feat_num:
            combine.append(torch.unsqueeze(n,3).mul(self.V))
        
        features=torch.cat(combine,2)
        floor=torch.relu(features.sum(2))
        
        features=features.reshape((*features.shape[:2],-1))
        dnn_hidden=F.dropout(torch.relu(self.fc1(features)),0.1)
        deep=torch.relu(self.fc2(dnn_hidden))
        
        vec_hidden=torch.relu(self.vec1(torch.cat([floor,dnn_hidden,deep],2)))
        
        return torch.tanh(self.vec(vec_hidden))
    
class Adjoint(nn.Module):
    def __init__(self, points_total_num=0
                 ,emb_dim=32
                 ,query_param={'num_feat_num':145,'cat_feat_num':4,'seq_feat_num':4,'cat_total_num':35,'hidden_dim':128}
                 ,item_param={'num_feat_num':55,'cat_feat_num':6,'cat_total_num':21,'hidden_dim':64}
                ):
        super(Adjoint,self).__init__()
        self.Point=None
        if points_total_num>0:
            self.Point=PointEmbding([points_total_num,emb_dim])
        self.Query=Single(**query_param,V_dim=emb_dim,index_emb=self.Point)
        self.Item=Single(**item_param,V_dim=emb_dim,index_emb=self.Point)
        
        self.shared_kwargs=dict(points_total_num=points_total_num,emb_dim=emb_dim)
    
    def forward(self,kwargs):
        
        query=self.Query(kwargs['query_ids'],kwargs['query_num'],kwargs['query_cat'],kwargs['query_seq'])
        item=self.Item(kwargs['item_ids'],kwargs['item_num'],kwargs['item_cat'],kwargs['item_seq'])
        query=query.div(torch.norm(query,dim=2,keepdim=True))
        item=item.div(torch.norm(item,dim=2,keepdim=True))        
        return query.mul(item).sum(2)


class HingeLoss(nn.Module):
    def __init__(self,margin=0.05):
        super(HingeLoss,self).__init__()
        self.margin=margin
        #self.zero=torch.zeros((1,1)).cuda()

    def forward(self,pred):        
        pos=pred[:,0:1]
        negs=pred[:,1:]        
        loss=negs-pos+self.margin
        return ((loss>0)*loss).sum(1).mean()
    
class SigmoidLoss(nn.Module):
    def __init__(self,batch_size=32,negs_size=32):
        super(SigmoidLoss,self).__init__()
        self.sigmoid=nn.BCEWithLogitsLoss()
        self.target=torch.tensor([[1]+[0]*(negs_size-1)]*batch_size,dtype=torch.float32).view(-1).cuda()

    def forward(self,pred): 
        return self.sigmoid(pred.view(-1),self.target)
    
class SoftmaxLoss(nn.Module):
    def __init__(self,batch_size=32):
        super(SoftmaxLoss,self).__init__()
        self.cross_entropy=nn.CrossEntropyLoss()
        self.target=torch.tensor([0]*batch_size,dtype=torch.long).cuda()

    def forward(self,pred): 
        return self.cross_entropy(pred,self.target)
    

class ItemModel(nn.Module):
    def __init__(self,
                 points_total_num=0
                 ,emb_dim=32
                 ,item_param={'num_feat_num':55,'cat_feat_num':6,'cat_total_num':21,'hidden_dim':64}
                ):
        super(ItemModel,self).__init__()
        self.Point=None
        if points_total_num>0:
            self.Point=PointEmbding([points_total_num,emb_dim])
        self.Item=Single(**item_param,V_dim=emb_dim,index_emb=self.Point)
    
    def forward(self,i,n,c,s):
        item=self.Item(i,n,c,s)
        item=item.div(torch.norm(item,dim=2,keepdim=True))        
        return item

class QueryModel(nn.Module):
    def __init__(self,
                 points_total_num=1573
                 ,emb_dim=32
                 ,query_param={'num_feat_num':145,'cat_feat_num':4,'seq_feat_num':4,'cat_total_num':35,'hidden_dim':128}
                ):
        super(QueryModel,self).__init__()
        self.Point=None
        if points_total_num>0:
            self.Point=PointEmbding([points_total_num,emb_dim])
        self.Query=Single(**query_param,V_dim=emb_dim,index_emb=self.Point)
    
    def forward(self,i,n,c,s):
        query=self.Query(i,n,c,s)
        query=query.div(torch.norm(query,dim=2,keepdim=True))     
        return query
