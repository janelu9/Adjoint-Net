## 简介

自研`Adjoint Net`(伴随网络, 一开始基于paddle开发, 因框架存在bug导致gpu下不收敛,遂转向更为灵活成熟的torch),适用于基于向量近邻进行排序的结构化数据场景, 如推荐算法。

网络由两个结构类似的`single`网络组成，同时支持输入`numeric`、`category`以及`sequence`类型变量。两个`single`网络在末端输出相同维度的`vector`。通过自研的`List-HingeLoss`进行约束优化。
该loss相比`logisticLoss`和YouTube的`crossEntropyLoss`分别有10%和4%左右的召回提升。

网络基于torch==1.5.1和faiss实现，支持LocalData和MemoryData两种训练模式，分别适用于不进入内存和进入内存的两种训练模式。均支持多线程加速。

在理财推荐场景中top40召回率在97.51%左右 相比传统协同过滤有4%以上提升

```
2020-07-15 03:25:16,082 - INFO - | epoch  34 | 25600/    1 batches | lr 0.05 | ms/batch 12.45 | loss 0.011075 | ppl     1.01
2020-07-15 03:25:18,600 - INFO - | epoch  34 | 32000/    1 batches | lr 0.05 | ms/batch 12.59 | loss 0.010102 | ppl     1.01
2020-07-15 03:25:21,070 - INFO - | epoch  34 | 38400/    1 batches | lr 0.05 | ms/batch 12.35 | loss 0.010212 | ppl     1.01
2020-07-15 03:25:23,435 - INFO - | epoch  34 | 44800/    1 batches | lr 0.05 | ms/batch 11.82 | loss 0.009832 | ppl     1.01
2020-07-15 03:25:25,860 - INFO - | epoch  34 | 51200/    1 batches | lr 0.05 | ms/batch 12.12 | loss 0.010099 | ppl     1.01
2020-07-15 03:25:28,305 - INFO - | epoch  34 | 57600/    1 batches | lr 0.05 | ms/batch 12.22 | loss 0.009295 | ppl     1.01
2020-07-15 03:25:30,739 - INFO - | epoch  34 | 64000/    1 batches | lr 0.05 | ms/batch 12.17 | loss 0.009621 | ppl     1.01
2020-07-15 03:25:33,028 - INFO - testing model ...
2020-07-15 03:25:33,893 - INFO - epoch:34 ,top40 recall:0.9751015046572725
2020-07-15 03:25:33,893 - INFO - saving model ...
2020-07-15 03:25:36,538 - INFO - | epoch  35 |  6400/    1 batches | lr 0.05 | ms/batch 28.99 | loss 0.020167 | ppl     1.02
2020-07-15 03:25:38,985 - INFO - | epoch  35 | 12800/    1 batches | lr 0.05 | ms/batch 12.23 | loss 0.010723 | ppl     1.01
```

query和item数据要存储为如下格式：

```
0	0.333333	0.386980	0.000000	0.976489	0.860645	0.964211	0.881172	0.062500	0.100	0.0	...	0.395131	0.461981	0.598012	0.450046	1	4	10	14	16	19
1	0.333333	0.511253	0.000000	0.976489	0.860645	0.964211	0.881172	0.062500	0.100	0.0	...	0.560406	0.581618	0.557519	0.545222	1	4	10	14	16	20
2	0.000000	0.000000	0.000000	1.000000	1.000000	1.000000	1.000000	0.209375	0.220	0.0	...	0.000000	0.000000	0.000000	0.000000	2	5	9	14	16	18
3	0.666667	0.000000	0.000000	1.000000	1.000000	0.913719	0.896514	0.212500	0.325	0.0	...	0.000000	0.000000	0.000000	0.000000	2	5	10	14	16	18
4	0.666667	0.571232	0.057612	0.976489	0.860645	0.913719	0.896514	0.440625	0.305	0.0	...	0.499352	0.533581	0.457819	0.453905	1	4	10	14	16	18
```

第一列为index编号, 前面浮点数为数值型特征, 后面整数型为分类型变量, 实际上也可以支持序列变量。

label存储为如下格式：
```
66317	223
66318	223
66319	223
66320	223
66321	223
```

第一列为query的index，第二列为item的index

执行python train.py -h 可查看更多参数。


## 训练

```
export CUDA_VISIBLE_DEVICES=0
python -u train.py \
            --query_file prodpath.txt \
            --item_file custpath.txt \
            --label_file detailpath.txt \
            --test_query_file test_queryss.csv \
            --test_label_file test_labelss.csv \
            --query_num_num $query_num_num \
            --query_cat_num $query_cat_num \
            --query_cat_dim $query_cat_dim \
            --item_num_num $item_num_num \
            --item_cat_num $item_cat_num \
            --item_cat_dim $item_cat_dim \
            --query_hidden_dim 128 \
            --item_hidden_dim 64 \
            --embedding_dim 32 \
            --epoch 100 \
            --lr 0.5 \
            --lr_scheduler 30 \
            --model_output_dir saved_model
```

## 多进程批量预估和建索引

```
python -u multiprocessing_index.py \
            --input_file "allcustpath,${partitions}" \
            --saved_model saved_model \
            --local_yield true \
            --batch_size 10000 \
            --fc_param 'IVF4096,SQ8' \
            --index_output_dir index \
            --vector_output_dir vector
```
> 基本上8并发下10分钟内可完成对50000000量级item的transfer和索引建立

## 生产部署

将代码打包一式两份分别命名为：
```
    trainCode.zip
    predictCode.zip
```

再和
```
    start_train.sh
    start_publish.sh
```

一起放在`122.20.141.10:/approot1/washome/HTTPServer/htdocs/MLPDL/REC/ADJOINT_NET/ `下

### 自动化训练

修改train_json为你的输入,运行如下命令启动自动化训练.

```
train_json='{"filemode":"etlhadoop","modelId":"ADJOINT_NET","trainDesc":"model train","app":"REC","aglorithm":"ADJOINT_NET","trainParams":"kinitUser#=#etlhadoop;custpath#=#/user/hive/warehouse/pc8.db/kli_pc8_1_recall_sample_data_d/pt_dt=01;allcustpath#=#/user/hive/warehouse/pc8.db/kli_pc8_1_recall_sample_data_d/pt_dt=02;prodpath#=#/user/hive/warehouse/pc8.db/kli_pc8_1_recall_prod_data_d/pt_dt=01;detailpath#=#/user/hive/warehouse/pc8.db/kli_pc8_1_recall_detail_data_d/pt_dt=01;custpath_0#=#cust_rownum;custpath_1#=#cust_id;prodpath_0#=#prod_rownum;prodpath_1#=#prod_code;detailpath_0#=#cust_rownum;detailpath_1#=#prod_rownum;cust_seq_num#=#75;cust_categery_column_num#=#51;cust_categery_lastnum#=#313;prod_seq_num#=#36;prod_categery_column_num#=#7;prod_categery_lastnum#=#41;label_1_num#=#47839"}'
train_url=http://122.19.13.145:8883/icbc/dds/mlp_model_train/jobcontrol/task/createDLTrainTask
curl -H "Content-Type:application/json" -X POST -d  "${train_json}" ${train_url}
```


### 实时预估

修改predict_json为你的输入,运行如下命令启动实时预估服务.

```
predict_json='{"app":"REC","aglorithm":"ADJOINT_NET","modelId":"ADJOINT_NET_SERVING","publishVersion":"1","topic":"tf"}'
predict_url=http://122.19.13.145:8883/icbc/dds/mlp_model_publish/publish/publish/createtask
predict_result=$(curl -H "Content-Type:application/json" -X POST -d "${predict_json}" ${predict_url})
```

## 服务体验

因为索引文件较大，需要长时间从oaas拉取，这里采取了自研的加载文件和挂起服务端口分离策略。先挂起预估服务，后台补充模型参数和索引文件。
也就是说容器启动后服务会迅速挂起。如果调用的话可能会返回

`
{'Items': None, 'errorMsg': "Initialization hasn't been completed, wait for minutes.", 'success': '1'}
`

直至模型参数和索引文件以及标签数据加载完毕。服务可正常返回,模型计算时效在3ms左右。

`{'errorMsg': None, 'items': ['020000237920898', '100100183726139', '390100001443838', '120200014676854', '240200017027159', '260300017586187', '200200015055442', '060300012810957', '020000067453946', '020000211740410'], 'success': '0'}`

> 注: 这样做主要是为了避免容器dds监控超时自毁。 

服务端口如：

[**http://122.19.13.145:8883/icbc/dds/dds_mlp_dl_rec_adjoint_net_serving/query**](url)

请求格式如：

`{"query": [1.0, 0.001, 0.40237, 0.47912, 1.095, 0.9769200000000001, 1.9508, 1.9674, 1.16737, -0.59511, 0.4014, 0.63088, 0.87108, 0.05, 1.0, 0.005, 0.05, 0.0, 0.01, 0.37143000000000004, 0.375, 0.0, 1.0, 1.0, 0.0, 1.0, 1.21781, 1.00485, 0.0, 0.0, 0.42105, 0.13417, 0.14062, 0.13013, 0.17001, 1.0, 3.0, 15.0, 20.0, 22.0, 25.0, 28.0, 36.0], "topk": 10}`