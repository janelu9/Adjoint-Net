## 简介

自研`Adjoint Net`,适用于基于向量近邻进行排序的结构化数据场景, 如推荐算法。

需要环境：
```
torch==1.5.1
faiss==1.6.3
```

网络由两个结构类似的`single`网络组成，同时支持输入`numeric`、`category`以及`sequence`类型变量。两个`single`网络在末端输出相同维度的`vector`。通过自研的`HingeLoss`进行约束优化。
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
