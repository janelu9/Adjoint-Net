#!/bin/bash

:<<!
Created on Mon Jul 22 17:24:30 2020
@author: kfzx-lujian
!


source /opt/hadoopclient/bigdata_env
curl -o ~/user.keytab ${codeRepo}/common/jupyter/keytab/etlhadoop.keytab
kinit -k etlhadoop/hadoop -t /root/user.keytab
source /approot1/start_common_util.sh
updateOaasApplicationProperty

export PATH="/root/anaconda3/bin:$PATH"
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64/
echo `which python`
set -eux
#create a queue with $nproc processes to download data
function multiprocessing_hadoop_get(){
    hadoop fs -ls $2/00*|awk '{print $8}'>hadoop_file.list
    num=`cat hadoop_file.list|wc -l`
    echo "${num} hadoop files look like:"&&head -n3 hadoop_file.list
    if [ -d $3 ];then
        rm -rf $3/*
    else
        mkdir $3
    fi
    mkfifo /tmp/$$.fifo && exec 666<>/tmp/$$.fifo && rm -rf /tmp/$$.fifo
    for i in `seq 1 $1`;do
        echo 
    done  >&666
    cnt=0
    echo "Begine to download ..."
    st=$(date +%H%M%S);
    while read line;do
        read -u666 
        cnt=$((cnt+1))
        if [ $((cnt % 50)) = 0 ];then
            echo "$cnt/${num} files have been downloaded to $3 with $1 processes in `expr $(date +%H%M%S) - $st` seconds"
        fi
        { 
        hadoop fs -get $line $3/
        echo >&666 
        } >/dev/null 2>&1 &
    done < hadoop_file.list && echo "$cnt/${num} files have been downloaded to $3 with $1 processes in `expr $(date +%H%M%S) - $st` seconds"
    wait
    exec 666<&- && exec 666>&-
    rm hadoop_file.list
    if [ $4 = 1 ];then
        echo "All are downloaded, begine to union files ..."
        cat $3/*>$3.txt && rm -rf $3 
    fi
    echo "Done"    
}

function spark_download()
{   
    export HADOOP_FILE_NAME=$1
    export PARTITIONS=$2
    spark-submit \
        --master local[$3] \
        --conf "spark.pyspark.python=/root/anaconda3/bin/python" \
     spark_download.py
}

function param(){
    eval pValue=$(echo $trainParams | jq -r .$1)
    if [ -z "$pValue" ] || [ "$pValue" == "null" ] ; then
       eval pValue=$(echo $trainParamsDefault | jq -r .$1)
    fi
    echo $pValue
}
echo "gpu testing"
if [ -f gpu_test_done ];then rm gpu_test_done;fi    
python -u gpu_test.py && echo "gpu is ok" && touch gpu_test_done &

#download dataset
echo "spark downloading"
spark_download custpath,prodpath,detailpath 1 36 >spark.log 2>&1
echo "spark downloading done"


#echo "hadoop getting"
#custpath=$(param custpath)
#multiprocessing_hadoop_get 20 $custpath custpath 1
#prodpath=$(param prodpath)
#multiprocessing_hadoop_get 20 $prodpath prodpath 1
#detailpath=$(param detailpath)
#multiprocessing_hadoop_get 20 $detailpath detailpath 1
#echo "hadoop got"

partitions=10
if [ -f allcustpath_download_done ];then rm allcustpath_download_done;fi
spark_download allcustpath $partitions 32 >>spark.log 2>&1&&touch allcustpath_download_done &
#allcustpath=$(param allcustpath)
#multiprocessing_hadoop_get 10 $allcustpath allcustpath 0 >allcustpath_download.log 2>&1 &&touch allcustpath_download_done &

#TRAIN
echo "parse train param"
item_num_num=$(param cust_seq_num)
item_cat_num=$(param cust_categery_column_num)
item_cat_dim=$(($(param cust_categery_lastnum) + 1))

query_num_num=$(param prod_seq_num)
query_cat_num=$(param prod_categery_column_num)
query_cat_dim=$(($(param prod_categery_lastnum) + 1))

if [ ! -d saved_model ];then mkdir saved_model;else rm -rf saved_model/* ; fi
echo "train model"
export CUDA_VISIBLE_DEVICES=0
python -u train.py \
            --query_file prodpath.txt \
            --item_file custpath.txt \
            --label_file detailpath.txt \
            --query_num_num $query_num_num \
            --query_cat_num $query_cat_num \
            --query_cat_dim $query_cat_dim \
            --item_num_num $item_num_num \
            --item_cat_num $item_cat_num \
            --item_cat_dim $item_cat_dim \
            --query_hidden_dim 256 \
            --item_hidden_dim 128 \
            --embedding_dim 32 \
            --epoch 100 \
            --lr 0.5 \
            --lr_scheduler 30 \
            --model_output_dir saved_model >>train.log 2>&1
            
echo "train done"
while  [ ! -f allcustpath_download_done ] || [ ! -f gpu_test_done ] ;do sleep 3 ;done && rm allcustpath_download_done gpu_test_done
echo "allcustpath downloaded, gpu tested"


#building index
echo "bulid index"
if [ ! -d index ];then mkdir index;else rm -rf index/* ;fi
if [ ! -d vector ];then mkdir vector;else rm -rf vector/* ;fi
#echo "spliting allcust data to pieces for asyn data loading from local disk to accelerate the train speed"
#ls allcustpath/* >allcust.list
#allcust_num=`sed -n $= allcust.list`
#batch=$((allcust_num/3))
#tail -n$((allcust_num - batch)) allcust.list>allcust.list2
#cat `head -n $batch allcust.list`>allcust1
#cat `head -n $batch allcust.list2`>allcust2
#cat `tail -n $((allcust_num - batch -batch)) allcust.list2`>allcust3

python -u multiprocessing_index.py \
            --input_file "allcustpath,${partitions}" \
            --saved_model saved_model \
            --local_yield true \
            --batch_size 10000 \
            --fc_param 'IVF4096,SQ8' \
            --index_output_dir index \
            --vector_output_dir vector  >>train.log 2>&1

echo "done. begine to zip the files..."
zip -r ${modelName}.zip saved_model index vector

echo "pass model index... to OAAS"

if [ ! -f oaasAccess.jar ];then wget -r -nd -P ./ ${codeRepo}/common/oaasAccess.jar;fi

oaas="java -jar ${localCodePath}/oaasAccess.jar \
{\"PROC_TYPE\":\"MODEL_UPLOAD\",\
\"APP_NAME\":\"${mlpOaasUser}\",\
\"DOMAIN\":\"${mlpOaasDomain}\",\
\"PASSWORD\":\"${mlpOaasPassword}\",\
\"CONTAINER_PATTERN\":\"0\",\
\"CONTAINER_ID\":\"\",\
\"MODEL_NAME\":\"${modelName}.zip\",\
\"KEY_ALLOC_PATTERN\":\"1\",\
\"FILE_PATH\":\"${localCodePath}\",\
\"KEY_VALUE\":\"${modelName}\",\
\"FILE_EXTEND\":\"zip\"}"

echo $oaas>oaas.log
$oaas>>oaas.log

if [ $? -ne 0 ] ;then reportFail modelUploadFailed ;fi

echo "All Done, Congratulations!"
