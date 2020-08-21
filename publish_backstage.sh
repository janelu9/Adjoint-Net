#!/bin/bash

export PATH="/root/anaconda3/bin:$PATH"
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64/

echo "gpu testing"	
python -u gpu_test.py && echo "gpu is ok" && touch gpu_test_done &

echo "oaas download"
source /approot1/start_common_util.sh
updateOaasApplicationProperty

if [ ! -f oaasAccess.jar ];then wget -r -nd -P ./ ${codeRepo}/common/oaasAccess.jar;fi

oaas="java -jar ${localCodePath}/oaasAccess.jar \
'{\"PROC_TYPE\":\"MODEL_DOWNLOAD\",\
\"APP_NAME\":\"${mlpOaasUser}\",\
\"DOMAIN\":\"${mlpOaasDomain}\",\
\"PASSWORD\":\"${mlpOaasPassword}\",\
\"CONTAINER_PATTERN\":\"0\",\
\"CONTAINER_ID\":\"\",\
\"MODEL_KEY\":\"${modelName%_*}\",\
\"FILE_PATH\":\"${localCodePath}\",\
\"FILE_EXTEND\":\"zip\"}'"

if [ $modelInOaas == 'Y' ];then 
	echo $oaas
	$oaas && yes|unzip ${modelName%_*}.zip &&echo "oaas is ok" &&touch oaas_download_done 
fi