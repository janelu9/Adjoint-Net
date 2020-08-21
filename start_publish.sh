#!/bin/bash

if [ -f gpu_test_done ];then rm gpu_test_done;fi
if [ -f oaas_download_done ];then rm oaas_download_done;fi	

chmod 777 publish_backstage.sh
nohup ./publish_backstage.sh >>$logdir/publish_backstage.log 2>&1 &

source ~/.bashrc

nohup gunicorn -c config.py server_new_:app >> $logdir/my_publish.log &
