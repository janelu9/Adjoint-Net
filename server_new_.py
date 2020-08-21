import time
from dds_abstract_metrics_ import AbstractMetrics
import logging
from flask import request,Flask
import json
app = Flask(__name__)
import redis
import os
from server import query,args
#from webserverForTF import recotaskstd

abstractMetrics = AbstractMetrics()
metricsKey=''
start_time=''
isSuccess=False
currentConcurrent=0
pid=str(os.getpid())

myredis=redis.Redis(host="localhost",port=6379)

'''
根据metricsKey获取当前的并发数统计
    * @param metricsKey
    *            当前统计所使用的key

'''
def incrementConcurrent(metricsKey): 
  global myredis
  if not myredis.exists(metricsKey):
    logging.info("======key not exist:"+metricsKey)
    myredis.set(metricsKey,1)
    concurrents=1
  else:
    #myredis.incr(metricsKey)
    concurrents=myredis.incr(metricsKey)
  print(str(pid)+" inc:"+str(concurrents))
  logging.info(str(pid)+" inc:"+str(concurrents))
  #logging.info("inc concurrents "+metricsKey+","+str(concurrents))
  return concurrents
def decrementConcurrent(metricsKey):
  global myredis
  concurrents=myredis.decr(metricsKey)
  print(str(pid)+" dec:"+str(concurrents))
  logging.info(str(pid)+" dec:"+str(concurrents))
  #logging.info("dec concurrents "+metricsKey+","+str(concurrents))
  return concurrents


@app.before_request
def before_request_predict():
    global metricsKey
    global start_time
    global currentConcurrent
    path=request.path
    clientIp=request.remote_addr
    logging.info("in before")
    logging.info(str(request.path))
    logging.info(clientIp)
    
    metricsKey = path + "@@" + clientIp
    start_time = time.time()
    #currentConcurrent=abstractMetrics.incrementConcurrent(metricsKey)
    currentConcurrent=incrementConcurrent(metricsKey)

@app.after_request
def after_request_predict(response):
    global isSuccess
    status = response.status_code
    logging.info("response status code:"+str(status))
    if  status >= 200 and status < 300 :
        isSuccess = True
    else :
        isSuccess = False
    return response

@app.teardown_request
def tear_down_predict(e):
    global metricsKey
    global start_time
    global isSuccess
    global currentConcurrent
    logging.info("in tear_down_predict:before collect,metricskey :"+metricsKey)
    #abstractMetrics.decrementConcurrent(metricsKey)
    decrementConcurrent(metricsKey) 
    abstractMetrics.collect(metricsKey, start_time, currentConcurrent, isSuccess)
    
    logging.info("in tear_down_predict:finish")


@app.route("/test", methods=['GET'])
def test():
    result = {'code':-1, 'result':'fail'}
    time.sleep(3)
    if request.method == 'GET':
        result = {'code':0, 'result':'testsuccess'}
        #result['data'] = data
    
    json_str = json.dumps(result)
    return json_str
    
@app.route("/query", methods=['POST'])
def predict_new():
    '''
    result = {'code':-1, 'result':'fail'}
    if request.method == 'POST':
        time.sleep(1)
        result = {'code':0, 'result':'success'}
        #result['data'] = data
    
    json_str = json.dumps(result)
    return json_str
    '''
    return query(request)

#BAPP
@app.route("/recotaskstd", methods=['POST','GET'])
def recotaskstd_new():
    return recotaskstd(request)

if __name__ == "__main__":

  app.run(host='0.0.0.0', port=args.port, debug=True, threaded=False)
  #app.run()