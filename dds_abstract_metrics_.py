from multiprocessing import Pool,Lock,Value,Manager
import time
#import logging
import threading
import copy
import urllib
import requests
import os
import redis

from logging.handlers import TimedRotatingFileHandler,RotatingFileHandler
import logging

class mylogger(logging.Logger):
    """
    mylogger
    """
    def __init__(self, name='log', level=logging.INFO, time = False,fmt=None):
        self.level = level
        self.time = time
        env = os.environ
        self.LOG_PATH = env.get('logdir')
        if not os.path.exists(self.LOG_PATH):
            os.makedirs(self.LOG_PATH)
        self.file_name = os.path.join(self.LOG_PATH, '{name}.log'.format(name=name))
        
        self.fmt= '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s' if fmt is None else fmt
 
        super(mylogger, self).__init__(name=name, level=level)        
        self.__setFileHandler__()                
                
    def __setFileHandler__(self, level=None):
        file_handler = TimedRotatingFileHandler(filename=self.file_name, when='D', interval=7, backupCount=5) if self.time\
                        else RotatingFileHandler(self.file_name, maxBytes=1024*1024*100, backupCount=5)
        if not level:
            file_handler.setLevel(self.level)
        else:
            file_handler.setLevel(level)
        formatter = logging.Formatter(self.fmt)
        file_handler.setFormatter(formatter)
        self.file_handler = file_handler
        self.file_handler.suffix = "%Y%m%d"
        self.addHandler(self.file_handler)

class AbstractMetrics():
    def __new__(cls):
        if not hasattr(cls,'instance'):
            cls.instance=super(AbstractMetrics,cls).__new__(cls)
            return cls.instance
    
    def __init__(self):
      pid=str(os.getpid())
      env_list = os.environ
      #logdir = env_list.get('logdir')
      #logPath = os.path.join(logdir,'ddsMetric.log')
      #logging.basicConfig(filename=logPath ,level=logging.INFO,filemode='a',format='%(asctime)s pid:%(process)d - %(levelname)s - %(message)s')
      fmt='%(asctime)s pid:%(process)d - %(levelname)s - %(message)s'
      self.logging=mylogger('ddsMetric',time=False,fmt=fmt)
      self.concurrents={}
      self.metricsMap={}
      self.metriclock=Lock()
      timer=threading.Timer(self.getEnv()["DDS_SEND_INTERVAL"],self.sendMetrics)
      timer.start()

    def getEnv(self):
        env_list = os.environ
        
        return {
            "DDS_DSF_METRICS_SERVICE":'http://122.19.13.146:8888/icbc/dds/dds_dsf_metrics_service/dds/metrics/upload' if not env_list.get("ddsAPI") else env_list.get("ddsAPI")+'/dds_dsf_metrics_service/dds/metrics/upload',
            "DDS_SEND_INTERVAL":60 if not env_list.get("dds_send_interval") else int(env_list.get("dds_send_interval")),
            "DDS_SERVICE_GROUP":'MLP-DL' if not env_list.get("ddsServiceGroup") else env_list.get("ddsServiceGroup"),
            "SERVICE_NAME":(env_list.get("ddsServiceName_prefix")+"_"+env_list.get("appName")+"_"+env_list.get("modelName")).upper(),
            "LOCAL_IP":env_list.get("_PAAS_NODE_NAME"),
            "LOCAL_PORT":env_list.get("_PAAS_PORT_"+env_list.get("predictPort"))
        }
        
        '''
        return {
            "DDS_DSF_METRICS_SERVICE":env_list.get("ddsAPI")+'/dds_dsf_metrics_service/dds/metrics/upload',
            "DDS_SEND_INTERVAL":60 if not env_list.get("dds_send_interval") else int(env_list.get("dds_send_interval")),
            "DDS_SERVICE_GROUP":'DSF-TEST' if not env_list.get("ddsServiceGroup") else env_list.get("ddsServiceGroup"),
            "SERVICE_NAME":(env_list.get("ddsServiceName_prefix")+"_"+env_list.get("appName")+"_"+env_list.get("modelName")).upper(),
            "LOCAL_IP":env_list.get("_PAAS_NODE_NAME"),
            "LOCAL_PORT":env_list.get("httpPort")
        }   
        return {
            "DDS_DSF_METRICS_SERVICE":'http://122.19.13.146:8888/icbc/dds/dds_dsf_metrics_service/dds/metrics/upload',
            "DDS_SEND_INTERVAL":60,
            "DDS_SERVICE_GROUP":'DSF-TEST',
            "SERVICE_NAME":"DDS_MLP_MODELNAME",
            "LOCAL_IP":"122.71.253.89",
            "LOCAL_PORT":"35030"
        }
        '''
    
    
    '''
    记录本次请求的调用结果
        * @param metricsKey
        *            当前统计所使用的key
        * @param start
        * @param concurrent
        *            当前并发数
        * @param isSuccess
        *            请求是否成功
    '''
    def collect(self,metricsKey,start,concurrent,isSuccess):
      self.logging.info("in collect ===========start time is:")
      self.logging.info(start)
      currentTime = time.time()
      self.logging.info("in collect ===========end time is:")
      self.logging.info(currentTime)
      elapsed = (currentTime - start)*1000
      successCount = 1 if isSuccess else 0
      failureCount = 1 if not isSuccess else 0
      metricsMap = self.metricsMap
      self.logging.info("in collect:map before")
      self.logging.info(self.metricsMap)
      with self.metriclock:
        if metricsKey not in metricsMap:
          #manager=Manager()
          #metricsMap=manager.dict({metricsKey:
          metricsMap[metricsKey]={
              "maxElapsed":int(elapsed),#最大耗时
              "totalElapsed":int(elapsed),#总计耗时
              "successCount":int(successCount),#成功次数
              "failureCount":int(failureCount),#失败次数
              "maxConcurrent":int(concurrent)#最大并发
            }
         #}
        else:
          reference=metricsMap[metricsKey]
          reference["maxElapsed"] =  reference["maxElapsed"] if reference["maxElapsed"] > elapsed else int(elapsed)
          reference["totalElapsed"] += int(elapsed);
          reference["successCount"] += int(successCount);
          reference["failureCount"] += int(failureCount);
          reference["maxConcurrent"] = reference["maxConcurrent"] if reference["maxConcurrent"] > concurrent else int(concurrent)
      self.logging.info("in collect:map after")
      self.logging.info(self.metricsMap)


    '''
   发送监控数据
     * @param metricsKey
    *            当前统计所使用的key
    * @param start
    *            请求开始时间
    * @param concurrent
    *            当前并发数
    * @param isSuccess
    *            请求是否成功
'''
    def send(self,metricsItems):
      self.logging.info("in send")
      self.logging.info(metricsItems)
      currentTimestamp = time.time()
      resultList = []
      for item in metricsItems.items():
        keys=item[0].split("@@")
        #path= ("/"+keys[0]) if not keys[0].startswith("/") else keys[0]
        metricsObject={
            "type":"provider",#统计类型：provider-提供者，consumer-消费者
            "timestamp":int(currentTimestamp*1000),#时间戳
            "consumer":keys[1],#消费方ip:port
            "provider":self.getEnv()["LOCAL_IP"]+":"+self.getEnv()["LOCAL_PORT"],#提供方ip:port
            "serviceName":self.getEnv()["SERVICE_NAME"],#服务名 
            "servicePath":keys[0],#服务路径
            "providerServiceGroup":self.getEnv()["DDS_SERVICE_GROUP"],#服务群组
            "intervalTime":self.getEnv()["DDS_SEND_INTERVAL"]*1000,#DDS统计监控线程的上报间隔
            "metricsItem":item[1]
            
        }
        self.logging.info(metricsObject)
        resultList.append(metricsObject)
      url=self.getEnv()["DDS_DSF_METRICS_SERVICE"]
      self.logging.info("in send:result List is:")
      self.logging.info(resultList)
      #tmp = urllib.parse.urlencode(resultList).encode('UTF-8')
      #self.logging.info (tmp)
      #self.logging.info(resultList)
      response = requests.post(url = url,  json = resultList ) # 生成页面请求的完整数据    
      #response = request.urlopen(req)# 发送页面请求    
      #self.logging.info help(response)
      code = response.status_code
      response.close()
      if code <200 or code >=300:
        self.logging.error('upload metric error'+url+',status_code is:'+str(code))
      else:
        self.logging.info('upload metric successful!'+url+',status_code is:'+str(code))
        
    '''
        收集统计数据并调用子类的发送接口

    '''
    def sendMetrics(self):
      self.logging.info("in send metrics:start")
      self.logging.info(self.metricsMap)
      if len(self.metricsMap) != 0:
          with self.metriclock:
            metricsItems = copy.deepcopy(self.metricsMap)
            self.metricsMap={}
          self.send(metricsItems)
      self.logging.info("in send metrics:end")
      #self.getEnv()["DDS_SEND_INTERVAL"]
      timer=threading.Timer(60,self.sendMetrics)
      timer.start()
      
        
if __name__ == "__main__":
    a= AbstractMetrics()
    metricItems={'/predict@@122.19.13.146': {'maxElapsed': 18686, 'totalElapsed': 18686, 'successCount': 1, 'failureCount': 0, 'maxConcurrent': 1}}
    a.send(metricItems)