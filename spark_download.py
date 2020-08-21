# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:37:07 2020

@author: kfzx-lujian
"""
import os,json
evrn=os.environ
trainParams=json.loads(evrn.get('trainParams'))
hadoop_file_names=evrn.get('HADOOP_FILE_NAME').split(",")
partitions=list(map(int,evrn.get('PARTITIONS').split(",")))
if len(partitions)==1:partitions=partitions*len(hadoop_file_names)
from pyspark.sql import SparkSession,SQLContext
spark=SparkSession.builder.getOrCreate()
sql=SQLContext(spark.sparkContext,spark)
cwd=os.getcwd()
for i,j in zip(hadoop_file_names,partitions):
    if j>0:
        if i == "custpath":
            sql.read.csv(trainParams[i],sep=chr(27),header=False).limit(5000000).repartition(j).write.csv(f"file://{cwd}/{i}",sep=chr(27),header=False,mode='overwrite')
        else:
            sql.read.csv(trainParams[i],sep=chr(27),header=False).repartition(j).write.csv(f"file://{cwd}/{i}",sep=chr(27),header=False,mode='overwrite')
        files= os.listdir(f"{cwd}/{i}")
        files=list(filter(lambda x:x[-4:]==".csv",files))
        if len(files)>1:
            for k,file in enumerate(files):
                assert os.system(f"mv {cwd}/{i}/{file} {cwd}/{i}{k}")==0
        else:
            file=files[0]
            assert os.system(f"mv {cwd}/{i}/{file} {cwd}/{i}.txt")==0
    else:
        sql.read.csv(trainParams[i],sep=chr(27),header=False).write.csv(f"file://{cwd}/{i}",sep=chr(27),header=False,mode='overwrite')