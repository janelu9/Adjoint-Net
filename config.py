# config.py
import os
import gevent.monkey
gevent.monkey.patch_all()
#import multiprocessing

env_list = os.environ
logdir = env_list.get('logdir')

debug = False
loglevel = 'info'
bind = "0.0.0.0:6520"
errorlog = os.path.join(logdir, "debug.log")
pidfile = os.path.join(logdir, "gunicorn.pid")
accesslog = os.path.join(logdir, "access.log")
daemon = False


# 启动的进程数
# workers = multiprocessing.cpu_count()
# threads = 2
workers = int(env_list.get('modelVersion'))
worker_class = 'gevent'
worker_connections = 20
x_forwarded_for_header = 'X-FORWARDED-FOR'
timeout = 60