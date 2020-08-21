import faiss
import numpy as np
index=faiss.IndexFlatIP(32)
index.add(np.random.random((2000,32)).astype('float32'))
res=faiss.StandardGpuResources()
res.setTempMemory(2*1024*1024*1024)
gpu_index = faiss.index_cpu_to_gpu(res,0,index)
