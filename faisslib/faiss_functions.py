import faiss

def Flat(vec,dim):
    index=faiss.IndexFlatIP(dim)
    index.add(vec)
    return index
    
    
def IVFFlat(vec,dim,nlist,nprobe):
    quantizer=faiss.IndexFlatIP(dim)
    index=faiss.IndexIVFFlat(quantizer,dim,nlist,faiss.METRIC_INNER_PRODUCT)
    index.train(vec)
    index.add(vec)
    index.nprobe=nprobe
    return index 

def IndexIVFPQ(vec,dim,nlist,nprobe,m):
    quantizer = faiss.IndexFlatIP(dim) 
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8) 
    index.train(vec) 
    index.add(vec) 
    index.nprobe = nprobe 
    return index
    
def StandardGpuIVF(vec,dim,nlist):
    res=faiss.StandardGpuResources()
    quantizer=faiss.IndexFlatIP(dim)
    index_flat = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    gpu_index = faiss.index_cpu_to_gpu(res,0,index_flat)
    gpu_index.train(vec)
    gpu_index.add(vec)
    return faiss.index_gpu_to_cpu(gpu_index)

def StandardGpu(vec,dim,nlist):
    res=faiss.StandardGpuResources()
    index_flat=faiss.IndexFlatIP(dim)
    gpu_index = faiss.index_cpu_to_gpu(res,0,index_flat)
    gpu_index.add(vec)
    return faiss.index_gpu_to_cpu(gpu_index)
    
def StandardGpuFC(vec,dim,param):
    res=faiss.StandardGpuResources()
    index=faiss.index_factory(dim,param,faiss.METRIC_INNER_PRODUCT)
    gpu_index = faiss.index_cpu_to_gpu(res,0,index)
    gpu_index.train(vec)
    gpu_index.add(vec)
    return faiss.index_gpu_to_cpu(gpu_index)
    
def FC(vec,dim,param):
    index=faiss.index_factory(dim,param,faiss.METRIC_INNER_PRODUCT)
    index.train(vec)
    index.add(vec)
    return index

def save(index,file_path):
    faiss.write_index(index,file_path)
    
def load(file_path):
    return faiss.read_index(file_path)
    
