import os
import subprocess
import shlex

try:
    gpu_id= int(os.environ['CUDA_VISIBLE_DEVICES'])
except:
    None



    
def getMemGPU(useGPU):
    if not(useGPU):
        return None
    else:
        command_line = 'nvidia-smi --query-gpu=memory.total,memory.used,memory.free -i {} --format=csv,noheader,nounits'.format(gpu_id)
        args = shlex.split(command_line)

        sp = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out_str = sp.communicate()
        outlist = out_str[0].decode("utf-8").split('\n')[0].split(',')
        return(int(outlist[0])*1024**2,int(outlist[1])*1024**2,int(outlist[2])*1024**2)

    
def freeGPU():
    a,b,_ = getMemGPU(True)
    return a-b
