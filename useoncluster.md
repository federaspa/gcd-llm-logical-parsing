## Install llama_cpp_python on Philips cluster: 

```
cadenv -r 12.1 cuda
export LD_LIBRARY_PATH=/cadappl/cuda/12.1/lib64/

CUDACXX=/cadappl/cuda/12.1/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
```