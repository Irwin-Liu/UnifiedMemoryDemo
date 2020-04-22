# UnifiedMemoryDemo
CUDA unified memory test on Jetson.

**Introduction**
ExplicitMemory.cu is used to test the time cost of H2D, KE(kernel execution) and D2H model, including a kernel and a postprocessing loop.
Unified Memory can used to save cudaMemcpy time on Jetson. In UnifiedMemory.cu, there is same scale kernel and postprocessing with ExplicitMemory.cu, but no need cudaMemcpy. In the postprocessing loop, if directly access unified memory of arrays of structures on CPU after KE, the performance is worse. If replaced by basic variable type, it can get better performance of access on CPU.

For the same arrays of structures on Jetson memory which both can be accessd by CPU, and the same postprocessing loop, it is strange for the different performance. 

**Build and Run**
```
make
./ExplicitMemory
./UnifiedMemory
```

