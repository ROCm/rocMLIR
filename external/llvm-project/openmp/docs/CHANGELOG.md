ROCm 5.7.0 OpenMP Release Notes

MI300:
-	Enable OMPX_APU_MAPS env var for MI200 and gfx942.  
-	Add gfx941 support id
-	Handle global pointers in forced USM
-	Add gfx940, gfx941, and gfx942 to build list
-	Support unsigned types for atomic CAS loop on gfx941.
-	Support for atomics as CAS loop for certain operations and data types. On gfx941, certain atomic operations must be implemented as CAS loops.
-	Limit OMPX_APU_MAPS behavior to APU archs only and only when HSA_XNACK=1.
-	Fix declare target variable access in unified_shared_memory mode link/to clause not yet support. Do not check for its presence to update declare target variable address in unified shared memory mode.

Nextgen AMDGPU plugin:   
-	Made nextgen plugin the default, the legacy plugin is deprecated.
-	Ensure nextgen plugins are installed in correct location for packaging.
-	Respect GPU_MAX_HW_QUEUES in AMDGPU nextgen plugin, takes precendence over the standard LIBOMPTARGET_AMDGPU_NUM_HSA_QUEUES environment variable.
-	Change LIBOMPTARGET_AMDGPU_TEAMS_PER_CU from 4 to 6
-	Initialize HSA queues lazily
-	Enable active HSA wait state. Adds HSA timeout hint of 2 seconds to the AMDGPU nextgen-plugin to improve performance of small kernels. Adds support for optional env vars LIBOMPTARGET_AMDGPU_KERNEL_BUSYWAIT and LIBOMPTARGET_AMDGPU_DATA_BUSYWAIT.
-	Fixed behavior of env-var OMPX_FORCE_SYNC_REGIONS. This env-var is used to force synchronous target regions. The default is to use an asynchronous implementation. 
-	Supporting COV5
-	Add hostexec tracing: LIBOMPTARGET_HOSTEXEC_TRACE
-	Implemented target OMPT callbacks and trace records support.

Specialized kernels:
-	Remove redundant copying of arrays when xteam reductions are active but not offloaded.
-	Require no thread state assertion for generating specialized kernels.
-	Tune number of teams for BigJumpLoop.
-	Cross Team Scan Implementation in the DeviceRTL
-	Enable specialized kernel generation with nested OpenMP pragma as long as there is no nested omp-parallel directive.

Misc:
-	Introduces compile-time limit for the number of GPUs supported in a
    system.
-	Correctly compute number of waves when workgroup size is less than wave size    
-	Restore LIBOMPTARGET_KERNEL_TRACE=3. will print DEVID traces, and API timings
-	Add -fopenpm-runtimelib={lib,lib-perf,lib-debug} to select libs
-	ASan support for openmp release,debug and perf libraries 
-	Fix RUNPATH for gdb plugin.
-	Change LDS lowering default to hybrid
-	Fix hang in OMPT support if flush trace is called when there are no helper threads.
-	Add warning if mixed HIP / OpenMP offloading, i.e. if HIP language mode is active but OpenMP target directives are encountered.
-	Insert alloca for kernel args at function entry block instead of the launch point.
    
    

