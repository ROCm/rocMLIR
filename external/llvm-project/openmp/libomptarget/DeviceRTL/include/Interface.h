//===-------- Interface.h - OpenMP interface ---------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_DEVICERTL_INTERFACE_H
#define OMPTARGET_DEVICERTL_INTERFACE_H

#include "Types.h"
#include "Xteamr.h"

/// External API
///
///{

extern "C" {

/// ICV: dyn-var, constant 0
///
/// setter: ignored.
/// getter: returns 0.
///
///{
void omp_set_dynamic(int);
int omp_get_dynamic(void);
///}

/// ICV: nthreads-var, integer
///
/// scope: data environment
///
/// setter: ignored.
/// getter: returns false.
///
/// implementation notes:
///
///
///{
void omp_set_num_threads(int);
int omp_get_max_threads(void);
///}

/// ICV: thread-limit-var, computed
///
/// getter: returns thread limited defined during launch.
///
///{
int omp_get_thread_limit(void);
///}

/// ICV: max-active-level-var, constant 1
///
/// setter: ignored.
/// getter: returns 1.
///
///{
void omp_set_max_active_levels(int);
int omp_get_max_active_levels(void);
///}

/// ICV: places-partition-var
///
///
///{
///}

/// ICV: active-level-var, 0 or 1
///
/// getter: returns 0 or 1.
///
///{
int omp_get_active_level(void);
///}

/// ICV: level-var
///
/// getter: returns parallel region nesting
///
///{
int omp_get_level(void);
///}

/// ICV: run-sched-var
///
///
///{
void omp_set_schedule(omp_sched_t, int);
void omp_get_schedule(omp_sched_t *, int *);
///}

/// TODO this is incomplete.
int omp_get_num_threads(void);
int omp_get_thread_num(void);
void omp_set_nested(int);

int omp_get_nested(void);

void omp_set_max_active_levels(int Level);

int omp_get_max_active_levels(void);

omp_proc_bind_t omp_get_proc_bind(void);

int omp_get_num_places(void);

int omp_get_place_num_procs(int place_num);

void omp_get_place_proc_ids(int place_num, int *ids);

int omp_get_place_num(void);

int omp_get_partition_num_places(void);

void omp_get_partition_place_nums(int *place_nums);

int omp_get_cancellation(void);

void omp_set_default_device(int deviceId);

int omp_get_default_device(void);

int omp_get_num_devices(void);

int omp_get_device_num(void);

int omp_get_num_teams(void);

int omp_get_team_num();

int omp_get_initial_device(void);

void *llvm_omp_target_dynamic_shared_alloc();

/// Synchronization
///
///{
void omp_init_lock(omp_lock_t *Lock);

void omp_destroy_lock(omp_lock_t *Lock);

void omp_set_lock(omp_lock_t *Lock);

void omp_unset_lock(omp_lock_t *Lock);

int omp_test_lock(omp_lock_t *Lock);
///}

/// Tasking
///
///{
extern "C" {
int omp_in_final(void);

int omp_get_max_task_priority(void);

void omp_fulfill_event(uint64_t);
}
///}

/// Misc
///
///{
double omp_get_wtick(void);

double omp_get_wtime(void);
///}

/// OpenMP 5.1 Memory Management routines (from libomp)
/// OpenMP allocator API is currently unimplemented, including traits.
/// All allocation routines will directly call the global memory allocation
/// routine and, consequently, omp_free will call device memory deallocation.
///
/// {
omp_allocator_handle_t omp_init_allocator(omp_memspace_handle_t m, int ntraits,
                                          omp_alloctrait_t traits[]);

void omp_destroy_allocator(omp_allocator_handle_t allocator);

void omp_set_default_allocator(omp_allocator_handle_t a);

omp_allocator_handle_t omp_get_default_allocator(void);

void *omp_alloc(uint64_t size,
                omp_allocator_handle_t allocator = omp_null_allocator);

void *omp_aligned_alloc(uint64_t align, uint64_t size,
                        omp_allocator_handle_t allocator = omp_null_allocator);

void *omp_calloc(uint64_t nmemb, uint64_t size,
                 omp_allocator_handle_t allocator = omp_null_allocator);

void *omp_aligned_calloc(uint64_t align, uint64_t nmemb, uint64_t size,
                         omp_allocator_handle_t allocator = omp_null_allocator);

void *omp_realloc(void *ptr, uint64_t size,
                  omp_allocator_handle_t allocator = omp_null_allocator,
                  omp_allocator_handle_t free_allocator = omp_null_allocator);

void omp_free(void *ptr, omp_allocator_handle_t allocator = omp_null_allocator);
/// }

/// CUDA exposes a native malloc/free API, while ROCm does not.
//// Any re-definitions of malloc/free delete the native CUDA
//// but they are necessary
#ifdef __AMDGCN__
void *malloc(uint64_t Size);
void free(void *Ptr);
size_t external_get_local_size(uint32_t dim);
size_t external_get_num_groups(uint32_t dim);
#endif
}

extern "C" {
/// Allocate \p Bytes in "shareable" memory and return the address. Needs to be
/// called balanced with __kmpc_free_shared like a stack (push/pop). Can be
/// called by any thread, allocation happens *per thread*.
void *__kmpc_alloc_shared(uint64_t Bytes);

/// Deallocate \p Ptr. Needs to be called balanced with __kmpc_alloc_shared like
/// a stack (push/pop). Can be called by any thread. \p Ptr has to be the
/// allocated by __kmpc_alloc_shared by the same thread.
void __kmpc_free_shared(void *Ptr, uint64_t Bytes);

/// Get a pointer to the memory buffer containing dynamically allocated shared
/// memory configured at launch.
void *__kmpc_get_dynamic_shared();

/// Allocate sufficient space for \p NumArgs sequential `void*` and store the
/// allocation address in \p GlobalArgs.
///
/// Called by the main thread prior to a parallel region.
///
/// We also remember it in GlobalArgsPtr to ensure the worker threads and
/// deallocation function know the allocation address too.
void __kmpc_begin_sharing_variables(void ***GlobalArgs, uint64_t NumArgs);

/// Deallocate the memory allocated by __kmpc_begin_sharing_variables.
///
/// Called by the main thread after a parallel region.
void __kmpc_end_sharing_variables();

/// Store the allocation address obtained via __kmpc_begin_sharing_variables in
/// \p GlobalArgs.
///
/// Called by the worker threads in the parallel region (function).
void __kmpc_get_shared_variables(void ***GlobalArgs);

/// External interface to get the thread ID.
uint32_t __kmpc_get_hardware_thread_id_in_block();

/// External interface to get the number of threads.
uint32_t __kmpc_get_hardware_num_threads_in_block();

/// External interface to get the warp size.
uint32_t __kmpc_get_warp_size();

/// External interface to get the block size
uint32_t __kmpc_get_hardware_num_blocks();

/// Kernel
///
///{
int8_t __kmpc_is_spmd_exec_mode();

int32_t __kmpc_target_init(IdentTy *Ident, int8_t Mode,
                           bool UseGenericStateMachine);

void __kmpc_target_deinit(IdentTy *Ident, int8_t Mode);

///}

/// Reduction
///
///{
void __kmpc_nvptx_end_reduce(int32_t TId);

void __kmpc_nvptx_end_reduce_nowait(int32_t TId);

int32_t __kmpc_nvptx_parallel_reduce_nowait_v2(
    IdentTy *Loc, int32_t TId, int32_t num_vars, uint64_t reduce_size,
    void *reduce_data, ShuffleReductFnTy shflFct, InterWarpCopyFnTy cpyFct);

int32_t __kmpc_nvptx_teams_reduce_nowait_v2(
    IdentTy *Loc, int32_t TId, void *GlobalBuffer, uint32_t num_of_records,
    void *reduce_data, ShuffleReductFnTy shflFct, InterWarpCopyFnTy cpyFct,
    ListGlobalFnTy lgcpyFct, ListGlobalFnTy lgredFct, ListGlobalFnTy glcpyFct,
    ListGlobalFnTy glredFct);
///}

/// Cross team helper functions for special case reductions
///{
///   THESE INTERFACES kmpc_xteam_ WILL BE DEPRACATED AND REPLACED WITH BELOW
///   kmpc_xteamr_
void __kmpc_xteam_sum_d(double, double *);
void __kmpc_xteam_sum_f(float, float *);
void __kmpc_xteam_sum_cd(double _Complex, double _Complex *);
void __kmpc_xteam_sum_cf(float _Complex, float _Complex *);
void __kmpc_xteam_sum_i(int, int *);
void __kmpc_xteam_sum_ui(unsigned int, unsigned int *);
void __kmpc_xteam_sum_l(long int, long int *);
void __kmpc_xteam_sum_ul(unsigned long, unsigned long *);
void __kmpc_xteam_max_d(double, double *);
void __kmpc_xteam_max_f(float, float *);
void __kmpc_xteam_max_i(int, int *);
void __kmpc_xteam_max_ui(unsigned int, unsigned int *);
void __kmpc_xteam_max_l(long int, long int *);
void __kmpc_xteam_max_ul(unsigned long, unsigned long *);
void __kmpc_xteam_min_d(double, double *);
void __kmpc_xteam_min_f(float, float *);
void __kmpc_xteam_min_i(int, int *);
void __kmpc_xteam_min_ui(unsigned int, unsigned int *);
void __kmpc_xteam_min_l(long int, long int *);
void __kmpc_xteam_min_ul(unsigned long, unsigned long *);

///  __kmpc_xteamr_<rtype>_<dtype>: Helper functions for Cross Team reductions
///    arg1: the thread local reduction value.
///    arg2: pointer to where result is written.
///    arg3: global array of team values for this reduction instance.
///    arg4: atomic counter of completed teams for this reduction instance.
void __kmpc_xteamr_sum_d(double, double *, double *, uint32_t *);
void __kmpc_xteamr_sum_f(float, float *, float *, uint32_t *);
void __kmpc_xteamr_sum_cd(double _Complex, double _Complex *, double _Complex *,
                          uint32_t *);
void __kmpc_xteamr_sum_cf(float _Complex, float _Complex *, float _Complex *,
                          uint32_t *);
void __kmpc_xteamr_sum_i(int, int *, int *, uint32_t *);
void __kmpc_xteamr_sum_ui(unsigned int, unsigned int *, unsigned int *,
                          uint32_t *);
void __kmpc_xteamr_sum_l(long int, long int *, long int *, uint32_t *);
void __kmpc_xteamr_sum_ul(unsigned long, unsigned long *, unsigned long *,
                          uint32_t *);
void __kmpc_xteamr_max_d(double, double *, double *, uint32_t *);
void __kmpc_xteamr_max_f(float, float *, float *, uint32_t *);
void __kmpc_xteamr_max_i(int, int *, int *, uint32_t *);
void __kmpc_xteamr_max_ui(unsigned int, unsigned int *, unsigned int *,
                          uint32_t *);
void __kmpc_xteamr_max_l(long int, long int *, long int *, uint32_t *);
void __kmpc_xteamr_max_ul(unsigned long, unsigned long *, unsigned long *,
                          uint32_t *);
void __kmpc_xteamr_min_d(double, double *, double *, uint32_t *);
void __kmpc_xteamr_min_f(float, float *, float *, uint32_t *);
void __kmpc_xteamr_min_i(int, int *, int *, uint32_t *);
void __kmpc_xteamr_min_ui(unsigned int, unsigned int *, unsigned int *,
                          uint32_t *);
void __kmpc_xteamr_min_l(long int, long int *, long int *, uint32_t *);
void __kmpc_xteamr_min_ul(unsigned long, unsigned long *, unsigned long *,
                          uint32_t *);
///}

/// Synchronization
///
///{
void __kmpc_ordered(IdentTy *Loc, int32_t TId);

void __kmpc_end_ordered(IdentTy *Loc, int32_t TId);

int32_t __kmpc_cancel_barrier(IdentTy *Loc_ref, int32_t TId);

void __kmpc_barrier(IdentTy *Loc_ref, int32_t TId);

void __kmpc_impl_syncthreads();

void __kmpc_barrier_simple_spmd(IdentTy *Loc_ref, int32_t TId);

void __kmpc_barrier_simple_generic(IdentTy *Loc_ref, int32_t TId);

int32_t __kmpc_master(IdentTy *Loc, int32_t TId);

void __kmpc_end_master(IdentTy *Loc, int32_t TId);

int32_t __kmpc_masked(IdentTy *Loc, int32_t TId, int32_t Filter);

void __kmpc_end_masked(IdentTy *Loc, int32_t TId);

int32_t __kmpc_single(IdentTy *Loc, int32_t TId);

void __kmpc_end_single(IdentTy *Loc, int32_t TId);

void __kmpc_flush(IdentTy *Loc);

void __kmpc_flush_acquire(IdentTy *Loc);

void __kmpc_flush_release(IdentTy *Loc);

void __kmpc_flush_acqrel(IdentTy *Loc);

uint64_t __kmpc_warp_active_thread_mask(void);

void __kmpc_syncwarp(uint64_t Mask);

void __kmpc_critical(IdentTy *Loc, int32_t TId, CriticalNameTy *Name);

void __kmpc_end_critical(IdentTy *Loc, int32_t TId, CriticalNameTy *Name);
///}

/// Parallelism
///
///{
/// TODO
void __kmpc_kernel_prepare_parallel(ParallelRegionFnTy WorkFn);

/// TODO
bool __kmpc_kernel_parallel(ParallelRegionFnTy *WorkFn);

/// TODO
void __kmpc_kernel_end_parallel();

/// TODO
void __kmpc_push_proc_bind(IdentTy *Loc, uint32_t TId, int ProcBind);

/// TODO
void __kmpc_push_num_teams(IdentTy *Loc, int32_t TId, int32_t NumTeams,
                           int32_t ThreadLimit);

/// TODO
uint16_t __kmpc_parallel_level(IdentTy *Loc, uint32_t);

///}

/// Tasking
///
///{
TaskDescriptorTy *__kmpc_omp_task_alloc(IdentTy *, int32_t, int32_t,
                                        size_t TaskSizeInclPrivateValues,
                                        size_t SharedValuesSize,
                                        TaskFnTy TaskFn);

int32_t __kmpc_omp_task(IdentTy *Loc, uint32_t TId,
                        TaskDescriptorTy *TaskDescriptor);

int32_t __kmpc_omp_task_with_deps(IdentTy *Loc, uint32_t TId,
                                  TaskDescriptorTy *TaskDescriptor, int32_t,
                                  void *, int32_t, void *);

void __kmpc_omp_task_begin_if0(IdentTy *Loc, uint32_t TId,
                               TaskDescriptorTy *TaskDescriptor);

void __kmpc_omp_task_complete_if0(IdentTy *Loc, uint32_t TId,
                                  TaskDescriptorTy *TaskDescriptor);

void __kmpc_omp_wait_deps(IdentTy *Loc, uint32_t TId, int32_t, void *, int32_t,
                          void *);

void __kmpc_taskgroup(IdentTy *Loc, uint32_t TId);

void __kmpc_end_taskgroup(IdentTy *Loc, uint32_t TId);

int32_t __kmpc_omp_taskyield(IdentTy *Loc, uint32_t TId, int);

int32_t __kmpc_omp_taskwait(IdentTy *Loc, uint32_t TId);

void __kmpc_taskloop(IdentTy *Loc, uint32_t TId,
                     TaskDescriptorTy *TaskDescriptor, int,
                     uint64_t *LowerBound, uint64_t *UpperBound, int64_t, int,
                     int32_t, uint64_t, void *);

void *__kmpc_task_allow_completion_event(IdentTy *loc_ref,
                                                uint32_t gtid,
                                                TaskDescriptorTy *task);

/// Misc
///
///{
int32_t __kmpc_cancellationpoint(IdentTy *Loc, int32_t TId, int32_t CancelVal);

int32_t __kmpc_cancel(IdentTy *Loc, int32_t TId, int32_t CancelVal);
///}

/// Shuffle
///
///{
int32_t __kmpc_shuffle_int32(int32_t val, int16_t delta, int16_t size);
int64_t __kmpc_shuffle_int64(int64_t val, int16_t delta, int16_t size);
///}

/// __init_ThreadDSTPtrPtr is defined in Workshare.cpp to initialize
/// the static LDS global variable ThreadDSTPtrPtr to 0.
/// It is called in Kernel.cpp at the end of initializeRuntime().
void __init_ThreadDSTPtrPtr();
}

/// Extra API exposed by ROCm
extern "C" {
int omp_ext_get_warp_id(void);
int omp_ext_get_lane_id(void);
int omp_ext_get_master_thread_id(void);
int omp_ext_get_smid(void);
int omp_ext_is_spmd_mode(void);
unsigned long long omp_ext_get_active_threads_mask(void);
} // extern "C"

#endif
