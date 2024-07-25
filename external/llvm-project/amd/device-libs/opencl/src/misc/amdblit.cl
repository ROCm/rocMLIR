/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#if !defined NO_BLIT

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

typedef enum BatchMemOpType {
  STREAM_WAIT_VALUE_32 = 0x1,
  STREAM_WRITE_VALUE_32 = 0x2,
  STREAM_WAIT_VALUE_64 = 0x4,
  STREAM_WRITE_VALUE_64 = 0x5,
  STREAM_MEM_OP_BARRIER = 0x6,            // Currently not supported
  STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 0x3 // Currently not supported
} BatchMemOpType;

typedef union streamBatchMemOpParams_union {
  BatchMemOpType operation;
  struct streamMemOpWaitValueParams_t{
    BatchMemOpType operation;
    atomic_ulong* address;
    union {
      uint value;
      ulong value64;
    };
    uint flags;
    atomic_ulong* alias; // Not valid for AMD backend
  } waitValue;
  struct streamMemOpWriteValueParams_t{
    BatchMemOpType operation;
    atomic_ulong* address;
    union {
      uint value;
      ulong value64;
    };
    uint flags;
    atomic_ulong* alias; // Not valid for AMD backend
  } writeValue;
  struct streamMemOpFlushRemoteWritesParams_t{ // Currently not supported
    BatchMemOpType operation;
    uint flags;
  } flushRemoteWrites;
  struct streamMemOpMemoryBarrierParams_t{ // Currently not supported
    BatchMemOpType operation;
    uint flags;
  } memoryBarrier;
  ulong pad[6];
} BatchMemOpParams;


static const uint SplitCount = 3;

__attribute__((always_inline)) void
__amd_copyBufferToImage(
    __global uint *src,
    __write_only image2d_array_t dst,
    ulong4 srcOrigin,
    int4 dstOrigin,
    int4 size,
    uint4 format,
    ulong4 pitch)
{
    ulong idxSrc;
    int4 coordsDst;
    uint4 pixel;
    __global uint* srcUInt = src;
    __global ushort* srcUShort = (__global ushort*)src;
    __global uchar* srcUChar  = (__global uchar*)src;
    ushort tmpUShort;
    uint tmpUInt;

    coordsDst.x = get_global_id(0);
    coordsDst.y = get_global_id(1);
    coordsDst.z = get_global_id(2);
    coordsDst.w = 0;

    if ((coordsDst.x >= size.x) ||
        (coordsDst.y >= size.y) ||
        (coordsDst.z >= size.z)) {
        return;
    }

    idxSrc = (coordsDst.z * pitch.y +
       coordsDst.y * pitch.x + coordsDst.x) *
       format.z + srcOrigin.x;

    coordsDst.x += dstOrigin.x;
    coordsDst.y += dstOrigin.y;
    coordsDst.z += dstOrigin.z;

    // Check components
    switch (format.x) {
    case 1:
        // Check size
        if (format.y == 1) {
            pixel.x = (uint)srcUChar[idxSrc];
        }
        else if (format.y == 2) {
            pixel.x = (uint)srcUShort[idxSrc];
        }
        else {
            pixel.x = srcUInt[idxSrc];
        }
    break;
    case 2:
        // Check size
        if (format.y == 1) {
            tmpUShort = srcUShort[idxSrc];
            pixel.x = (uint)(tmpUShort & 0xff);
            pixel.y = (uint)(tmpUShort >> 8);
        }
        else if (format.y == 2) {
            tmpUInt = srcUInt[idxSrc];
            pixel.x = (tmpUInt & 0xffff);
            pixel.y = (tmpUInt >> 16);
        }
        else {
            pixel.x = srcUInt[idxSrc++];
            pixel.y = srcUInt[idxSrc];
        }
    break;
    case 4:
        // Check size
        if (format.y == 1) {
            tmpUInt = srcUInt[idxSrc];
            pixel.x = tmpUInt & 0xff;
            pixel.y = (tmpUInt >> 8) & 0xff;
            pixel.z = (tmpUInt >> 16) & 0xff;
            pixel.w = (tmpUInt >> 24) & 0xff;
        }
        else if (format.y == 2) {
            tmpUInt = srcUInt[idxSrc++];
            pixel.x = tmpUInt & 0xffff;
            pixel.y = (tmpUInt >> 16);
            tmpUInt = srcUInt[idxSrc];
            pixel.z = tmpUInt & 0xffff;
            pixel.w = (tmpUInt >> 16);
        }
        else {
            pixel.x = srcUInt[idxSrc++];
            pixel.y = srcUInt[idxSrc++];
            pixel.z = srcUInt[idxSrc++];
            pixel.w = srcUInt[idxSrc];
        }
    break;
    }
    // Write the final pixel
    write_imageui(dst, coordsDst, pixel);
}

__attribute__((always_inline)) void
__amd_copyImageToBuffer(
    __read_only image2d_array_t src,
    __global uint* dstUInt,
    __global ushort* dstUShort,
    __global uchar* dstUChar,
    int4 srcOrigin,
    ulong4 dstOrigin,
    int4 size,
    uint4 format,
    ulong4 pitch)
{
    ulong idxDst;
    int4 coordsSrc;
    uint4 texel;

    coordsSrc.x = get_global_id(0);
    coordsSrc.y = get_global_id(1);
    coordsSrc.z = get_global_id(2);
    coordsSrc.w = 0;

    if ((coordsSrc.x >= size.x) ||
        (coordsSrc.y >= size.y) ||
        (coordsSrc.z >= size.z)) {
        return;
    }

    idxDst = (coordsSrc.z * pitch.y + coordsSrc.y * pitch.x +
        coordsSrc.x) * format.z + dstOrigin.x;

    coordsSrc.x += srcOrigin.x;
    coordsSrc.y += srcOrigin.y;
    coordsSrc.z += srcOrigin.z;

    texel = read_imageui(src, coordsSrc);

    // Check components
    switch (format.x) {
    case 1:
        // Check size
        switch (format.y) {
        case 1:
            dstUChar[idxDst] = (uchar)texel.x;
            break;
        case 2:
            dstUShort[idxDst] = (ushort)texel.x;
            break;
        case 4:
            dstUInt[idxDst] = texel.x;
            break;
        }
    break;
    case 2:
        // Check size
        switch (format.y) {
        case 1:
            dstUShort[idxDst] = (ushort)texel.x |
               ((ushort)texel.y << 8);
            break;
        case 2:
            dstUInt[idxDst] = texel.x | (texel.y << 16);
            break;
        case 4:
            dstUInt[idxDst++] = texel.x;
            dstUInt[idxDst] = texel.y;
            break;
        }
    break;
    case 4:
        // Check size
        switch (format.y) {
        case 1:
            dstUInt[idxDst] = (uint)texel.x |
               (texel.y << 8) |
               (texel.z << 16) |
               (texel.w << 24);
            break;
        case 2:
            dstUInt[idxDst++] = texel.x | (texel.y << 16);
            dstUInt[idxDst] = texel.z | (texel.w << 16);
            break;
        case 4:
            dstUInt[idxDst++] = texel.x;
            dstUInt[idxDst++] = texel.y;
            dstUInt[idxDst++] = texel.z;
            dstUInt[idxDst] = texel.w;
            break;
        }
    break;
    }
}

__attribute__((always_inline)) void
__amd_copyImage(
    __read_only image2d_array_t src,
    __write_only image2d_array_t dst,
    int4 srcOrigin,
    int4 dstOrigin,
    int4 size)
{
    int4    coordsDst;
    int4    coordsSrc;

    coordsDst.x = get_global_id(0);
    coordsDst.y = get_global_id(1);
    coordsDst.z = get_global_id(2);
    coordsDst.w = 0;

    if ((coordsDst.x >= size.x) ||
        (coordsDst.y >= size.y) ||
        (coordsDst.z >= size.z)) {
        return;
    }

    coordsSrc = srcOrigin + coordsDst;
    coordsDst += dstOrigin;

    uint4  texel;
    texel = read_imageui(src, coordsSrc);
    write_imageui(dst, coordsDst, texel);
}

__attribute__((always_inline)) void
__amd_copyImage1DA(
    __read_only image2d_array_t src,
    __write_only image2d_array_t dst,
    int4 srcOrigin,
    int4 dstOrigin,
    int4 size)
{
    int4 coordsDst;
    int4 coordsSrc;

    coordsDst.x = get_global_id(0);
    coordsDst.y = get_global_id(1);
    coordsDst.z = get_global_id(2);
    coordsDst.w = 0;

    if ((coordsDst.x >= size.x) ||
        (coordsDst.y >= size.y) ||
        (coordsDst.z >= size.z)) {
        return;
    }

    coordsSrc = srcOrigin + coordsDst;
    coordsDst += dstOrigin;
    if (srcOrigin.w != 0) {
       coordsSrc.z = coordsSrc.y;
       coordsSrc.y = 0;
    }
    if (dstOrigin.w != 0) {
       coordsDst.z = coordsDst.y;
       coordsDst.y = 0;
    }

    uint4  texel;
    texel = read_imageui(src, coordsSrc);
    write_imageui(dst, coordsDst, texel);
}

__attribute__((always_inline)) void
__amd_copyBufferRect(
    __global uchar* src,
    __global uchar* dst,
    ulong4 srcRect,
    ulong4 dstRect,
    ulong4 size)
{
    ulong x = get_global_id(0);
    ulong y = get_global_id(1);
    ulong z = get_global_id(2);

    if ((x >= size.x) ||
        (y >= size.y) ||
        (z >= size.z)) {
        return;
    }

    ulong offsSrc = srcRect.z + x + y * srcRect.x + z * srcRect.y;
    ulong offsDst = dstRect.z + x + y * dstRect.x + z * dstRect.y;

    dst[offsDst] = src[offsSrc];
}

__attribute__((always_inline)) void
__amd_copyBufferRectAligned(
    __global uint* src,
    __global uint* dst,
    ulong4 srcRect,
    ulong4 dstRect,
    ulong4 size)
{
    ulong x = get_global_id(0);
    ulong y = get_global_id(1);
    ulong z = get_global_id(2);

    if ((x >= size.x) ||
        (y >= size.y) ||
        (z >= size.z)) {
        return;
    }

    ulong offsSrc = srcRect.z + x + y * srcRect.x + z * srcRect.y;
    ulong offsDst = dstRect.z + x + y * dstRect.x + z * dstRect.y;

    if (size.w == 16) {
        __global uint4* src4 = (__global uint4*)src;
        __global uint4* dst4 = (__global uint4*)dst;
        dst4[offsDst] = src4[offsSrc];
    }
    else {
        dst[offsDst] = src[offsSrc];
    }
}

__attribute__((always_inline)) void
__amd_copyBuffer(
    __global uchar* srcI,
    __global uchar* dstI,
    ulong srcOrigin,
    ulong dstOrigin,
    ulong size,
    uint remain)
{
    ulong id = get_global_id(0);

    if (id >= size) {
        return;
    }

    __global uchar* src = srcI + srcOrigin;
    __global uchar* dst = dstI + dstOrigin;

    if (remain == 8) {
        dst[id] = src[id];
    }
    else {
        if (id < (size - 1)) {
            __global uint* srcD = (__global uint*)(src);
            __global uint* dstD = (__global uint*)(dst);
            dstD[id] = srcD[id];
        }
        else {
            for (uint i = 0; i < remain; ++i) {
                dst[id * 4 + i] = src[id * 4 + i];
            }
        }
    }
}

__attribute__((always_inline)) void
__amd_copyBufferAligned(
    __global uint* src,
    __global uint* dst,
    ulong srcOrigin,
    ulong dstOrigin,
    ulong size,
    uint alignment)
{
    ulong id = get_global_id(0);

    if (id >= size) {
        return;
    }

    ulong   offsSrc = id + srcOrigin;
    ulong   offsDst = id + dstOrigin;

    if (alignment == 16) {
        __global uint4* src4 = (__global uint4*)src;
        __global uint4* dst4 = (__global uint4*)dst;
        dst4[offsDst] = src4[offsSrc];
    }
    else {
        dst[offsDst] = src[offsSrc];
    }
}

__attribute__((always_inline)) void
__amd_copyBufferExt(
    __global uchar* srcI,
    __global uchar* dstI,
    ulong srcOrigin,
    ulong dstOrigin,
    ulong size,
    uint remainder,
    uint aligned_size,
    ulong end_ptr,
    uint next_chunk) {
  ulong id = get_global_id(0);
  ulong id_remainder = id;

  __global uchar* src = srcI + srcOrigin;
  __global uchar* dst = dstI + dstOrigin;

  if (aligned_size == sizeof(ulong2)) {
    __global ulong2* srcD = (__global ulong2*)(src);
    __global ulong2* dstD = (__global ulong2*)(dst);
    while ((ulong)(&dstD[id]) < end_ptr) {
      dstD[id] = srcD[id];
      id += next_chunk;
    }
  } else {
    __global uint* srcD = (__global uint*)(src);
    __global uint* dstD = (__global uint*)(dst);
    while ((ulong)(&dstD[id]) < end_ptr) {
      dstD[id] = srcD[id];
      id += next_chunk;
    }
  }
  if ((remainder != 0) && (id_remainder == 0)) {
    for (ulong i = size - remainder; i < size; ++i) {
      dst[i] = src[i];
    }
  }
}

__attribute__((always_inline)) void
__amd_fillBuffer(
    __global uchar* bufUChar,
    __global uint* bufUInt,
    __constant uchar* pattern,
    uint patternSize,
    ulong offset,
    ulong size)
{
    ulong id = get_global_id(0);

    if (id >= size) {
        return;
    }

    if (bufUInt) {
       __global uint* element = &bufUInt[offset + id * patternSize];
       __constant uint*  pt = (__constant uint*)pattern;

        for (uint i = 0; i < patternSize; ++i) {
            element[i] = pt[i];
        }
    }
    else {
        __global uchar* element = &bufUChar[offset + id * patternSize];

        for (uint i = 0; i < patternSize; ++i) {
            element[i] = pattern[i];
        }
    }
}

__attribute__((always_inline)) void
__amd_fillBufferAligned(
    __global uchar* bufUChar,
    __global ushort* bufUShort,
    __global uint* bufUInt,
    __global ulong* bufULong,
    __constant uchar* pattern,
    uint patternSize,
    ulong offset,
    ulong size)
{
    ulong id = get_global_id(0);

    if (id >= size) {
        return;
    }

    if (bufULong) {
        __global ulong* element = &bufULong[offset + id * patternSize];
        __constant ulong*  pt = (__constant ulong*)pattern;

        for (uint i = 0; i < patternSize; ++i) {
            element[i] = pt[i];
        }
    }
    else if (bufUInt) {
        __global uint* element = &bufUInt[offset + id * patternSize];
        __constant uint*  pt = (__constant uint*)pattern;

        for (uint i = 0; i < patternSize; ++i) {
            element[i] = pt[i];
        }
    }
    else if (bufUShort) {
        __global ushort* element = &bufUShort[offset + id * patternSize];
        __constant ushort*  pt = (__constant ushort*)pattern;

        for (uint i = 0; i < patternSize; ++i) {
            element[i] = pt[i];
        }
    }
    else {
        __global uchar* element = &bufUChar[offset + id * patternSize];

        for (uint i = 0; i < patternSize; ++i) {
            element[i] = pattern[i];
        }
    }
}

__attribute__((always_inline)) void
    __amd_fillBufferAlignedExt(
    __global uchar* bufUChar,
    __global ushort* bufUShort,
    __global uint* bufUInt,
    __global ulong* bufULong,
    __global ulong2* bufULong2,
    __constant uchar* pattern,
    uint pattern_size,
    ulong offset,
    ulong end_ptr,
    uint next_chunk)
{
  int id = get_global_id(0);
  long cur_id = offset + id * pattern_size;
  if (bufULong2) {
    __global ulong2* element = &bufULong2[cur_id];
    __constant ulong2* pt = (__constant ulong2*)pattern;
    while ((ulong)element < end_ptr) {
      for (uint i = 0; i < pattern_size; ++i) {
        element[i] = pt[i];
      }
      element += next_chunk;
    }
  } else if (bufULong) {
    __global ulong* element = &bufULong[cur_id];
    __constant ulong* pt = (__constant ulong*)pattern;
    while ((ulong)element < end_ptr) {
      for (uint i = 0; i < pattern_size; ++i) {
        element[i] = pt[i];
      }
      element += next_chunk;
    }
  } else if (bufUInt) {
    __global uint* element = &bufUInt[cur_id];
    __constant uint* pt = (__constant uint*)pattern;
    while ((ulong)element < end_ptr) {
      for (uint i = 0; i < pattern_size; ++i) {
        element[i] = pt[i];
      }
      element += next_chunk;
    }
  } else if (bufUShort) {
    __global ushort* element = &bufUShort[cur_id];
    __constant ushort* pt = (__constant ushort*)pattern;
    while ((ulong)element < end_ptr) {
      for (uint i = 0; i < pattern_size; ++i) {
        element[i] = pt[i];
      }
      element += next_chunk;
    }
  } else {
    __global uchar* element = &bufUChar[cur_id];
    while ((ulong)element < end_ptr) {
      for (uint i = 0; i < pattern_size; ++i) {
        element[i] = pattern[i];
      }
      element += next_chunk;
    }
  }
}

__attribute__((always_inline)) void
__amd_fillBufferAligned2D(__global uchar* bufUChar,
                          __global ushort* bufUShort,
                          __global uint* bufUInt,
                          __global ulong* bufULong,
                          __constant uchar* pattern,
                          uint patternSize,
                          ulong origin,
                          ulong width,
                          ulong height,
                          ulong pitch)
{
  ulong tid_x = get_global_id(0);
  ulong tid_y = get_global_id(1);

  if (tid_x >= width || tid_y >= height) {
    return;
  }

  ulong offset = (tid_y * pitch + tid_x);

  if (bufULong) {
    __global ulong* element = &bufULong[origin + offset];
    __constant ulong* pt = (__constant ulong*)pattern;
    for (uint i = 0; i < patternSize; ++i) {
      element[i] = pt[i];
    }
  } else if (bufUInt) {
    __global uint* element = &bufUInt[origin + offset];
    __constant uint* pt = (__constant uint*)pattern;
    for (uint i = 0; i < patternSize; ++i) {
      element[i] = pt[i];
    }
  } else if (bufUShort) {
    __global ushort* element = &bufUShort[origin + offset];
    __constant ushort* pt = (__constant ushort*)pattern;
    for (uint i = 0; i < patternSize; ++i) {
      element[i] = pt[i];
    }
  } else if (bufUChar) {
    __global uchar* element = &bufUChar[origin + offset];
    __constant uchar* pt = (__constant uchar*)pattern;
    for (uint i = 0; i < patternSize; ++i) {
      element[i] = pt[i];
    }
  }
}

__attribute__((always_inline)) void
__amd_fillImage(
    __write_only image2d_array_t image,
    float4 patternFLOAT4,
    int4 patternINT4,
    uint4 patternUINT4,
    int4 origin,
    int4 size,
    uint type)
{
    int4  coords;

    coords.x = get_global_id(0);
    coords.y = get_global_id(1);
    coords.z = get_global_id(2);
    coords.w = 0;

    if ((coords.x >= size.x) ||
        (coords.y >= size.y) ||
        (coords.z >= size.z)) {
        return;
    }

    coords += origin;

    int SizeX = get_global_size(0);
    int AdjustedSizeX = size.x + origin.x;

    for (uint i = 0; i < SplitCount; ++i) {
        // Check components
        switch (type) {
        case 0:
            write_imagef(image, coords, patternFLOAT4);
            break;
        case 1:
            write_imagei(image, coords, patternINT4);
            break;
        case 2:
            write_imageui(image, coords, patternUINT4);
            break;
        }
        coords.x += SizeX;
        if (coords.x >= AdjustedSizeX) return;
    }
}


__attribute__((always_inline)) void
__amd_streamOpsWrite(
    __global atomic_uint* ptrUint,
    __global atomic_ulong* ptrUlong,
    ulong value) {

  // The launch parameters for this shader is a 1 grid work-item

  // 32-bit write
  if (ptrUint) {
    atomic_store_explicit(ptrUint, (uint)value, memory_order_relaxed, memory_scope_all_svm_devices);
  }
  // 64-bit write
  else {
    atomic_store_explicit(ptrUlong, value, memory_order_relaxed, memory_scope_all_svm_devices);
  }
}


__attribute__((always_inline)) void
__amd_streamOpsWait(
    __global atomic_uint* ptrUint,
    __global atomic_ulong* ptrUlong,
    ulong value, ulong compareOp, ulong mask) {

    // The launch parameters for this shader is a 1 grid work-item

    switch (compareOp) {
    case 0: //GEQ
      if (ptrUint) {
        while ((int)(atomic_load_explicit(ptrUint, memory_order_relaxed,
                    memory_scope_all_svm_devices) & (uint)mask) < (uint)value) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
      else {
        while ((long)(atomic_load_explicit(ptrUlong, memory_order_relaxed,
                    memory_scope_all_svm_devices) & mask) < value) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
      break;

    case 1: // EQ
      if (ptrUint) {
        while ((atomic_load_explicit(ptrUint, memory_order_relaxed,
                   memory_scope_all_svm_devices) & (uint)mask) != (uint)value) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
      else {
        while ((atomic_load_explicit(ptrUlong, memory_order_relaxed,
                   memory_scope_all_svm_devices) & mask) != value) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
      break;

    case 2: //AND
      if (ptrUint) {
        while (!((atomic_load_explicit(ptrUint, memory_order_relaxed,
                   memory_scope_all_svm_devices) & (uint)mask) & (uint)value)) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
      else {
        while (!((atomic_load_explicit(ptrUlong, memory_order_relaxed,
                   memory_scope_all_svm_devices) & mask) & value)) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
      break;

    case 3: //NOR
      if (ptrUint) {
        while (((atomic_load_explicit(ptrUint, memory_order_relaxed,
                 memory_scope_all_svm_devices) | (uint)value) & (uint)mask) == (uint)mask) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
      else {
        while (((atomic_load_explicit(ptrUlong, memory_order_relaxed,
                     memory_scope_all_svm_devices) | value) & mask) == mask) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
      break;
    }
}

// The kernel calling this function must be launched with 'count' workgroups each of size 1
__attribute__((always_inline)) void
__amd_batchMemOp(__global BatchMemOpParams* param,
                 uint count) {

  ulong id = get_global_id(0);

  switch (param[id].operation) {
    case STREAM_WAIT_VALUE_32:
      __amd_streamOpsWait((__global atomic_uint*)param[id].waitValue.address, NULL,
                          (uint)param[id].waitValue.value, (uint)param[id].waitValue.flags,
                          (ulong)~0UL);
      break;
    case STREAM_WRITE_VALUE_32:
      __amd_streamOpsWrite((__global atomic_uint*)param[id].writeValue.address, NULL,
                           (uint)param[id].writeValue.value);
      break;
    case STREAM_WAIT_VALUE_64:
      __amd_streamOpsWait(NULL, (__global atomic_ulong*)param[id].waitValue.address,
                          (ulong)param[id].waitValue.value64, (uint)param[id].waitValue.flags,
                          (ulong)~0UL);
      break;
    case STREAM_WRITE_VALUE_64:
      __amd_streamOpsWrite(NULL, (__global atomic_ulong*)param[id].writeValue.address,
                           (ulong)param[id].writeValue.value64);
      break;
    default:
      break;
  }
}
#endif
