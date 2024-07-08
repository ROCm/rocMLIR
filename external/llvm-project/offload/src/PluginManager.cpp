//===-- PluginManager.cpp - Plugin loading and communication API ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functionality for handling plugins.
//
//===----------------------------------------------------------------------===//

#include "PluginManager.h"
#include "OmptTracing.h"
#include "OpenMP/OMPT/Callback.h"
#include "Shared/Debug.h"
#include "Shared/Profile.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>

using namespace llvm;
using namespace llvm::sys;

PluginManager *PM = nullptr;

// Every plugin exports this method to create an instance of the plugin type.
#define PLUGIN_TARGET(Name) extern "C" GenericPluginTy *createPlugin_##Name();
#include "Shared/Targets.def"

void PluginManager::init() {
  TIMESCOPE();
  DP("Loading RTLs...\n");

  // Attempt to create an instance of each supported plugin.
#define PLUGIN_TARGET(Name)                                                    \
  do {                                                                         \
    Plugins.emplace_back(                                                      \
        std::unique_ptr<GenericPluginTy>(createPlugin_##Name()));              \
  } while (false);
#include "Shared/Targets.def"

  DP("RTLs loaded!\n");
}

void PluginManager::deinit() {
  TIMESCOPE();
  DP("Unloading RTLs...\n");

  for (auto &Plugin : Plugins) {
    if (auto Err = Plugin->deinit()) {
      [[maybe_unused]] std::string InfoMsg = toString(std::move(Err));
      DP("Failed to deinit plugin: %s\n", InfoMsg.c_str());
    }
    Plugin.release();
  }

  DP("RTLs unloaded!\n");
}

void PluginManager::initDevices(GenericPluginTy &RTL) {
  // If this RTL has already been initialized.
  if (PM->DeviceOffsets.contains(&RTL))
    return;
  TIMESCOPE();

  // If this RTL is not already in use, initialize it.
  assert(RTL.number_of_devices() > 0 && "Tried to initialize useless plugin!");

  // Initialize the device information for the RTL we are about to use.
  auto ExclusiveDevicesAccessor = getExclusiveDevicesAccessor();

  // Initialize the index of this RTL and save it in the used RTLs.
  int32_t DeviceOffset = ExclusiveDevicesAccessor->size();

  // Set the device identifier offset in the plugin.
  RTL.set_device_offset(DeviceOffset);

  int32_t NumberOfUserDevices = 0;
  int32_t NumPD = RTL.number_of_devices();
  ExclusiveDevicesAccessor->reserve(DeviceOffset + NumPD);
  // Auto zero-copy is a per-device property. We need to ensure
  // that all devices are suggesting to use it.
  bool UseAutoZeroCopy = !(NumPD == 0);
  // The following properties must have the same value for all devices.
  // They are surfaced per-device because the related properties
  // are computed as such in the plugins.
  for (int32_t PDevI = 0, UserDevId = DeviceOffset; PDevI < NumPD; PDevI++) {
    auto Device = std::make_unique<DeviceTy>(&RTL, UserDevId, PDevI);
    if (auto Err = Device->init()) {
      DP("Skip plugin known device %d: %s\n", PDevI,
         toString(std::move(Err)).c_str());
      continue;
    }
    UseAutoZeroCopy = UseAutoZeroCopy && Device->useAutoZeroCopy();

    ExclusiveDevicesAccessor->push_back(std::move(Device));
    ++NumberOfUserDevices;
    ++UserDevId;
  }

  // IsAPU, EagerMapsRequested and SupportsUnifiedMemory are properties
  // associated with devices but they must be the same for all devices.
  // We do not mix APUs with discrete GPUs. Eager maps is set by a host
  // environment variable.
  bool IsAPU = false;
  if (ExclusiveDevicesAccessor->size() > 0) {
    auto &Device = *(*ExclusiveDevicesAccessor)[0];
    IsAPU = Device.checkIfAPU();
  }
  bool EagerMapsRequested = BoolEnvar("OMPX_EAGER_ZERO_COPY_MAPS", false).get();

  // Auto Zero-Copy can only be currently triggered when the system is an
  // homogeneous APU architecture without attached discrete GPUs.
  // If all devices suggest to use it, change requirment flags to trigger
  // zero-copy behavior when mapping memory.
  if (UseAutoZeroCopy)
    addRequirements(OMPX_REQ_AUTO_ZERO_COPY);

  // Eager Zero-Copy Maps makes a "copy" execution turn into
  // an automatic zero-copy. It also applies to unified_shared_memory.
  // It is only available on APUs.
  if (IsAPU && EagerMapsRequested) {
    addRequirements(OMPX_REQ_EAGER_ZERO_COPY_MAPS);
    if (!(getRequirements() & OMP_REQ_UNIFIED_SHARED_MEMORY))
      addRequirements(OMPX_REQ_AUTO_ZERO_COPY);
  }

  // sanity checks for zero-copy depend on specific devices: request it here
  if ((ExclusiveDevicesAccessor->size() > 0) &&
      ((getRequirements() & OMP_REQ_UNIFIED_SHARED_MEMORY) ||
       (getRequirements() & OMPX_REQ_AUTO_ZERO_COPY))) {
    // APUs are assumed to be a homogeneous set of GPUs: ask
    // the first device in the system to run a sanity check.
    auto &Device = *(*ExclusiveDevicesAccessor)[0];
    // just skip checks if no devices are found in the system
    Device.zeroCopySanityChecksAndDiag(
        (getRequirements() & OMP_REQ_UNIFIED_SHARED_MEMORY),
        (getRequirements() & OMPX_REQ_AUTO_ZERO_COPY),
        (getRequirements() & OMPX_REQ_EAGER_ZERO_COPY_MAPS));
  }

  DeviceOffsets[&RTL] = DeviceOffset;
  DeviceUsed[&RTL] = NumberOfUserDevices;
  DP("Plugin has index %d, exposes %d out of %d devices!\n", DeviceOffset,
     NumberOfUserDevices, RTL.number_of_devices());
}

void PluginManager::initAllPlugins() {
  for (auto &R : Plugins)
    initDevices(*R);
}

static void registerImageIntoTranslationTable(TranslationTable &TT,
                                              int32_t DeviceOffset,
                                              int32_t NumberOfUserDevices,
                                              __tgt_device_image *Image) {

  // same size, as when we increase one, we also increase the other.
  assert(TT.TargetsTable.size() == TT.TargetsImages.size() &&
         "We should have as many images as we have tables!");

  // Resize the Targets Table and Images to accommodate the new targets if
  // required
  unsigned TargetsTableMinimumSize = DeviceOffset + NumberOfUserDevices;

  if (TT.TargetsTable.size() < TargetsTableMinimumSize) {
    TT.DeviceTables.resize(TargetsTableMinimumSize, {});
    TT.TargetsImages.resize(TargetsTableMinimumSize, 0);
    TT.TargetsEntries.resize(TargetsTableMinimumSize, {});
    TT.TargetsTable.resize(TargetsTableMinimumSize, 0);
  }

  // Register the image in all devices for this target type.
  for (int32_t I = 0; I < NumberOfUserDevices; ++I) {
    // If we are changing the image we are also invalidating the target table.
    if (TT.TargetsImages[DeviceOffset + I] != Image) {
      TT.TargetsImages[DeviceOffset + I] = Image;
      TT.TargetsTable[DeviceOffset + I] =
          0; // lazy initialization of target table.
    }
  }
}

void PluginManager::registerLib(__tgt_bin_desc *Desc) {
  PM->RTLsMtx.lock();

  // Add in all the OpenMP requirements associated with this binary.
  for (__tgt_offload_entry &Entry :
       llvm::make_range(Desc->HostEntriesBegin, Desc->HostEntriesEnd))
    if (Entry.flags == OMP_REGISTER_REQUIRES)
      PM->addRequirements(Entry.data);

  // Initialize all the plugins that have associated images.
  for (auto &Plugin : Plugins) {
    // Extract the exectuable image and extra information if availible.
    for (int32_t i = 0; i < Desc->NumDeviceImages; ++i) {
    if (Plugin->is_initialized())
      continue;

      if (!Plugin->is_valid_binary(&Desc->DeviceImages[i],
                                   /*Initialized=*/false))
        continue;

      if (auto Err = Plugin->init()) {
        [[maybe_unused]] std::string InfoMsg = toString(std::move(Err));
        DP("Failed to init plugin: %s\n", InfoMsg.c_str());
      } else {
        DP("Registered plugin %s with %d visible device(s)\n",
           Plugin->getName(), Plugin->number_of_devices());
      }
    }
  }

  // Extract the exectuable image and extra information if availible.
  for (int32_t i = 0; i < Desc->NumDeviceImages; ++i)
    PM->addDeviceImage(*Desc, Desc->DeviceImages[i]);

  // Register the images with the RTLs that understand them, if any.
  bool FoundCompatibleImage = false;
  for (DeviceImageTy &DI : PM->deviceImages()) {
    // Obtain the image and information that was previously extracted.
    __tgt_device_image *Img = &DI.getExecutableImage();

    GenericPluginTy *FoundRTL = nullptr;

    // Scan the RTLs that have associated images until we find one that supports
    // the current image.
    for (auto &R : PM->plugins()) {
      if (!R.number_of_devices())
        continue;

      if (!R.is_valid_binary(Img, /*Initialized=*/true)) {
        DP("Image " DPxMOD " is NOT compatible with RTL %s!\n",
           DPxPTR(Img->ImageStart), R.getName());
        continue;
      }

      DP("Image " DPxMOD " is compatible with RTL %s!\n",
         DPxPTR(Img->ImageStart), R.getName());

      PM->initDevices(R);

      // Initialize (if necessary) translation table for this library.
      PM->TrlTblMtx.lock();
      if (!PM->HostEntriesBeginToTransTable.count(Desc->HostEntriesBegin)) {
        PM->HostEntriesBeginRegistrationOrder.push_back(Desc->HostEntriesBegin);
        TranslationTable &TransTable =
            (PM->HostEntriesBeginToTransTable)[Desc->HostEntriesBegin];
        TransTable.HostTable.EntriesBegin = Desc->HostEntriesBegin;
        TransTable.HostTable.EntriesEnd = Desc->HostEntriesEnd;
      }

      // Retrieve translation table for this library.
      TranslationTable &TransTable =
          (PM->HostEntriesBeginToTransTable)[Desc->HostEntriesBegin];

      DP("Registering image " DPxMOD " with RTL %s!\n", DPxPTR(Img->ImageStart),
         R.getName());

      registerImageIntoTranslationTable(TransTable, PM->DeviceOffsets[&R],
                                        PM->DeviceUsed[&R], Img);
      PM->UsedImages.insert(Img);

      PM->TrlTblMtx.unlock();
      FoundRTL = &R;

      // if an RTL was found we are done - proceed to register the next image
      break;
    }

    if (!FoundRTL) {
      DP("No RTL found for image " DPxMOD "!\n", DPxPTR(Img->ImageStart));
    } else {
      FoundCompatibleImage = true;
    }
  }

  // Check if I can report any XNACK related image failures. The report
  // should happen only when we have not found a compatible RTL with
  // matching XNACK and we were expecting to have a match (i.e. the
  // image was hoping to find an RTL for an AMD GPU with XNACK support).
  if (!FoundCompatibleImage) {
    for (DeviceImageTy &DI : PM->deviceImages()) {
      __tgt_device_image *Img = &DI.getExecutableImage();
      for (auto &R : PM->plugins())
        R.check_invalid_image(Img);
    }
  }
  PM->RTLsMtx.unlock();

  DP("Done registering entries!\n");
}

// Temporary forward declaration, old style CTor/DTor handling is going away.
int target(ident_t *Loc, DeviceTy &Device, void *HostPtr,
           KernelArgsTy &KernelArgs, AsyncInfoTy &AsyncInfo);

void PluginManager::unregisterLib(__tgt_bin_desc *Desc) {
  DP("Unloading target library!\n");

  // Flush in-process OMPT trace records and shut down helper threads
  // before unloading the library.
  OMPT_TRACING_IF_ENABLED(llvm::omp::target::ompt::TraceRecordManager
                              .flushAndShutdownHelperThreads(););

  PM->RTLsMtx.lock();
  // Find which RTL understands each image, if any.
  for (DeviceImageTy &DI : PM->deviceImages()) {
    // Obtain the image and information that was previously extracted.
    __tgt_device_image *Img = &DI.getExecutableImage();

    GenericPluginTy *FoundRTL = NULL;

    // Scan the RTLs that have associated images until we find one that supports
    // the current image. We only need to scan RTLs that are already being used.
    for (auto &R : PM->plugins()) {
      if (!DeviceOffsets.contains(&R))
        continue;

      // Ensure that we do not use any unused images associated with this RTL.
      if (!UsedImages.contains(Img))
        continue;

      FoundRTL = &R;

      DP("Unregistered image " DPxMOD " from RTL\n", DPxPTR(Img->ImageStart));

      break;
    }

    // if no RTL was found proceed to unregister the next image
    if (!FoundRTL) {
      DP("No RTLs in use support the image " DPxMOD "!\n",
         DPxPTR(Img->ImageStart));
    }
  }
  PM->RTLsMtx.unlock();
  DP("Done unregistering images!\n");

  // Remove entries from PM->HostPtrToTableMap
  PM->TblMapMtx.lock();
  for (__tgt_offload_entry *Cur = Desc->HostEntriesBegin;
       Cur < Desc->HostEntriesEnd; ++Cur) {
    PM->HostPtrToTableMap.erase(Cur->addr);
  }

  // Remove translation table for this descriptor.
  auto TransTable =
      PM->HostEntriesBeginToTransTable.find(Desc->HostEntriesBegin);
  if (TransTable != PM->HostEntriesBeginToTransTable.end()) {
    DP("Removing translation table for descriptor " DPxMOD "\n",
       DPxPTR(Desc->HostEntriesBegin));
    PM->HostEntriesBeginToTransTable.erase(TransTable);
  } else {
    DP("Translation table for descriptor " DPxMOD " cannot be found, probably "
       "it has been already removed.\n",
       DPxPTR(Desc->HostEntriesBegin));
  }

  PM->TblMapMtx.unlock();

  DP("Done unregistering library!\n");
}

Expected<DeviceTy &> PluginManager::getDevice(uint32_t DeviceNo) {
  auto ExclusiveDevicesAccessor = getExclusiveDevicesAccessor();
  if (DeviceNo >= ExclusiveDevicesAccessor->size())
    return createStringError(
        inconvertibleErrorCode(),
        "Device number '%i' out of range, only %i devices available", DeviceNo,
        ExclusiveDevicesAccessor->size());

  return *(*ExclusiveDevicesAccessor)[DeviceNo];
}
