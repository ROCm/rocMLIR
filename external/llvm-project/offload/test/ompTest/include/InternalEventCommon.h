#ifndef OFFLOAD_TEST_OMPTEST_INCLUDE_INTERNALEVENTCOMMON_H
#define OFFLOAD_TEST_OMPTEST_INCLUDE_INTERNALEVENTCOMMON_H

#include "omp-tools.h"

#include <cassert>
#include <string>

namespace omptest {

namespace internal {
/// Enum values are used for comparison of observed and asserted events
/// List is based on OpenMP 5.2 specification, table 19.2 (page 447)
enum class EventTy {
  None,                     // not part of OpenMP spec, used for implementation
  AssertionSyncPoint,       // not part of OpenMP spec, used for implementation
  AssertionSuspend,         // not part of OpenMP spec, used for implementation
  BufferRecord,             // not part of OpenMP spec, used for implementation
  BufferRecordDeallocation, // not part of OpenMP spec, used for implementation
  ThreadBegin,
  ThreadEnd,
  ParallelBegin,
  ParallelEnd,
  Work,
  Dispatch,
  TaskCreate,     // TODO: Implement
  Dependences,    // TODO: Implement
  TaskDependence, // TODO: Implement
  TaskSchedule,   // TODO: Implement
  ImplicitTask,   // TODO: Implement
  Masked,         // TODO: Implement
  SyncRegion,
  MutexAcquire, // TODO: Implement
  Mutex,        // TODO: Implement
  NestLock,     // TODO: Implement
  Flush,        // TODO: Implement
  Cancel,       // TODO: Implement
  DeviceInitialize,
  DeviceFinalize,
  DeviceLoad,
  DeviceUnload,
  BufferRequest,
  BufferComplete,
  TargetDataOp,
  TargetDataOpEmi,
  Target,
  TargetEmi,
  TargetSubmit,
  TargetSubmitEmi,
  ControlTool
};

struct InternalEvent {
  EventTy Type;
  EventTy getType() const { return Type; }

  InternalEvent() : Type(EventTy::None) {}
  InternalEvent(EventTy T) : Type(T) {}
  virtual ~InternalEvent() = default;

  virtual bool equals(const InternalEvent *o) const {
    assert(false && "Base class implementation");
    return false;
  };

  virtual std::string toString() const {
    std::string S{"InternalEvent: Type="};
    S.append(std::to_string((uint32_t)Type));
    return S;
  }
};

#define event_class_stub(EvTy)                                                 \
  struct EvTy : public InternalEvent {                                         \
    virtual bool equals(const InternalEvent *o) const override;                \
    EvTy() : InternalEvent(EventTy::EvTy) {}                                   \
  };

#define event_class_w_custom_body(EvTy, ...)                                   \
  struct EvTy : public InternalEvent {                                         \
    virtual bool equals(const InternalEvent *o) const override;                \
    std::string toString() const override;                                     \
    __VA_ARGS__                                                                \
  };

#define event_class_operator_stub(EvTy)                                        \
  bool operator==(const EvTy &Expected, const EvTy &Observed) { return true; }

#define event_class_operator_w_body(EvTy, ...)                                 \
  bool operator==(const EvTy &Expected, const EvTy &Observed) { __VA_ARGS__ }

/// Template "base" for the cast functions generated in the define_cast_func
/// macro
template <typename To> const To *cast(const InternalEvent *From) {
  return nullptr;
}

/// Generates template specialization of the cast operation for the specified
/// EvTy as the template parameter
#define define_cast_func(EvTy)                                                 \
  template <> const EvTy *cast(const InternalEvent *From) {                    \
    if (From->getType() == EventTy::EvTy)                                      \
      return static_cast<const EvTy *>(From);                                  \
    return nullptr;                                                            \
  }

/// Auto generate the equals override to cast and dispatch to the specific class
/// operator==
#define class_equals_op(EvTy)                                                  \
  bool EvTy::equals(const InternalEvent *o) const {                            \
    if (const auto O = cast<EvTy>(o))                                          \
      return *this == *O;                                                      \
    return false;                                                              \
  }

} // namespace internal

} // namespace omptest

#endif
