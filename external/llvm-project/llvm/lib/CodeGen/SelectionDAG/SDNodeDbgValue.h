//===-- llvm/CodeGen/SDNodeDbgValue.h - SelectionDAG dbg_value --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the SDDbgValue class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_SELECTIONDAG_SDNODEDBGVALUE_H
#define LLVM_LIB_CODEGEN_SELECTIONDAG_SDNODEDBGVALUE_H

#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/DataTypes.h"
#include <utility>

namespace llvm {

class DIVariable;
class DIExpression;
class SDNode;
class Value;
class raw_ostream;

/// Holds the information for a single machine location through SDISel; either
/// an SDNode, a constant, a stack location, or a virtual register.
class SDDbgOperand {
public:
  enum Kind {
    SDNODE = 0,  ///< Value is the result of an expression.
    CONST = 1,   ///< Value is a constant.
    FRAMEIX = 2, ///< Value is contents of a stack location.
    VREG = 3     ///< Value is a virtual register.
  };
  Kind getKind() const { return kind; }

  /// Returns the SDNode* for a register ref
  SDNode *getSDNode() const {
    assert(kind == SDNODE);
    return u.s.Node;
  }

  /// Returns the ResNo for a register ref
  unsigned getResNo() const {
    assert(kind == SDNODE);
    return u.s.ResNo;
  }

  /// Returns the Value* for a constant
  const Value *getConst() const {
    assert(kind == CONST);
    return u.Const;
  }

  /// Returns the FrameIx for a stack object
  unsigned getFrameIx() const {
    assert(kind == FRAMEIX);
    return u.FrameIx;
  }

  /// Returns the Virtual Register for a VReg
  unsigned getVReg() const {
    assert(kind == VREG);
    return u.VReg;
  }

  static SDDbgOperand fromNode(SDNode *Node, unsigned ResNo) {
    return SDDbgOperand(Node, ResNo);
  }
  static SDDbgOperand fromFrameIdx(unsigned FrameIdx) {
    return SDDbgOperand(FrameIdx, FRAMEIX);
  }
  static SDDbgOperand fromVReg(unsigned VReg) {
    return SDDbgOperand(VReg, VREG);
  }
  static SDDbgOperand fromConst(const Value *Const) {
    return SDDbgOperand(Const);
  }

  bool operator!=(const SDDbgOperand &Other) const { return !(*this == Other); }
  bool operator==(const SDDbgOperand &Other) const {
    if (kind != Other.kind)
      return false;
    switch (kind) {
    case SDNODE:
      return getSDNode() == Other.getSDNode() && getResNo() == Other.getResNo();
    case CONST:
      return getConst() == Other.getConst();
    case VREG:
      return getVReg() == Other.getVReg();
    case FRAMEIX:
      return getFrameIx() == Other.getFrameIx();
    }
    return false;
  }

private:
  Kind kind;
  union {
    struct {
      SDNode *Node;   ///< Valid for expressions.
      unsigned ResNo; ///< Valid for expressions.
    } s;
    const Value *Const; ///< Valid for constants.
    unsigned FrameIx;   ///< Valid for stack objects.
    unsigned VReg;      ///< Valid for registers.
  } u;

  /// Constructor for non-constants.
  SDDbgOperand(SDNode *N, unsigned R) : kind(SDNODE) {
    u.s.Node = N;
    u.s.ResNo = R;
  }
  /// Constructor for constants.
  SDDbgOperand(const Value *C) : kind(CONST) { u.Const = C; }
  /// Constructor for virtual registers and frame indices.
  SDDbgOperand(unsigned VRegOrFrameIdx, Kind Kind) : kind(Kind) {
    assert((Kind == VREG || Kind == FRAMEIX) &&
           "Invalid SDDbgValue constructor");
    if (kind == VREG)
      u.VReg = VRegOrFrameIdx;
    else
      u.FrameIx = VRegOrFrameIdx;
  }
};

class SDDbgDefKill {
  DILifetime *LifeTime;
  unsigned Order;
  bool Emitted = false;
  DebugLoc DL;

public:
  typedef enum { DK_KIND_DEF = 0, DK_KIND_KILL } DKKindT;

protected:
  SDDbgDefKill(DILifetime *LT, unsigned O, DebugLoc DLoc, DKKindT K)
      : LifeTime(LT), Order(O), DL(DLoc), Kind(K) {}
  DKKindT Kind;

public:
  DKKindT getKind() const { return Kind; }
  unsigned getOrder() const { return Order; }
  DILifetime *getLifetime() { return LifeTime; }
  DebugLoc getDebugLoc() { return DL; }
  void setEmitted() { Emitted = true; }
  bool isEmitted() { return Emitted; }
};

class SDDbgDef : public SDDbgDefKill {
  const Value *Referrer;
  // IROrder of the referrer Value
  // it is used to further insert the DBG_DEF
  // in the proper position
  unsigned ReferrerOrder;
  int FrameIndex = 0;
  Register Reg;
  std::optional<SDValue> SDVal;

public:
  SDDbgDef(DILifetime *LT, unsigned O, const Value *Ref, DebugLoc DLoc)
      : SDDbgDefKill(LT, O, DLoc, DK_KIND_DEF), Referrer(Ref) {}
  const Value *getReferrer() { return Referrer; }
  unsigned getReferrerOrder() { return ReferrerOrder; }
  void setReferrerOrder(unsigned O) { ReferrerOrder = O; }
  void SaveFI(int FI) { FrameIndex = FI; }
  void SaveReg(Register R) { Reg = R; }
  int getFI() { return FrameIndex; }
  Register getReg() { return Reg; }
  void setSDValue(std::optional<SDValue> SDV) { SDVal = SDV; }
  std::optional<SDValue> getSDValue() { return SDVal; }
  static bool classof(const SDDbgDefKill *DK) {
    return DK->getKind() == DK_KIND_DEF;
  }
};

class SDDbgKill : public SDDbgDefKill {
public:
  SDDbgKill(DILifetime *LT, unsigned O, DebugLoc DLoc)
      : SDDbgDefKill(LT, O, DLoc, DK_KIND_KILL) {}
  static bool classof(const SDDbgDefKill *DK) {
    return DK->getKind() == DK_KIND_KILL;
  }
};

/// Holds the information from a dbg_value node through SDISel.
/// We do not use SDValue here to avoid including its header.
class SDDbgValue {
public:

private:
  // SDDbgValues are allocated by a BumpPtrAllocator, which means the destructor
  // may not be called; therefore all member arrays must also be allocated by
  // that BumpPtrAllocator, to ensure that they are correctly freed.
  size_t NumLocationOps;
  SDDbgOperand *LocationOps;
  // SDNode dependencies will be calculated as SDNodes that appear in
  // LocationOps plus these AdditionalDependencies.
  size_t NumAdditionalDependencies;
  SDNode **AdditionalDependencies;
  DIVariable *Var;
  DIExpression *Expr;
  DebugLoc DL;
  unsigned Order;
  bool IsIndirect;
  bool IsVariadic;
  bool Invalid = false;
  bool Emitted = false;

public:
  SDDbgValue(BumpPtrAllocator &Alloc, DIVariable *Var, DIExpression *Expr,
             ArrayRef<SDDbgOperand> L, ArrayRef<SDNode *> Dependencies,
             bool IsIndirect, DebugLoc DL, unsigned O, bool IsVariadic)
      : NumLocationOps(L.size()),
        LocationOps(Alloc.Allocate<SDDbgOperand>(L.size())),
        NumAdditionalDependencies(Dependencies.size()),
        AdditionalDependencies(Alloc.Allocate<SDNode *>(Dependencies.size())),
        Var(Var), Expr(Expr), DL(DL), Order(O), IsIndirect(IsIndirect),
        IsVariadic(IsVariadic) {
    assert(IsVariadic || L.size() == 1);
    assert(!(IsVariadic && IsIndirect));
    std::copy(L.begin(), L.end(), LocationOps);
    std::copy(Dependencies.begin(), Dependencies.end(), AdditionalDependencies);
  }

  // We allocate arrays with the BumpPtrAllocator and never free or copy them,
  // for LocationOps and AdditionalDependencies, as we never expect to copy or
  // destroy an SDDbgValue. If we ever start copying or destroying instances, we
  // should manage the allocated memory appropriately.
  SDDbgValue(const SDDbgValue &Other) = delete;
  SDDbgValue &operator=(const SDDbgValue &Other) = delete;
  ~SDDbgValue() = delete;

  /// Returns the DIVariable pointer for the variable.
  DIVariable *getVariable() const { return Var; }

  /// Returns the DIExpression pointer for the expression.
  DIExpression *getExpression() const { return Expr; }

  ArrayRef<SDDbgOperand> getLocationOps() const {
    return ArrayRef<SDDbgOperand>(LocationOps, NumLocationOps);
  }

  SmallVector<SDDbgOperand> copyLocationOps() const {
    return SmallVector<SDDbgOperand>(LocationOps, LocationOps + NumLocationOps);
  }

  // Returns the SDNodes which this SDDbgValue depends on.
  SmallVector<SDNode *> getSDNodes() const {
    SmallVector<SDNode *> Dependencies;
    for (const SDDbgOperand &DbgOp : getLocationOps())
      if (DbgOp.getKind() == SDDbgOperand::SDNODE)
        Dependencies.push_back(DbgOp.getSDNode());
    for (SDNode *Node : getAdditionalDependencies())
      Dependencies.push_back(Node);
    return Dependencies;
  }

  ArrayRef<SDNode *> getAdditionalDependencies() const {
    return ArrayRef<SDNode *>(AdditionalDependencies,
                              NumAdditionalDependencies);
  }

  /// Returns whether this is an indirect value.
  bool isIndirect() const { return IsIndirect; }

  bool isVariadic() const { return IsVariadic; }

  /// Returns the DebugLoc.
  const DebugLoc &getDebugLoc() const { return DL; }

  /// Returns the SDNodeOrder.  This is the order of the preceding node in the
  /// input.
  unsigned getOrder() const { return Order; }

  /// setIsInvalidated / isInvalidated - Setter / getter of the "Invalidated"
  /// property. A SDDbgValue is invalid if the SDNode that produces the value is
  /// deleted.
  void setIsInvalidated() { Invalid = true; }
  bool isInvalidated() const { return Invalid; }

  /// setIsEmitted / isEmitted - Getter/Setter for flag indicating that this
  /// SDDbgValue has been emitted to an MBB.
  void setIsEmitted() { Emitted = true; }
  bool isEmitted() const { return Emitted; }

  /// clearIsEmitted - Reset Emitted flag, for certain special cases where
  /// SDDbgValue is emitted twice. DBG_INSTR_REF depends on this behaviour.
  void clearIsEmitted() { Emitted = false; }

  LLVM_DUMP_METHOD void dump() const;
  LLVM_DUMP_METHOD void print(raw_ostream &OS) const;
};

/// Holds the information from a dbg_label node through SDISel.
/// We do not use SDValue here to avoid including its header.
class SDDbgLabel {
  MDNode *Label;
  DebugLoc DL;
  unsigned Order;

public:
  SDDbgLabel(MDNode *Label, DebugLoc dl, unsigned O)
      : Label(Label), DL(std::move(dl)), Order(O) {}

  /// Returns the MDNode pointer for the label.
  MDNode *getLabel() const { return Label; }

  /// Returns the DebugLoc.
  const DebugLoc &getDebugLoc() const { return DL; }

  /// Returns the SDNodeOrder.  This is the order of the preceding node in the
  /// input.
  unsigned getOrder() const { return Order; }
};

} // end llvm namespace

#endif
