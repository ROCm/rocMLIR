//===- llvm/CodeGen/DwarfExpression.h - Dwarf Compile Unit ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf compile unit.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DWARFEXPRESSION_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DWARFEXPRESSION_H

#include "ByteStreamer.h"
#include "DwarfDebug.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include <cassert>
#include <cstdint>
#include <iterator>
#include <optional>

namespace llvm {

class AsmPrinter;
class APInt;
class DwarfCompileUnit;
class DIELoc;
class TargetRegisterInfo;
class MachineLocation;

/// Base class containing the logic for constructing DWARF expressions
/// independently of whether they are emitted into a DIE or into a .debug_loc
/// entry.
///
/// Some DWARF operations, e.g. DW_OP_entry_value, need to calculate the size
/// of a succeeding DWARF block before the latter is emitted to the output.
/// To handle such cases, data can conditionally be emitted to a temporary
/// buffer, which can later on be committed to the main output. The size of the
/// temporary buffer is queryable, allowing for the size of the data to be
/// emitted before the data is committed.
class DwarfExpression {
protected:
  /// Holds information about all subregisters comprising a register location.
  struct Register {
    int DwarfRegNo;
    unsigned SubRegSize;
    const char *Comment;

    /// Create a full register, no extra DW_OP_piece operators necessary.
    static Register createRegister(int RegNo, const char *Comment) {
      return {RegNo, 0, Comment};
    }

    /// Create a subregister that needs a DW_OP_piece operator with SizeInBits.
    static Register createSubRegister(int RegNo, unsigned SizeInBits,
                                      const char *Comment) {
      return {RegNo, SizeInBits, Comment};
    }

    bool isSubRegister() const { return SubRegSize; }
  };

  /// Whether we are currently emitting an entry value operation.
  bool IsEmittingEntryValue = false;

  DwarfCompileUnit &CU;

  /// The register location, if any.
  SmallVector<Register, 2> DwarfRegs;

  /// Current Fragment Offset in Bits.
  uint64_t OffsetInBits = 0;

  /// Sometimes we need to add a DW_OP_bit_piece to describe a subregister.
  unsigned SubRegisterSizeInBits : 16;
  unsigned SubRegisterOffsetInBits : 16;

  /// The kind of location description being produced.
  enum { Unknown = 0, Register, Memory, Implicit };

  /// Additional location flags which may be combined with any location kind.
  /// Currently, entry values are not supported for the Memory location kind.
  enum { EntryValue = 1 << 0, Indirect = 1 << 1, CallSiteParamValue = 1 << 2 };

  unsigned LocationKind : 3;
  unsigned SavedLocationKind : 3;
  unsigned LocationFlags : 3;
  unsigned DwarfVersion : 4;

public:
  /// Set the location (\p Loc) and \ref DIExpression (\p DIExpr) to describe.
  void setLocation(const MachineLocation &Loc, const DIExpression *DIExpr);

  bool isUnknownLocation() const { return LocationKind == Unknown; }

  bool isMemoryLocation() const { return LocationKind == Memory; }

  bool isRegisterLocation() const { return LocationKind == Register; }

  bool isImplicitLocation() const { return LocationKind == Implicit; }

  bool isEntryValue() const { return LocationFlags & EntryValue; }

  bool isIndirect() const { return LocationFlags & Indirect; }

  bool isParameterValue() { return LocationFlags & CallSiteParamValue; }

  std::optional<uint8_t> TagOffset;

protected:
  /// Push a DW_OP_piece / DW_OP_bit_piece for emitting later, if one is needed
  /// to represent a subregister.
  void setSubRegisterPiece(unsigned SizeInBits, unsigned OffsetInBits) {
    assert(SizeInBits < 65536 && OffsetInBits < 65536);
    SubRegisterSizeInBits = SizeInBits;
    SubRegisterOffsetInBits = OffsetInBits;
  }

  /// Add masking operations to stencil out a subregister.
  void maskSubRegister();

  /// Output a dwarf operand and an optional assembler comment.
  virtual void emitOp(uint8_t Op, const char *Comment = nullptr) = 0;

  /// Emit a raw signed value.
  virtual void emitSigned(int64_t Value) = 0;

  /// Emit a raw unsigned value.
  virtual void emitUnsigned(uint64_t Value) = 0;

  virtual void emitData1(uint8_t Value) = 0;

  virtual void emitBaseTypeRef(uint64_t Idx) = 0;

  /// Start emitting data to the temporary buffer. The data stored in the
  /// temporary buffer can be committed to the main output using
  /// commitTemporaryBuffer().
  virtual void enableTemporaryBuffer() = 0;

  /// Disable emission to the temporary buffer. This does not commit data
  /// in the temporary buffer to the main output.
  virtual void disableTemporaryBuffer() = 0;

  /// Return the emitted size, in number of bytes, for the data stored in the
  /// temporary buffer.
  virtual unsigned getTemporaryBufferSize() = 0;

  /// Commit the data stored in the temporary buffer to the main output.
  virtual void commitTemporaryBuffer() = 0;

  /// Emit a normalized unsigned constant.
  void emitConstu(uint64_t Value);

  /// Return whether the given machine register is the frame register in the
  /// current function.
  virtual bool isFrameRegister(const TargetRegisterInfo &TRI,
                               llvm::Register MachineReg) = 0;

  /// Emit a DW_OP_reg operation. Note that this is only legal inside a DWARF
  /// register location description.
  void addReg(int DwarfReg, const char *Comment = nullptr);

  /// Emit a DW_OP_breg operation.
  void addBReg(int DwarfReg, int Offset);

  /// Emit DW_OP_fbreg <Offset>.
  void addFBReg(int Offset);

  /// Emit a partial DWARF register operation.
  ///
  /// \param MachineReg           The register number.
  /// \param MaxSize              If the register must be composed from
  ///                             sub-registers this is an upper bound
  ///                             for how many bits the emitted DW_OP_piece
  ///                             may cover.
  ///
  /// If size and offset is zero an operation for the entire register is
  /// emitted: Some targets do not provide a DWARF register number for every
  /// register.  If this is the case, this function will attempt to emit a DWARF
  /// register by emitting a fragment of a super-register or by piecing together
  /// multiple subregisters that alias the register.
  ///
  /// \return false if no DWARF register exists for MachineReg.
  bool addMachineReg(const TargetRegisterInfo &TRI, llvm::Register MachineReg,
                     unsigned MaxSize = ~1U);

  /// Emit a DW_OP_piece or DW_OP_bit_piece operation for a variable fragment.
  /// \param OffsetInBits    This is an optional offset into the location that
  /// is at the top of the DWARF stack.
  void addOpPiece(unsigned SizeInBits, unsigned OffsetInBits = 0);

  /// Emit a shift-right dwarf operation.
  void addShr(unsigned ShiftBy);

  /// Emit a bitwise and dwarf operation.
  void addAnd(unsigned Mask);

  /// Emit a DW_OP_stack_value, if supported.
  ///
  /// The proper way to describe a constant value is DW_OP_constu <const>,
  /// DW_OP_stack_value.  Unfortunately, DW_OP_stack_value was not available
  /// until DWARF 4, so we will continue to generate DW_OP_constu <const> for
  /// DWARF 2 and DWARF 3. Technically, this is incorrect since DW_OP_const
  /// <const> actually describes a value at a constant address, not a constant
  /// value.  However, in the past there was no better way to describe a
  /// constant value, so the producers and consumers started to rely on
  /// heuristics to disambiguate the value vs. location status of the
  /// expression.  See PR21176 for more details.
  void addStackValue();

  /// Finalize an entry value by emitting its size operand, and committing the
  /// DWARF block which has been emitted to the temporary buffer.
  void finalizeEntryValue();

  /// Cancel the emission of an entry value.
  void cancelEntryValue();

  ~DwarfExpression() = default;

public:
  DwarfExpression(unsigned DwarfVersion, DwarfCompileUnit &CU)
      : CU(CU), SubRegisterSizeInBits(0), SubRegisterOffsetInBits(0),
        LocationKind(Unknown), SavedLocationKind(Unknown),
        LocationFlags(Unknown), DwarfVersion(DwarfVersion) {}

  /// This needs to be called last to commit any pending changes.
  void finalize();

  /// Emit a signed constant.
  void addSignedConstant(int64_t Value);

  /// Emit an unsigned constant.
  void addUnsignedConstant(uint64_t Value);

  /// Emit an unsigned constant.
  void addUnsignedConstant(const APInt &Value);

  /// Emit an floating point constant.
  void addConstantFP(const APFloat &Value, const AsmPrinter &AP);

  /// Lock this down to become a memory location description.
  void setMemoryLocationKind() {
    assert(isUnknownLocation());
    LocationKind = Memory;
  }

  /// Lock this down to become an entry value location.
  void setEntryValueFlags(const MachineLocation &Loc);

  /// Lock this down to become a call site parameter location.
  void setCallSiteParamValueFlag() { LocationFlags |= CallSiteParamValue; }

  /// Emit a machine register location. As an optimization this may also consume
  /// the prefix of a DwarfExpression if a more efficient representation for
  /// combining the register location and the first operation exists.
  ///
  /// \param FragmentOffsetInBits     If this is one fragment out of a
  /// fragmented
  ///                                 location, this is the offset of the
  ///                                 fragment inside the entire variable.
  /// \return                         false if no DWARF register exists
  ///                                 for MachineReg.
  bool addMachineRegExpression(const TargetRegisterInfo &TRI,
                               DIExpressionCursor &Expr,
                               llvm::Register MachineReg,
                               unsigned FragmentOffsetInBits = 0);

  /// Begin emission of an entry value dwarf operation. The entry value's
  /// first operand is the size of the DWARF block (its second operand),
  /// which needs to be calculated at time of emission, so we don't emit
  /// any operands here.
  void beginEntryValueExpression(DIExpressionCursor &ExprCursor);

  /// Return the index of a base type with the given properties and
  /// create one if necessary.
  unsigned getOrCreateBaseType(unsigned BitSize, dwarf::TypeKind Encoding);

  /// Emit all remaining operations in the DIExpressionCursor. The
  /// cursor must not contain any DW_OP_LLVM_arg operations.
  void addExpression(DIExpressionCursor &&Expr);

  /// Emit all remaining operations in the DIExpressionCursor.
  /// DW_OP_LLVM_arg operations are resolved by calling (\p InsertArg).
  //
  /// \return false if any call to (\p InsertArg) returns false.
  bool addExpression(
      DIExpressionCursor &&Expr,
      llvm::function_ref<bool(unsigned, DIExpressionCursor &)> InsertArg);

  /// If applicable, emit an empty DW_OP_piece / DW_OP_bit_piece to advance to
  /// the fragment described by \c Expr.
  void addFragmentOffset(const DIExpression *Expr);

  void emitLegacySExt(unsigned FromBits);
  void emitLegacyZExt(unsigned FromBits);

  /// Emit location information expressed via WebAssembly location + offset
  /// The Index is an identifier for locals, globals or operand stack.
  void addWasmLocation(unsigned Index, uint64_t Offset);
};

/// DwarfExpression implementation for .debug_loc entries.
class DebugLocDwarfExpression final : public DwarfExpression {

  struct TempBuffer {
    SmallString<32> Bytes;
    std::vector<std::string> Comments;
    BufferByteStreamer BS;

    TempBuffer(bool GenerateComments) : BS(Bytes, Comments, GenerateComments) {}
  };

  std::unique_ptr<TempBuffer> TmpBuf;
  BufferByteStreamer &OutBS;
  bool IsBuffering = false;

  /// Return the byte streamer that currently is being emitted to.
  ByteStreamer &getActiveStreamer() { return IsBuffering ? TmpBuf->BS : OutBS; }

  void emitOp(uint8_t Op, const char *Comment = nullptr) override;
  void emitSigned(int64_t Value) override;
  void emitUnsigned(uint64_t Value) override;
  void emitData1(uint8_t Value) override;
  void emitBaseTypeRef(uint64_t Idx) override;

  void enableTemporaryBuffer() override;
  void disableTemporaryBuffer() override;
  unsigned getTemporaryBufferSize() override;
  void commitTemporaryBuffer() override;

  bool isFrameRegister(const TargetRegisterInfo &TRI,
                       llvm::Register MachineReg) override;

public:
  DebugLocDwarfExpression(unsigned DwarfVersion, BufferByteStreamer &BS,
                          DwarfCompileUnit &CU)
      : DwarfExpression(DwarfVersion, CU), OutBS(BS) {}
};

/// DwarfExpression implementation for singular DW_AT_location.
class DIEDwarfExpression final : public DwarfExpression {
  const AsmPrinter &AP;
  DIELoc &OutDIE;
  DIELoc TmpDIE;
  bool IsBuffering = false;

  /// Return the DIE that currently is being emitted to.
  DIELoc &getActiveDIE() { return IsBuffering ? TmpDIE : OutDIE; }

  void emitOp(uint8_t Op, const char *Comment = nullptr) override;
  void emitSigned(int64_t Value) override;
  void emitUnsigned(uint64_t Value) override;
  void emitData1(uint8_t Value) override;
  void emitBaseTypeRef(uint64_t Idx) override;

  void enableTemporaryBuffer() override;
  void disableTemporaryBuffer() override;
  unsigned getTemporaryBufferSize() override;
  void commitTemporaryBuffer() override;

  bool isFrameRegister(const TargetRegisterInfo &TRI,
                       llvm::Register MachineReg) override;

public:
  DIEDwarfExpression(const AsmPrinter &AP, DwarfCompileUnit &CU, DIELoc &DIE);

  DIELoc *finalize() {
    DwarfExpression::finalize();
    return &OutDIE;
  }
};

class DwarfExprAST {
public:
  class Node {
  private:
    DIOp::Variant Element;
    // FIXME(KZHURAVL): Use pool/arena allocator instead of individual smart
    // pointers?
    SmallVector<std::unique_ptr<Node>> Children;

    bool IsLowered = false;
    Type *ResultType = nullptr;

  public:
    Node(DIOp::Variant Element)
        : Element(Element) {}

    const DIOp::Variant &getElement() const {
      return Element;
    }
    const SmallVector<std::unique_ptr<Node>> &getChildren() const {
      return Children;
    }

    DIOp::Variant &getElement() {
      return Element;
    }
    SmallVector<std::unique_ptr<Node>> &getChildren() {
      return Children;
    }

    const bool &isLowered() const {
      return IsLowered;
    }
    const Type *getResultType() const {
      return ResultType;
    }

    bool &isLowered() {
      return IsLowered;
    }
    Type *getResultType() {
      return ResultType;
    }

    void setIsLowered(bool IL = true) {
      IsLowered = IL;
    }
    void setResultType(Type *RT) {
      ResultType = RT;
    }

    size_t getChildrenCount() const;
  };

  const AsmPrinter &AP;
  // An `std::optional<const TargetRegisterInfo&>` where `nullptr` represents
  // `None`. Only present when in a function context.
  const TargetRegisterInfo *TRI;
  DwarfCompileUnit &CU;
  const DILifetime &Lifetime;
  // An `std::optional<MachineOperand>` where `nullptr` represents `None`.
  // Only present when in a function context.
  const MachineOperand *Referrer;
  // An `std::optional<const DenseMap<_, _>&>` where `nullptr` represents
  // `None`. Only present and applicable as part of an optimization for
  // DIFragments which refer to global variable fragments.
  const DwarfDebug::GVFragmentMapTy *GVFragmentMap;
  std::unique_ptr<DwarfExprAST::Node> Root;
  // FIXME(KZHURAVL): This is a temporary boolean variable that indicates
  // whether the lowering of this expression is supported or not. If the
  // lowering is supported, then a valid DIE is returned, otherwise an empty
  // DIE is returned (which indicates that there is no debug information
  // available).
  bool IsImplemented = true;

  void buildDIExprAST();

  /// Describes a kind of value on the DWARF expression stack. ValueKind::Value
  /// is a DWARF5-style value, and ValueKind::LocationDesc is a location
  /// description.
  enum class ValueKind {
    Value,
    LocationDesc,
  };

  /// The result of evaluating a DIExpr operation. Describes the value that the
  /// operation will push onto the DWARF expression stack.
  struct OpResult {
    Type *Ty;
    ValueKind VK;
  };

  /// Optionally emit DWARF operations to convert the value at the top of the
  /// stack to RequiredVK. Nop if Res.VK is RequiredVK.
  OpResult convertValueKind(const OpResult &Res, ValueKind RequiredVK);

  void readToValue(Type *Ty);
  void readToValue(Node *OpNode);

  using ChildrenT = ArrayRef<std::unique_ptr<DwarfExprAST::Node>>;

  /// Dispatch to a specific traverse() function, and convert the result to
  /// ReqVK if non-nullopt.
  std::optional<OpResult> traverse(Node *OpNode,
                                   std::optional<ValueKind> ReqVK);

  std::optional<OpResult> traverse(DIOp::Arg Arg, ChildrenT Children);
  std::optional<OpResult> traverse(DIOp::Constant Constant, ChildrenT Children);
  std::optional<OpResult> traverse(DIOp::PushLane PushLane, ChildrenT Children);
  std::optional<OpResult> traverse(DIOp::Referrer Referrer, ChildrenT Children);
  std::optional<OpResult> traverse(DIOp::TypeObject TypeObject,
                                   ChildrenT Children);
  std::optional<OpResult> traverse(DIOp::AddrOf AddrOf, ChildrenT Children);
  std::optional<OpResult> traverse(DIOp::Convert Convert, ChildrenT Children);
  std::optional<OpResult> traverse(DIOp::Deref Deref, ChildrenT Children);
  std::optional<OpResult> traverse(DIOp::Extend Extend, ChildrenT Children);
  std::optional<OpResult> traverse(DIOp::Read Read, ChildrenT Children);
  std::optional<OpResult> traverse(DIOp::Reinterpret Reinterpret,
                                   ChildrenT Children);
  std::optional<OpResult> traverse(DIOp::Select Select, ChildrenT Children);
  std::optional<OpResult> traverse(DIOp::Composite Composite,
                                   ChildrenT Children);

  std::optional<OpResult> traverseMathOp(uint8_t DwarfOp, ChildrenT Children);
  std::optional<OpResult> traverse(DIOp::Add Op, ChildrenT Children) {
    return traverseMathOp(dwarf::DW_OP_plus, Children);
  }
  std::optional<OpResult> traverse(DIOp::Div Op, ChildrenT Children) {
    return traverseMathOp(dwarf::DW_OP_div, Children);
  }
  std::optional<OpResult> traverse(DIOp::Mul Op, ChildrenT Children) {
    return traverseMathOp(dwarf::DW_OP_mul, Children);
  }
  std::optional<OpResult> traverse(DIOp::Shl Op, ChildrenT Children) {
    return traverseMathOp(dwarf::DW_OP_shl, Children);
  }
  std::optional<OpResult> traverse(DIOp::Shr Op, ChildrenT Children) {
    return traverseMathOp(dwarf::DW_OP_shr, Children);
  }
  std::optional<OpResult> traverse(DIOp::Sub Op, ChildrenT Children) {
    return traverseMathOp(dwarf::DW_OP_minus, Children);
  }

  std::optional<OpResult> traverse(DIOp::BitOffset BitOffset,
                                   ChildrenT Children);
  std::optional<OpResult> traverse(DIOp::ByteOffset ByteOffset,
                                   ChildrenT Children);

  void emitReg(int32_t DwarfReg, const char *Comment = nullptr);
  void emitSigned(int64_t SignedValue);
  void emitUnsigned(uint64_t UnsignedValue);
  virtual void emitDwarfData1(uint8_t Data1Value) = 0;
  virtual void emitDwarfOp(uint8_t DwarfOpValue,
                           const char *Comment = nullptr) = 0;
  virtual void emitDwarfSigned(int64_t SignedValue) = 0;
  virtual void emitDwarfUnsigned(uint64_t UnsignedValue) = 0;
  virtual void emitDwarfAddr(const MCSymbol *Sym) = 0;
  virtual void emitDwarfOpAddrx(unsigned Index) = 0;
  virtual void emitDwarfLabelDelta(const MCSymbol *Hi, const MCSymbol *Lo) = 0;

public:
  DwarfExprAST(const AsmPrinter &AP, const TargetRegisterInfo *TRI,
               DwarfCompileUnit &CU, const DILifetime &Lifetime,
               const MachineOperand *Referrer,
               const DwarfDebug::GVFragmentMapTy *GVFragmentMap)
      : AP(AP), TRI(TRI), CU(CU), Lifetime(Lifetime), Referrer(Referrer),
        GVFragmentMap(GVFragmentMap) {
    buildDIExprAST();
  }
  virtual ~DwarfExprAST() {}
};

class DebugLocDwarfExprAST final : DwarfExprAST {
  BufferByteStreamer &OutBS;

  ByteStreamer &getActiveStreamer();

  void emitDwarfData1(uint8_t Data1Value) override;
  void emitDwarfOp(uint8_t DwarfOpValue, const char *Comment = nullptr) override;
  void emitDwarfSigned(int64_t SignedValue) override;
  void emitDwarfUnsigned(uint64_t UnsignedValue) override;
  void emitDwarfAddr(const MCSymbol *Sym) override;
  void emitDwarfOpAddrx(unsigned Index) override;
  void emitDwarfLabelDelta(const MCSymbol *Hi, const MCSymbol *Lo) override;

  DebugLocDwarfExprAST(const AsmPrinter &AP, const TargetRegisterInfo *TRI,
                       DwarfCompileUnit &CU, BufferByteStreamer &BS,
                       const DILifetime &Lifetime,
                       const MachineOperand *Referrer,
                       const DwarfDebug::GVFragmentMapTy *GVFragmentMap)
      : DwarfExprAST(AP, TRI, CU, Lifetime, Referrer, GVFragmentMap),
        OutBS(BS) {}

public:
  DebugLocDwarfExprAST(const AsmPrinter &AP, const TargetRegisterInfo &TRI,
                       DwarfCompileUnit &CU, BufferByteStreamer &BS,
                       const DILifetime &Lifetime,
                       const MachineOperand &Referrer)
      : DebugLocDwarfExprAST(AP, &TRI, CU, BS, Lifetime, &Referrer, nullptr) {}
  DebugLocDwarfExprAST(const AsmPrinter &AP, DwarfCompileUnit &CU,
                       BufferByteStreamer &BS, const DILifetime &Lifetime,
                       const DwarfDebug::GVFragmentMapTy &GVFragmentMap)
      : DebugLocDwarfExprAST(AP, nullptr, CU, BS, Lifetime, nullptr,
                             &GVFragmentMap) {}
  DebugLocDwarfExprAST(const DebugLocDwarfExprAST &) = delete;
  ~DebugLocDwarfExprAST() {}

  bool finalize() {
    traverse(Root.get(), ValueKind::LocationDesc);
    return IsImplemented;
  }
};

// FIXME(KZHURAVL): Write documentation for DIEDwarfExprAST.
class DIEDwarfExprAST final : DwarfExprAST {
  DIELoc &OutDIE;

  DIELoc &getActiveDIE();

  void emitDwarfData1(uint8_t Data1Value) override;
  void emitDwarfOp(uint8_t DwarfOpValue, const char *Comment = nullptr) override;
  void emitDwarfSigned(int64_t SignedValue) override;
  void emitDwarfUnsigned(uint64_t UnsignedValue) override;
  void emitDwarfAddr(const MCSymbol *Sym) override;
  void emitDwarfOpAddrx(unsigned Index) override;
  void emitDwarfLabelDelta(const MCSymbol *Hi, const MCSymbol *Lo) override;

  DIEDwarfExprAST(const AsmPrinter &AP, const TargetRegisterInfo *TRI,
                  DwarfCompileUnit &CU, DIELoc &DIE, const DILifetime &Lifetime,
                  const MachineOperand *Referrer,
                  const DwarfDebug::GVFragmentMapTy *GVFragmentMap)
      : DwarfExprAST(AP, TRI, CU, Lifetime, Referrer, GVFragmentMap),
        OutDIE(DIE) {}

public:
  DIEDwarfExprAST(const AsmPrinter &AP, const TargetRegisterInfo &TRI,
                  DwarfCompileUnit &CU, DIELoc &DIE, const DILifetime &Lifetime,
                  const MachineOperand &Referrer)
      : DIEDwarfExprAST(AP, &TRI, CU, DIE, Lifetime, &Referrer, nullptr) {}
  DIEDwarfExprAST(const AsmPrinter &AP, DwarfCompileUnit &CU, DIELoc &DIE,
                  const DILifetime &Lifetime,
                  const DwarfDebug::GVFragmentMapTy &GVFragmentMap)
      : DIEDwarfExprAST(AP, nullptr, CU, DIE, Lifetime, nullptr,
                        &GVFragmentMap) {}
  DIEDwarfExprAST(const DIEDwarfExprAST &) = delete;
  ~DIEDwarfExprAST() {}

  DIELoc *finalize() {
    traverse(Root.get(), ValueKind::LocationDesc);
    return IsImplemented ? &OutDIE : nullptr;
  }
};

} // end namespace llvm

#endif // LLVM_LIB_CODEGEN_ASMPRINTER_DWARFEXPRESSION_H
