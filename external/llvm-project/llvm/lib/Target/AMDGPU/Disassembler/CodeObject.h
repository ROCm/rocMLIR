//===- CodeObject.hpp - ELF object file implementation ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the HSA Code Object file class.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGPU_DISASSEMBLER_HSA_CODE_OBJECT_HPP
#define AMDGPU_DISASSEMBLER_HSA_CODE_OBJECT_HPP

#include "AMDKernelCodeT.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Endian.h"

namespace llvm {

//===----------------------------------------------------------------------===//
// ELFNote
//===----------------------------------------------------------------------===//

struct amdgpu_hsa_code_object_version {
  support::ulittle32_t major_version;
  support::ulittle32_t minor_version;
};


struct amdgpu_hsa_isa {
  support::ulittle16_t vendor_name_size;
  support::ulittle16_t architecture_name_size;
  support::ulittle32_t major;
  support::ulittle32_t minor;
  support::ulittle32_t stepping;
  char names[1];

  StringRef getVendorName() const {
    return StringRef(names, vendor_name_size - 1);
  }

  StringRef getArchitectureName() const {
    return StringRef(names + vendor_name_size, architecture_name_size - 1);
  }
};

struct ELFNote {
  support::ulittle32_t namesz;
  support::ulittle32_t descsz;
  support::ulittle32_t type;

  enum {ALIGN = 4};

  ELFNote() = delete;
  ELFNote(const ELFNote&) = delete;
  ELFNote& operator =(const ELFNote&) = delete;

  StringRef getName() const {
    return StringRef(reinterpret_cast<const char*>(this) + sizeof(*this), namesz);
  }

  StringRef getDesc() const {
    return StringRef(getName().data() + alignTo(namesz, ALIGN), descsz);
  }

  size_t getSize() const {
    return sizeof(*this) + alignTo(namesz, ALIGN) + alignTo(descsz, ALIGN);
  }

  template <typename D> Expected<const D*> as() const {
    if (descsz < sizeof(D)) {
      return make_error<StringError>("invalid descsz",
                                     object::object_error::parse_failed);
    }

    return reinterpret_cast<const D*>(getDesc().data());
  }
};

const ELFNote* getNext(const ELFNote &N);


template <typename Item>
class const_varsize_item_iterator :
  std::iterator<std::forward_iterator_tag, const Item, void> {
  ArrayRef<uint8_t> Ref;

  const Item *item() const {
    return reinterpret_cast<const Item*>(Ref.data());
  }

  size_t getItemPadSize() const {
    assert(Ref.size() >= sizeof(Item));
    return (const uint8_t*)getNext(*item()) - (const uint8_t*)item();
  }

public:
  const_varsize_item_iterator() {}
  const_varsize_item_iterator(ArrayRef<uint8_t> Ref_) : Ref(Ref_) {}

  bool valid() const {
    return Ref.size() >= sizeof(Item) && Ref.size() >= getItemPadSize();
  }

  Expected<const Item&> operator*() const {
    if (!valid()) {
      return make_error<StringError>("invalid item",
                                     object::object_error::parse_failed);
    }

    return *item();
  }

  bool operator==(const const_varsize_item_iterator &Other) const {
    return (Ref.size() == Other.Ref.size()) &&
           (Ref.empty() || Ref.data() == Other.Ref.data());
  }

  bool operator!=(const const_varsize_item_iterator &Other) const {
    return !(*this == Other);
  }

  const_varsize_item_iterator &operator++() { // preincrement
    Ref = Ref.size() >= sizeof(Item) ?
      Ref.slice((std::min)(getItemPadSize(), Ref.size())) :
      decltype(Ref)();
    return *this;
  }
};

//===----------------------------------------------------------------------===//
// FunctionSym
//===----------------------------------------------------------------------===//

class HSACodeObject;

class FunctionSym : public object::ELF64LEObjectFile::Elf_Sym {
public:
  Expected<uint64_t> getAddress(const HSACodeObject *CodeObject) const;

  Expected<uint64_t> getAddress(
    const HSACodeObject *CodeObject,
    const object::ELF64LEObjectFile::Elf_Shdr *Text) const;

  Expected<uint64_t> getSectionOffset(const HSACodeObject *CodeObject) const;

  Expected<uint64_t> getSectionOffset(
    const HSACodeObject *CodeObject,
    const object::ELF64LEObjectFile::Elf_Shdr *Text) const;

  Expected<uint64_t> getCodeOffset(
    const HSACodeObject *CodeObject,
    const object::ELF64LEObjectFile::Elf_Shdr *Text) const;

  static Expected<const FunctionSym *>
  asFunctionSym(Expected<const object::ELF64LEObjectFile::Elf_Sym *> Sym);
};

class KernelSym : public FunctionSym {
public:
  Expected<uint64_t>
  getCodeOffset(const HSACodeObject *CodeObject,
                const object::ELF64LEObjectFile::Elf_Shdr *Text) const;

  Expected<const amd_kernel_code_t *>
  getAmdKernelCodeT(const HSACodeObject *CodeObject) const;

  Expected<const amd_kernel_code_t *>
  getAmdKernelCodeT(const HSACodeObject *CodeObject,
                    const object::ELF64LEObjectFile::Elf_Shdr *Text) const;

  static Expected<const KernelSym *> asKernelSym(const FunctionSym *Sym);
};

template <typename BaseIterator>
class conditional_iterator : public iterator_adaptor_base<
                                              conditional_iterator<BaseIterator>,
                                              BaseIterator,
                                              std::forward_iterator_tag> {
  
public:
  typedef std::function<
    bool(const typename conditional_iterator::iterator_adaptor_base::value_type&)
  > PredicateTy;
  
protected:
  BaseIterator End;
  PredicateTy Predicate;

public:

  conditional_iterator(BaseIterator BI, BaseIterator E, PredicateTy P)
    : conditional_iterator::iterator_adaptor_base(BI), End(E), Predicate(P) {
    while (this->I != End && !Predicate(*this->I)) {
      ++this->I;
    } 
  }

  conditional_iterator &operator++() {
    do {
      ++this->I;
    } while (this->I != End && !Predicate(*this->I));
    return *this;
  }
};

class function_sym_iterator
    : public conditional_iterator<object::elf_symbol_iterator> {
public:
  function_sym_iterator(object::elf_symbol_iterator It,
                        object::elf_symbol_iterator End, PredicateTy P)
      : conditional_iterator<object::elf_symbol_iterator>(It, End, P) {}

  const object::ELFSymbolRef &operator*() const {
    return *I;
  }
};

//===----------------------------------------------------------------------===//
// HSACodeObject
//===----------------------------------------------------------------------===//

class HSACodeObject : public object::ELF64LEObjectFile {
private:
  mutable SmallVector<uint64_t, 8> FunctionMarkers;

  void InitMarkers() const;

  HSACodeObject(object::ELF64LEObjectFile &&Obj)
    : object::ELF64LEObjectFile(std::move(Obj)) {
    InitMarkers();
  }

public:
  static Expected<std::unique_ptr<HSACodeObject>>
  create(MemoryBufferRef Wrapper) {
    auto Obj = object::ELF64LEObjectFile::create(Wrapper);
    if (auto E = Obj.takeError())
      return std::move(E);
    std::unique_ptr<HSACodeObject> Ret(new HSACodeObject(std::move(*Obj)));
    return std::move(Ret);
  }

  typedef const_varsize_item_iterator<ELFNote> note_iterator;

  note_iterator notes_begin() const;
  note_iterator notes_end() const;
  iterator_range<note_iterator> notes() const;

  function_sym_iterator functions_begin() const;
  function_sym_iterator functions_end() const;
  iterator_range<function_sym_iterator> functions() const;

  Expected<ArrayRef<uint8_t>> getCode(const FunctionSym *Function) const;

  Expected<const Elf_Shdr *> getSectionByName(StringRef Name) const;

  Expected<uint32_t> getSectionIdxByName(StringRef) const;
  Expected<uint32_t> getTextSectionIdx() const;
  Expected<uint32_t> getNoteSectionIdx() const;
  Expected<const Elf_Shdr *> getTextSection() const;
  Expected<const Elf_Shdr *> getNoteSection() const;

  friend class FunctionSym;
  friend class KernelSym;
};

} // namespace llvm

#endif
