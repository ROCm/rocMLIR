// RUN: rocmlir-opt --split-input-file --verify-diagnostics %s -o /dev/null

// COM: Exercises the error/return-{} branches of the custom parsers in
// COM: external/mlir-hal/lib/Dialect/MHAL/IR/MHAL.cpp for the two attributes
// COM: with hand-written parse(): TargetObjectAttr::parse and
// COM: KernelPackageAttr::parse. Each section under --split-input-file is
// COM: parsed independently and is expected to emit exactly one diagnostic,
// COM: matched by the // expected-error directive.
// COM:
// COM: Custom MHAL.cpp diagnostics ("expected a name of a known target
// COM: object type" / "expected a name of a known target type") are
// COM: emitted via parser.emitError(). All other expected messages come
// COM: from upstream MLIR's AsmParser methods (parseLess/Greater/Equal/
// COM: Arrow/Colon, parseKeywordOrString, parseInteger, parseAttribute).

// COM: ============================================================
// COM: TargetObjectAttr -- one section per parse() error branch
// COM: ============================================================

// COM: ---- parseLess() -- missing leading '<'.
// expected-error @+1 {{expected '<'}}
func.func @target_obj_missing_less() attributes {x = #mhal.target_obj} { return }

// -----

// COM: ---- parseKeywordOrString(&typeName) fails on a non-keyword token
// COM: (here an integer literal in the type slot).
// expected-error @+1 {{expected valid keyword or string}}
func.func @target_obj_bad_type_token() attributes {x = #mhal.target_obj<123 = "x" -> "B">} { return }

// -----

// COM: ---- getTargetObjectTypeForName(typeName) returns nullopt;
// COM: emitError(typeLoc, "expected a name of a known target object type").
// expected-error @+1 {{expected a name of a known target object type}}
func.func @target_obj_unknown_type() attributes {x = #mhal.target_obj<NOTATYPE = "x" -> "B">} { return }

// -----

// COM: ---- parseEqual() -- missing '=' between type and arch.
// expected-error @+1 {{expected '='}}
func.func @target_obj_missing_equal() attributes {x = #mhal.target_obj<ELF "x" -> "B">} { return }

// -----

// COM: ---- parseKeywordOrString(&archName) fails on a non-keyword token
// COM: in the arch slot.
// expected-error @+1 {{expected valid keyword or string}}
func.func @target_obj_bad_arch_token() attributes {x = #mhal.target_obj<ELF = 123 -> "B">} { return }

// -----

// COM: ---- parseArrow() -- missing '->' between arch (and optional attr
// COM: dict) and the binary attribute.
// expected-error @+1 {{expected '->'}}
func.func @target_obj_missing_arrow() attributes {x = #mhal.target_obj<ELF = "gfx90a" "B">} { return }

// -----

// COM: ---- parseAttribute(binary) -- '->' present but no attribute value
// COM: follows it.
// expected-error @+1 {{expected attribute value}}
func.func @target_obj_missing_binary() attributes {x = #mhal.target_obj<ELF = "gfx90a" -> >} { return }

// -----

// COM: ---- parseGreater() -- missing closing '>'. Use a stray comma so
// COM: the lexer doesn't roll the next token into the binary attr.
// expected-error @+1 {{expected '>'}}
func.func @target_obj_missing_greater() attributes {x = #mhal.target_obj<ELF = "gfx90a" -> "B" ,>} { return }

// -----

// COM: ============================================================
// COM: KernelPackageAttr -- one section per parse() error branch
// COM: ============================================================

// COM: ---- parseLess() -- missing leading '<'.
// expected-error @+1 {{expected '<'}}
func.func @kernel_pkg_missing_less() attributes {x = #mhal.kernel_pkg} { return }

// -----

// COM: ---- parseKeywordOrString(&typeName) fails on a non-keyword token
// COM: in the type slot.
// expected-error @+1 {{expected valid keyword or string}}
func.func @kernel_pkg_bad_type_token() attributes {x = #mhal.kernel_pkg<123 = "x" : k [16, 64] -> #mhal.target_obj<ELF = "gfx90a" -> "B">>} { return }

// -----

// COM: ---- getTargetTypeForName(typeName) returns nullopt;
// COM: emitError(typeLoc, "expected a name of a known target type").
// expected-error @+1 {{expected a name of a known target type}}
func.func @kernel_pkg_unknown_type() attributes {x = #mhal.kernel_pkg<NOPE = "x" : k [16, 64] -> #mhal.target_obj<ELF = "gfx90a" -> "B">>} { return }

// -----

// COM: ---- parseEqual() -- missing '=' between type and target.
// expected-error @+1 {{expected '='}}
func.func @kernel_pkg_missing_equal() attributes {x = #mhal.kernel_pkg<GPU "x" : k [16, 64] -> #mhal.target_obj<ELF = "gfx90a" -> "B">>} { return }

// -----

// COM: ---- parseKeywordOrString(&targetName) fails on a non-keyword token
// COM: in the target slot.
// expected-error @+1 {{expected valid keyword or string}}
func.func @kernel_pkg_bad_target_token() attributes {x = #mhal.kernel_pkg<GPU = 123 : k [16, 64] -> #mhal.target_obj<ELF = "gfx90a" -> "B">>} { return }

// -----

// COM: ---- parseColon() -- missing ':' between target and entry.
// expected-error @+1 {{expected ':'}}
func.func @kernel_pkg_missing_colon() attributes {x = #mhal.kernel_pkg<GPU = "gfx90a" k [16, 64] -> #mhal.target_obj<ELF = "gfx90a" -> "B">>} { return }

// -----

// COM: ---- parseKeywordOrString(&entryName) fails on a non-keyword token
// COM: in the entry-name slot.
// expected-error @+1 {{expected valid keyword or string}}
func.func @kernel_pkg_bad_entry_token() attributes {x = #mhal.kernel_pkg<GPU = "gfx90a" : 123 [16, 64] -> #mhal.target_obj<ELF = "gfx90a" -> "B">>} { return }

// -----

// COM: ---- parseAndGather<unsigned>(...) wraps parser.parseInteger(out);
// COM: a non-integer launch-dim entry triggers "expected integer value".
// expected-error @+1 {{expected integer value}}
func.func @kernel_pkg_bad_launch_dim() attributes {x = #mhal.kernel_pkg<GPU = "gfx90a" : k [16, "B"] -> #mhal.target_obj<ELF = "gfx90a" -> "B">>} { return }

// -----

// COM: ---- parseArrow() -- missing '->' between launch dims (and optional
// COM: attr dict) and the object attribute.
// expected-error @+1 {{expected '->'}}
func.func @kernel_pkg_missing_arrow() attributes {x = #mhal.kernel_pkg<GPU = "gfx90a" : k [16, 64] #mhal.target_obj<ELF = "gfx90a" -> "B">>} { return }

// -----

// COM: ---- parseAttribute(object) -- '->' present but no attribute value
// COM: follows it.
// expected-error @+1 {{expected attribute value}}
func.func @kernel_pkg_missing_object() attributes {x = #mhal.kernel_pkg<GPU = "gfx90a" : k [16, 64] -> >} { return }

// -----

// COM: ---- parseGreater() -- missing closing '>'.
// expected-error @+1 {{expected '>'}}
func.func @kernel_pkg_missing_greater() attributes {x = #mhal.kernel_pkg<GPU = "gfx90a" : k [16, 64] -> #mhal.target_obj<ELF = "gfx90a" -> "B"> ,>} { return }
