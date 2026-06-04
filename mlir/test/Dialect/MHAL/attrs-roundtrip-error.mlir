// RUN: rocmlir-opt --split-input-file --verify-diagnostics %s -o /dev/null

// COM: Exercises the error/return-{} branches of the custom parsers in
// COM: external/mlir-hal/lib/Dialect/MHAL/IR/MHAL.cpp for the two attributes
// COM: with hand-written parse(): TargetObjectAttr (lines 54-105) and
// COM: KernelPackageAttr (lines 147-213). Each section under
// COM: --split-input-file is parsed independently and is expected to emit
// COM: exactly one diagnostic, matched by the // expected-error directive.
// COM:
// COM: Custom MHAL.cpp diagnostics ("expected a name of a known target
// COM: object type" / "expected a name of a known target type") are
// COM: emitted via parser.emitError(). All other expected messages come
// COM: from upstream MLIR's AsmParser methods (parseLess/Greater/Equal/
// COM: Arrow/Colon, parseKeywordOrString, parseInteger, parseAttribute).

// COM: ============================================================
// COM: TargetObjectAttr -- one section per parse() error branch
// COM: ============================================================

// COM: ---- L57-58: parseLess() -- missing leading '<'.
// expected-error @+1 {{expected '<'}}
func.func @target_obj_missing_less() attributes {x = #mhal.target_obj} { return }

// -----

// COM: ---- L62-64: parseKeywordOrString(&typeName) fails on a non-keyword
// COM: token (here an integer literal in the type slot).
// expected-error @+1 {{expected valid keyword or string}}
func.func @target_obj_bad_type_token() attributes {x = #mhal.target_obj<123 = "x" -> "B">} { return }

// -----

// COM: ---- L69-72: getTargetObjectTypeForName(typeName) returns nullopt;
// COM: emitError(typeLoc, "expected a name of a known target object type").
// expected-error @+1 {{expected a name of a known target object type}}
func.func @target_obj_unknown_type() attributes {x = #mhal.target_obj<NOTATYPE = "x" -> "B">} { return }

// -----

// COM: ---- L74-76: parseEqual() -- missing '=' between type and arch.
// expected-error @+1 {{expected '='}}
func.func @target_obj_missing_equal() attributes {x = #mhal.target_obj<ELF "x" -> "B">} { return }

// -----

// COM: ---- L79-81: parseKeywordOrString(&archName) fails on a non-keyword
// COM: token in the arch slot.
// expected-error @+1 {{expected valid keyword or string}}
func.func @target_obj_bad_arch_token() attributes {x = #mhal.target_obj<ELF = 123 -> "B">} { return }

// -----

// COM: ---- L89-91: parseArrow() -- missing '->' between arch (and optional
// COM: attr dict) and the binary attribute.
// expected-error @+1 {{expected '->'}}
func.func @target_obj_missing_arrow() attributes {x = #mhal.target_obj<ELF = "gfx90a" "B">} { return }

// -----

// COM: ---- L94-96: parseAttribute(binary) -- '->' present but no attribute
// COM: value follows it.
// expected-error @+1 {{expected attribute value}}
func.func @target_obj_missing_binary() attributes {x = #mhal.target_obj<ELF = "gfx90a" -> >} { return }

// -----

// COM: ---- L99-100: parseGreater() -- missing closing '>'. Use a stray
// COM: comma so the lexer doesn't roll the next token into the binary attr.
// expected-error @+1 {{expected '>'}}
func.func @target_obj_missing_greater() attributes {x = #mhal.target_obj<ELF = "gfx90a" -> "B" ,>} { return }

// -----

// COM: ============================================================
// COM: KernelPackageAttr -- one section per parse() error branch
// COM: ============================================================

// COM: ---- L151-152: parseLess() -- missing leading '<'.
// expected-error @+1 {{expected '<'}}
func.func @kernel_pkg_missing_less() attributes {x = #mhal.kernel_pkg} { return }

// -----

// COM: ---- L156-157: parseKeywordOrString(&typeName) fails on a non-keyword
// COM: token in the type slot.
// expected-error @+1 {{expected valid keyword or string}}
func.func @kernel_pkg_bad_type_token() attributes {x = #mhal.kernel_pkg<123 = "x" : k [16, 64] -> #mhal.target_obj<ELF = "gfx90a" -> "B">>} { return }

// -----

// COM: ---- L162-164: getTargetTypeForName(typeName) returns nullopt;
// COM: emitError(typeLoc, "expected a name of a known target type").
// expected-error @+1 {{expected a name of a known target type}}
func.func @kernel_pkg_unknown_type() attributes {x = #mhal.kernel_pkg<NOPE = "x" : k [16, 64] -> #mhal.target_obj<ELF = "gfx90a" -> "B">>} { return }

// -----

// COM: ---- L167-168: parseEqual() -- missing '=' between type and target.
// expected-error @+1 {{expected '='}}
func.func @kernel_pkg_missing_equal() attributes {x = #mhal.kernel_pkg<GPU "x" : k [16, 64] -> #mhal.target_obj<ELF = "gfx90a" -> "B">>} { return }

// -----

// COM: ---- L172-173: parseKeywordOrString(&targetName) fails on a non-
// COM: keyword token in the target slot.
// expected-error @+1 {{expected valid keyword or string}}
func.func @kernel_pkg_bad_target_token() attributes {x = #mhal.kernel_pkg<GPU = 123 : k [16, 64] -> #mhal.target_obj<ELF = "gfx90a" -> "B">>} { return }

// -----

// COM: ---- L176-177: parseColon() -- missing ':' between target and entry.
// expected-error @+1 {{expected ':'}}
func.func @kernel_pkg_missing_colon() attributes {x = #mhal.kernel_pkg<GPU = "gfx90a" k [16, 64] -> #mhal.target_obj<ELF = "gfx90a" -> "B">>} { return }

// -----

// COM: ---- L181-182: parseKeywordOrString(&entryName) fails on a non-
// COM: keyword token in the entry-name slot.
// expected-error @+1 {{expected valid keyword or string}}
func.func @kernel_pkg_bad_entry_token() attributes {x = #mhal.kernel_pkg<GPU = "gfx90a" : 123 [16, 64] -> #mhal.target_obj<ELF = "gfx90a" -> "B">>} { return }

// -----

// COM: ---- L184-190: parseAndGather<unsigned>(...) wraps
// COM: parser.parseInteger(out); a non-integer launch-dim entry triggers
// COM: "expected integer value".
// expected-error @+1 {{expected integer value}}
func.func @kernel_pkg_bad_launch_dim() attributes {x = #mhal.kernel_pkg<GPU = "gfx90a" : k [16, "B"] -> #mhal.target_obj<ELF = "gfx90a" -> "B">>} { return }

// -----

// COM: ---- L199-200: parseArrow() -- missing '->' between launch dims (and
// COM: optional attr dict) and the object attribute.
// expected-error @+1 {{expected '->'}}
func.func @kernel_pkg_missing_arrow() attributes {x = #mhal.kernel_pkg<GPU = "gfx90a" : k [16, 64] #mhal.target_obj<ELF = "gfx90a" -> "B">>} { return }

// -----

// COM: ---- L204-205: parseAttribute(object) -- '->' present but no
// COM: attribute value follows it.
// expected-error @+1 {{expected attribute value}}
func.func @kernel_pkg_missing_object() attributes {x = #mhal.kernel_pkg<GPU = "gfx90a" : k [16, 64] -> >} { return }

// -----

// COM: ---- L208-209: parseGreater() -- missing closing '>'.
// expected-error @+1 {{expected '>'}}
func.func @kernel_pkg_missing_greater() attributes {x = #mhal.kernel_pkg<GPU = "gfx90a" : k [16, 64] -> #mhal.target_obj<ELF = "gfx90a" -> "B"> ,>} { return }
