# How the Quick-Tuning List Is Generated

This document explains, in detail, how the **quick-tuning perfconfig list** is produced and consumed in rocMLIR. The quick-tuning list is the small set of pre-selected `perfConfig` strings that the compiler tries when the user has *not* supplied a full tuning database. It is the compromise between "use a single heuristic" and "tune every kernel":

- A full tuning database (from `tuningRunner.py`) has a winning config for **every specific problem shape** the user tunes.
- A quick-tune list has a **small, fixed list of configs per `(arch, op, dtype)`** that together cover the majority of problems reasonably well.

The quick-tune list lives in a generated header: `mlir/include/mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc`

It is produced offline by a Python script that solves a **set-cover optimization problem** over tuning data.

---

## 1. High-level pipeline

The pipeline is linear (no wide ASCII diagram needed here). The flow has three stages, each implemented in a different place:

1. **Collect data** — `mlir/utils/performance/tuningRunner.py` runs an exhaustive or space-limited tuning sweep on real hardware and writes one row per `(problem, perfConfig)` pair — including the measured `TFlops` — to a `<output>.debug` TSV file.
2. **Reduce to a covering set** — `mlir/utils/performance/analysis/quickTuningGen.py` loads those TSVs, and for each `(arch, op, dtype)` solves a set-cover ILP to find the **minimum number of perfconfigs** such that every problem is "well covered" by at least one of them.
3. **Emit the header** — the same script rewrites sections inside `QuickTuningPerfconfigs.inc`. The compiler `#include`s that file in three modes to:
   - declare `initParameters*[]` arrays inside `PopulateParams*` classes,
   - define those arrays in the `.cpp` files,
   - populate a `StringRef → ArrayRef<StringRef>` lookup table keyed by `"<arch>_<op>_<dtype>"`.

---

## 2. Stage 1 — Collecting tuning data (`tuningRunner.py`)

The input to the generator is a set of `.debug` TSV files produced by the tuning runner.

Key pieces in `mlir/utils/performance/tuningRunner.py`:

- The script drives `rocmlir-gen` and `rocmlir-tuning-driver` for each `testVector` (i.e. problem configuration). For each problem it enumerates all `perfConfig`s in the configured tuning space and measures each one on GPU.
- `find_best_perfconfig()` parses the tuning driver output and builds a list of per-config rows; every row contains the problem fields, the `PerfConfig` string, and the measured `TFlops`.
- When the user passes `--debug`, a `DebugFileWriter` appends those rows to `<output>.debug` in TSV form:

```946:973:mlir/utils/performance/tuningRunner.py
class DebugFileWriter:
    """Context manager for writing debug entries to TSV file."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
        self._header_written = False

    def __enter__(self):
        self.file = open(self.filepath, 'a')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.close()

    def write_result(self, result: TuningResult):
        if not result.success:
            raise ValueError("write_result called with unsuccessful result")
        if not result.entries:
            raise ValueError("write_result called without entries")
        pd.DataFrame(result.entries).to_csv(
            self.file, sep='\t', header=not self._header_written, index=False)
        self.file.flush()
        self._header_written = True
```

The resulting `.debug` file therefore contains, for every tuned problem, **one row per `perfConfig`** that the tuning driver tried, along with its `TFlops`, `Chip` (e.g. `gfx942`), `DataType` (e.g. `f16`), and the problem dimensions (`M,K,N,G,...` for GEMM or layout/shape fields for Conv / Attention).

That per-config granularity is exactly what the generator needs — without it we would only know the single winning config per problem, not which alternative configs are also "close to optimal".

---

## 3. Stage 2 — Reducing to a covering set (`quickTuningGen.py`)

Everything in this stage lives in `mlir/utils/performance/analysis/quickTuningGen.py`. The core idea is:

> For each `(arch, dtype)` pick the **smallest** set of `perfConfig` strings such that, for every tuned problem, at least one selected config achieves within `threshold` of that problem's best measured TFlops.

That is precisely the classic **Set Cover** problem, solved here as a 0/1 Integer Linear Program with `pulp` + the CBC solver.

### 3.1 Loading and cleaning

`load_data()` reads all `.debug` files (or stdin) via pandas and concatenates them:

```135:159:mlir/utils/performance/analysis/quickTuningGen.py
def load_data(files, no_splitk):
    """Load tuning data from files or stdin."""
    if files:
        validate_files(files)
        print(f"Processing {len(files)} file(s):")
        for f in files:
            print(f"  {f}")
        dfs = [pd.read_csv(f, sep='\t', index_col=None) for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        print("Reading from stdin...")
        df = pd.read_csv(sys.stdin, sep='\t', index_col=None)

    if no_splitk and not df.empty:
        before = len(df)
        mask = df['PerfConfig'].apply(lambda x: get_splitk_value(x) in (None, '1'))
        df = df[mask]
        if len(df) < before:
            print(f"Filtered out {before - len(df)} out of {before} Split-K configs")

    return df
```

The `--no-splitk` flag uses `parse_perfconfig()` / `get_splitk_value()` to strip Split-K configs. Those functions understand the perfconfig string format (`<format>:v<version>:<csv params>`) and know where the Split-K field sits per version, which is useful when a backend cannot use Split-K yet.

### 3.2 Identifying a "problem"

Each operation type defines which columns uniquely identify a problem shape:

```18:26:mlir/utils/performance/analysis/quickTuningGen.py
GEMM_COLUMNS = ['TransA', 'TransB', 'G', 'M', 'K', 'N']
CONV_COLUMNS = [
    'Direction', 'FilterLayout', 'InputLayout', 'OutputLayout', 'N', 'C', 'H', 'W', 'K', 'Y', 'X',
    'DilationH', 'DilationW', 'StrideH', 'StrideW', 'PaddingH', 'PaddingW',
]
ATTENTION_COLUMNS = [
    'TransQ', 'TransK', 'TransV', 'TransO', 'Causal', 'ReturnLSE', 'SplitKV', 'WithAttnScale',
    'WithAttnBias', 'G', 'SeqLenQ', 'SeqLenK', 'NumHeadsQ', 'NumHeadsKV', 'HeadDimQK', 'HeadDimV',
]
```

Everything outside these "problem columns" (chip, dtype, perfConfig, TFlops, measurements, etc.) is treated as metadata.

### 3.3 Building the coverage relation

For every `(arch, dtype)` combination the function `find_perfconfigs()`:

1. Groups rows by `(problem, PerfConfig)` and keeps only the **best TFlops** seen for that pair — multiple runs of the same config on the same problem collapse into one number.
2. For each problem, finds `max_tflops` and the set of configs whose TFlops is at least `threshold * max_tflops` (default `0.93`). Those configs are the candidates that "cover" that problem.
3. Builds a binary coverage matrix `M[i, j] = 1` iff config `j` covers problem `i`.

### 3.4 Solving Set Cover as an ILP

The ILP is stated directly in the source:

```python
# Excerpt from quickTuningGen.py (line breaks adjusted for preview; see repo for exact text)
def find_perfconfigs(df, op, threshold):
    """Minimal covering set via set cover (ILP). Eligible configs: TFlops >= threshold * best_tflops."""
    target_cols = get_target_columns(op)
    results = {}

    for dtype in sorted(df['DataType'].unique()):
        df_typed = df[df['DataType'] == dtype]
        df_typed = df_typed.groupby(target_cols + ['PerfConfig'], as_index=False)['TFlops'].max()

        coverage = {}
        for name, group in df_typed.groupby(target_cols):
            max_tflops = group['TFlops'].max()
            good = group['TFlops'] >= max_tflops * threshold
            coverage[name] = group[good]['PerfConfig'].tolist()

        problems = sorted(coverage.keys())
        configs = sorted({c for cs in coverage.values() for c in cs})
        config_idx = {c: i for i, c in enumerate(configs)}
        n_problems, n_configs = len(problems), len(configs)

        matrix = np.zeros((n_problems, n_configs), dtype=int)
        for i, prob in enumerate(problems):
            for cfg in coverage[prob]:
                matrix[i, config_idx[cfg]] = 1

        prob = pulp.LpProblem("SetCover", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("x", range(n_configs), cat='Binary')
        prob += pulp.lpSum(x[j] for j in range(n_configs))
        for i in range(n_problems):
            prob += pulp.lpSum(matrix[i, j] * x[j] for j in range(n_configs)) >= 1

        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if status != pulp.LpStatusOptimal:
            status_name = pulp.LpStatus.get(status, "Unknown")
            raise RuntimeError(
                f"Set cover failed for {dtype}: {status_name}. "
                f"This likely indicates corrupted input data or a bug.")

        selected = [configs[j] for j in range(n_configs) if x[j].varValue == 1]
        counts = {c: sum(matrix[i, config_idx[c]] for i in range(n_problems)) for c in selected}
        results[dtype] = sorted(selected, key=lambda c: counts[c], reverse=True)

    return results
```

Two important properties of the output:

- **Order matters**: selected configs are sorted by **how many problems they cover** (descending). The compiler tries them in order, so the first entries of each generated array should be the most broadly useful configs — this tends to reduce compile time when only a partial walk through the list is performed.
- **If the ILP is infeasible or non-optimal** (e.g. some problem has zero covering configs because its data is corrupted), the script aborts rather than silently emitting bad output.

### 3.5 Coverage threshold, the main knob

The single most important parameter is `--th` (default `0.93`):

- Higher threshold (e.g. `0.98`) → fewer configs per problem count as "good" → more unique configs selected → larger, more precise quick-tune list, higher JIT/tuning-walk cost.
- Lower threshold (e.g. `0.90`) → more configs count as "good" → more overlap → smaller quick-tune list but potentially lower average quality.

### 3.6 Classifying the instruction family

The generator also decides which class / array symbol to emit to, using the arch + dtype + op triple:

```36:62:mlir/utils/performance/analysis/quickTuningGen.py
def get_instruction_type(arch, dtype, op):
    """Determine instruction type based on architecture, data type, and operation."""
    if op == "attention":
        return "GemmGemm"
    if arch.startswith("gfx9"):
        return "XDL"
    elif arch.startswith("gfx1") and dtype != "f32":
        return "Wmma"
    return "NonAccel"

def is_accel(arch, dtype, op):
    """Check if this combination uses accelerated instructions."""
    return get_instruction_type(arch, dtype, op) != "NonAccel"

def get_class_name(arch, dtype, op):
    """Get the PopulateParams class name."""
    instr = get_instruction_type(arch, dtype, op)
    return f"PopulateParams{instr}" if instr != "NonAccel" else "PopulateParams"

def get_param_names(arch, dtype, op):
    """Generate array and count variable names."""
    base = f"initParameters{dtype.capitalize()}{op.capitalize()}{arch.capitalize()}"
    return base, f"n{base[0].upper()}{base[1:]}"
```

So a `(gfx942, gemm, f16)` triple becomes:

- instr = `XDL`
- class = `PopulateParamsXDL`
- array = `initParametersF16GemmGfx942`
- count = `NInitParametersF16GemmGfx942`
- section key used in the `.inc` file = `GEMM_XDL_f16_gfx942`

---

## 4. Stage 3 — Emitting `QuickTuningPerfconfigs.inc`

The generator rewrites `mlir/include/mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc` by surgical text replacement on regions delimited by comment markers. Each `(instr, dtype, arch, op)` produces **three pieces** of output — and all three live in one file but are activated by different preprocessor macros:

1. **Definition of the array** (guarded by `{instr}_DEFINITIONS_GEN`):

   ```c
   // BEGIN_GEMM_XDL_f16_gfx942_DEFS
   const StringRef PopulateParamsXDL::initParametersF16GemmGfx942[] = {
       "v4:..."[,]
       ...
   };
   // END_GEMM_XDL_f16_gfx942_DEFS
   ```

2. **Declaration of the array** inside the class body (guarded by `{instr}_DECLARATIONS_GEN`):

   ```c
   // BEGIN_GEMM_XDL_f16_gfx942_DECS
   static constexpr size_t NInitParametersF16GemmGfx942 = N;
   static const StringRef initParametersF16GemmGfx942[NInitParametersF16GemmGfx942];
   // END_GEMM_XDL_f16_gfx942_DECS
   ```

3. **Lookup-table entry** keyed by `"arch_op_dtype"` (guarded by `{NonAccel,Accel,GemmGemm}_LOOKUP_TABLE_GEN`):

   ```c
   {"gfx942_gemm_f16", {PopulateParamsXDL::initParametersF16GemmGfx942,
                        PopulateParamsXDL::NInitParametersF16GemmGfx942}},
   ```

The rewriting itself is driven by `update_inc_file()` using `replace_section()` (for DEFS/DECS blocks) and `add_lookup_entry()` (for the map entries). If a section doesn’t exist yet, the functions create it before the appropriate `#endif` marker; if it already exists, they replace it atomically. The first-time layout of the file is created by `init_inc_file()`, which lays down empty `#ifdef` blocks for each instruction family (`NonAccel`, `XDL`, `Wmma`, `GemmGemm`) and for each lookup section (`NonAccel`, `Accel`, `GemmGemm`).

### Aliases: `--alias FROM TO`

Because tuning every dtype on every arch is expensive, the script also supports a *fallback alias* mode. `--alias bf16 f16` scans every existing entry whose dtype is `f16`, and for any arch/op that **does not already have a `bf16` entry**, it adds a lookup-table row pointing to the same array symbol. Concretely, `add_type_aliases()` copies the `{PopulateParams...::initParameters..., ...}` value from the `f16` entry and re-registers it under key `<arch>_<op>_bf16` with a trailing `// alias -> f16` comment. No new array is produced — only a new row in the lookup table.

---

## 5. Stage 4 — How the compiler consumes the header

The generated file is `#include`d three times in rocMLIR with different macros defined, exactly matching the three regions described above.

### 5.1 Declarations — inside `PopulateParams*` classes

In `mlir/include/mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h`, each `PopulateParams*` class opens a private block, defines the appropriate `*_DECLARATIONS_GEN` macro, and includes the inc file to inject its `static const StringRef initParameters...[]` and corresponding `static constexpr size_t N...` members. For example:

```166:171:mlir/include/mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h
class PopulateParams : public BasePopulateParams<GeneralGemmParamsAttr> {
private:
#define NonAccel_DECLARATIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef NonAccel_DECLARATIONS_GEN
```

`PopulateParamsXDL` and `PopulateParamsWmma` do the same with `XDL_DECLARATIONS_GEN` and `Wmma_DECLARATIONS_GEN`. `GemmGemmParamsAttr` gets its declarations from `GridwiseGemmGemmParams.h` in the same pattern.

### 5.2 Definitions — in the `.cpp` files

In `mlir/lib/Dialect/Rock/Tuning/GridwiseGemmParams.cpp`, each family defines its matching `*_DEFINITIONS_GEN` macro and includes the inc file to emit the bodies of those arrays (i.e. the literal perfconfig strings):

```41:46:mlir/lib/Dialect/Rock/Tuning/GridwiseGemmParams.cpp
/// Non-xdlops
// clang-format off
#define NonAccel_DEFINITIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef NonAccel_DEFINITIONS_GEN
// clang-format on
```

and similarly for `XDL_DEFINITIONS_GEN` near line 407 and `Wmma_DEFINITIONS_GEN` near line 577. `GridwiseGemmGemmParams.cpp` emits the `GemmGemm_DEFINITIONS_GEN` section for attention-style kernels.

### 5.3 Lookup table — `ParamLookupTable.cpp`

The runtime keying — `"<arch>_<op>_<dtype>" → ArrayRef<StringRef>` — is built by three template specializations in `mlir/lib/Dialect/Rock/Tuning/ParamLookupTable.cpp`, each of which includes the inc file with a different `*_LOOKUP_TABLE_GEN` macro:

```150:158:mlir/lib/Dialect/Rock/Tuning/ParamLookupTable.cpp
template <>
std::map<StringRef, ArrayRef<StringRef>>
ParamLookupTable<GeneralGemmParamsAttr>::buildTable() {
  return {
#define NonAccel_LOOKUP_TABLE_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef NonAccel_LOOKUP_TABLE_GEN
  };
}
```

The `AccelGemmParamsAttr` and `GemmGemmParamsAttr` specializations are the same pattern with `Accel_LOOKUP_TABLE_GEN` and `GemmGemm_LOOKUP_TABLE_GEN` instead of `NonAccel_LOOKUP_TABLE_GEN` (see lines 160–177 in the same file).

The runtime lookup logic (`ParamLookupTable<ParamsType>::lookup()`) then:

1. Normalizes the arch string to just `gfxNNN` via `normalizeArch()`.
2. Forms `"gfxNNN_<op>_<dtype>"` via `makeKey()`.
3. Looks it up. If not found, `findFallback()` / `getRelatives()` walk the keys to find the closest available arch with the same `<op>_<dtype>` suffix — this is how a newer gfx chip falls back to its closest available sibling.
4. Returns the `ArrayRef<StringRef>` of perfconfigs — exactly the array emitted by the generator.

That `ArrayRef<StringRef>` is what `PopulateParams::getTuningParameters()` iterates over to build the heuristic tuning list shown to the rest of the compiler:

```273:288:mlir/lib/Dialect/Rock/Tuning/GridwiseGemmParams.cpp
std::vector<GeneralGemmParamsAttr>
PopulateParams::getTuningParameters(OpBuilder &b, KernelType opType,
                                    Type dataTypeA, Type dataTypeB,
                                    StringRef arch) const {
  auto perfConfigs =
      ParamLookupTable<GeneralGemmParamsAttr>::lookup(arch, opType, dataTypeA);
  std::vector<GeneralGemmParamsAttr> result;
  result.reserve(perfConfigs.size());
  for (StringRef perfConfig : perfConfigs) {
    auto perfConfigAttr = StringAttr::get(b.getContext(), perfConfig);
    if (auto params = GeneralGemmParamsAttr::get(perfConfigAttr)) {
      result.push_back(params);
    }
  }
  return result;
}
```

---

## 6. Typical end-to-end workflow

To refresh the quick-tuning list for, say, GEMM on `gfx942` and `gfx90a`:

```bash
# 1. Run full tuning sweeps on the GPUs with --debug so per-config rows are kept.
python3 mlir/utils/performance/tuningRunner.py --config gemm.tsv --arch gfx942 --debug -o results/gfx942_gemm.tsv
python3 mlir/utils/performance/tuningRunner.py --config gemm.tsv --arch gfx90a --debug -o results/gfx90a_gemm.tsv
# => results/gfx942_gemm.tsv.debug and results/gfx90a_gemm.tsv.debug

# 2. Reduce via set-cover, update the checked-in .inc file.
python3 mlir/utils/performance/analysis/quickTuningGen.py \
  results/gfx942_gemm.tsv.debug results/gfx90a_gemm.tsv.debug --op gemm --th 0.93 --update

# 3. (Optional) Add a bf16 fallback to the f16 entries.
python3 mlir/utils/performance/analysis/quickTuningGen.py --alias bf16 f16

# 4. Rebuild rocMLIR. The new perfconfigs are baked into the compiler.
```

After step 2/3, run `git diff` on `mlir/include/mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc` to see the new list. After step 4, every invocation of the compiler that does not supply a tuning DB will use exactly these configs as the quick-tune candidates via `ParamLookupTable`.

---

## 7. Summary of components

- **Tuning sweep** — `mlir/utils/performance/tuningRunner.py` — writes `<out>.debug` TSVs: one row per `(problem, perfConfig)` with `TFlops`.
- **Generator** — `mlir/utils/performance/analysis/quickTuningGen.py` — set-cover per `(arch, op, dtype)`; rewrites the inc file.
- **Generated header** — `mlir/include/mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc` — checked-in arrays and lookup entries.
- **Class declarations** — `GridwiseGemmParams.h` and `GridwiseGemmGemmParams.h` — `#include` with `*_DECLARATIONS_GEN`.
- **Array definitions** — `GridwiseGemmParams.cpp` and `GridwiseGemmGemmParams.cpp` — `#include` with `*_DEFINITIONS_GEN`.
- **Lookup table** — `ParamLookupTable.cpp` — `#include` with `*_LOOKUP_TABLE_GEN`; runtime arch fallback in `lookup()`.

The key design trick is that **one generated text file** serves three structural roles in the C++ codebase through three preprocessor-gated sections. That keeps the declarations, the definitions and the lookup table perfectly in sync — they are all derived from the same optimizer run over the same tuning data.
