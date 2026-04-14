@echo off
REM Script to run DXML dialect examples with rocmlir-driver on Windows
REM Usage: run_examples.bat [example] [arch] [pipeline]
REM
REM   run_examples.bat                      - Run all examples through ALL pipeline stages
REM   run_examples.bat all gfx1200 parse    - Run all examples, parse stage only
REM   run_examples.bat model1 gfx1200 gpu   - Run model1 through GPU lowering stage
REM   run_examples.bat phi_silica gfx1200 full - Full compilation of phi_silica
REM
REM Available examples:
REM   all           - All 11 DXML examples (default)
REM   model1        - Simple CNN with depth-to-space
REM   model2        - CNN variant
REM   model3        - CNN variant
REM   audio2face    - Audio2face model with reduce ops
REM   simple_gemm   - Simple GEMM+bias+relu (DXML native)
REM   conv_example  - Conv+BN+ReLU+MaxPool (DXML equiv of migraphx_convolution_example)
REM   llama32_dec   - LLaMA 3.2 decoder (GQA, dequantize)
REM   llama32_pre   - LLaMA 3.2 pre-fill (GQA, dequantize)
REM   nemotron_dec  - Nemotron decoder
REM   nemotron_pre  - Nemotron pre-fill
REM   phi_silica    - Phi Silica QDQ (quantized)
REM
REM Available pipelines:
REM   parse     - Parse and validate MLIR (default when no pipeline given)
REM   dxgml     - Lower DXML dialect -> MIGraphX -> TOSA (host-pipeline=dxgml)
REM   highlevel - Run bufferization / high-level Rock passes
REM   gpu       - Lower Rock kernels to GPU dialect (requires arch); writes <name>_<arch>.gpu
REM   rocdl     - Lower to ROCDL (requires arch); writes <name>_<arch>.rocdl
REM   binary    - Compile ELF binary: gpu + binary (requires arch); writes <name>_<arch>.bin
REM   full      - Full compilation pipeline: gpu + binary (requires arch); writes <name>_<arch>.bin
REM
REM Output files (gpu/rocdl/binary/full pipelines) are written to:
REM   <script-dir>\output\<testname>_<arch>.<ext>

setlocal enabledelayedexpansion

REM Parse arguments
set EXAMPLE=%1
set ARCH=%2
set PIPELINE=%3

if "%ARCH%"==""  set ARCH=gfx1200

REM Locate driver relative to this script
set SCRIPT_DIR=%~dp0
set ROOT_DIR=%SCRIPT_DIR%..\..
set DRIVER_PATH=

if exist "%ROOT_DIR%\build_vs_clang\bin\rocmlir-driver.exe" (
    set DRIVER_PATH=%ROOT_DIR%\build_vs_clang\bin\rocmlir-driver.exe
) else if exist "%ROOT_DIR%\build\bin\rocmlir-driver.exe" (
    set DRIVER_PATH=%ROOT_DIR%\build\bin\rocmlir-driver.exe
) else (
    echo Error: rocmlir-driver.exe not found.
    echo Please build rocMLIR first:
    echo   cmake --build build_vs_clang --config Release --target rocmlir-driver
    exit /b 1
)

REM Output directory for compiled artifacts (gpu, rocdl, binary, full pipelines)
set OUTPUT_DIR=%SCRIPT_DIR%output
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM -----------------------------------------------------------------------
REM No arguments: run ALL examples through ALL pipeline stages sequentially
REM -----------------------------------------------------------------------
if "%EXAMPLE%"=="" (
    echo ======================================
    echo DXML Dialect - Full Pipeline Validation
    echo ======================================
    echo Arch:   %ARCH%
    echo Driver: %DRIVER_PATH%
    echo Output: %OUTPUT_DIR%
    echo Running all 11 examples through every pipeline stage.
    echo.

    set TOTAL_PASS=0
    set TOTAL_FAIL=0

    for %%S in (parse dxgml highlevel gpu) do (
        echo ======================================
        echo Pipeline stage: %%S
        echo ======================================
        set PASS_COUNT=0
        set FAIL_COUNT=0
        set CURRENT_STAGE=%%S
        call :SetPipelineFlags "%%S"
        call :RunAll
        echo Stage %%S: !PASS_COUNT! passed, !FAIL_COUNT! failed
        set /a TOTAL_PASS+=!PASS_COUNT!
        set /a TOTAL_FAIL+=!FAIL_COUNT!
        echo.
    )

    echo ======================================
    echo Grand total: !TOTAL_PASS! passed, !TOTAL_FAIL! failed
    echo ======================================
    if !TOTAL_FAIL! GTR 0 exit /b 1
    exit /b 0
)

REM -----------------------------------------------------------------------
REM Explicit arguments provided
REM -----------------------------------------------------------------------
if "%PIPELINE%"=="" set PIPELINE=parse

echo ======================================
echo DXML Dialect Example Runner
echo ======================================
echo Example:  %EXAMPLE%
echo Arch:     %ARCH%
echo Pipeline: %PIPELINE%
echo Driver:   %DRIVER_PATH%
echo Output:   %OUTPUT_DIR%
echo ======================================
echo.

set PASS_COUNT=0
set FAIL_COUNT=0
set CURRENT_STAGE=%PIPELINE%
call :SetPipelineFlags "%PIPELINE%"
if errorlevel 1 exit /b 1

if /i "%EXAMPLE%"=="all" (
    call :RunAll
    goto show_results
)

if /i "%EXAMPLE%"=="model1"       call :RunExample "model1\model.mlir"                              "model1 (CNN with depth-to-space)"        "model1"        & goto show_results
if /i "%EXAMPLE%"=="model2"       call :RunExample "model2\model.mlir"                              "model2 (CNN variant)"                    "model2"        & goto show_results
if /i "%EXAMPLE%"=="model3"       call :RunExample "model3\model.mlir"                              "model3 (CNN variant)"                    "model3"        & goto show_results
if /i "%EXAMPLE%"=="audio2face"   call :RunExample "audio2face\model.mlir"                          "audio2face (reduce ops)"                 "audio2face"    & goto show_results
if /i "%EXAMPLE%"=="simple_gemm"  call :RunExample "simple_gemm\model.mlir"                          "simple_gemm (GEMM+bias+relu)"            "simple_gemm"   & goto show_results
if /i "%EXAMPLE%"=="conv_example" call :RunExample "conv_example\model.mlir" "conv_example (Conv+BN+ReLU+MaxPool)" "conv_example" & goto show_results
if /i "%EXAMPLE%"=="llama32_dec"  call :RunExample "llama32\llama32_dxgml_static_decoder.mlir"      "llama32 decoder"                         "llama32_decoder"   & goto show_results
if /i "%EXAMPLE%"=="llama32_pre"  call :RunExample "llama32\llama32_dxgml_static_pre-fill.mlir"     "llama32 pre-fill"                        "llama32_pre-fill"  & goto show_results
if /i "%EXAMPLE%"=="nemotron_dec" call :RunExample "nemotron\model_decoder.mlir"                    "nemotron decoder"                        "nemotron_decoder"  & goto show_results
if /i "%EXAMPLE%"=="nemotron_pre" call :RunExample "nemotron\model_pre-fill.mlir"                   "nemotron pre-fill"                       "nemotron_pre-fill" & goto show_results
if /i "%EXAMPLE%"=="phi_silica"   call :RunExample "phi_silica_qdq\model.mlir"                      "phi_silica_qdq (quantized)"              "phi_silica_qdq"    & goto show_results

echo Unknown example: %EXAMPLE%
echo Run without arguments to validate all examples through all stages.
exit /b 1

:show_results
echo ======================================
echo Results: !PASS_COUNT! passed, !FAIL_COUNT! failed
echo ======================================
if !FAIL_COUNT! GTR 0 (
    echo.
    echo To debug a failure, run:
    echo   "!DRIVER_PATH!" "path\to\model.mlir" !PIPELINE_FLAGS! ^> output\testname_!ARCH!.!PIPELINE!
    exit /b 1
)
echo.
echo All tests passed!
exit /b 0


REM ======================================
REM Subroutine: SetPipelineFlags <pipeline>
REM Sets PIPELINE_FLAGS and OUTPUT_EXT for the given stage name.
REM OUTPUT_EXT is non-empty only for pipelines that write compiled output.
REM ======================================
:SetPipelineFlags
    set "PIPELINE_FLAGS="
    set "OUTPUT_EXT="
    set "OUTPUT_BINARY="
    if /i "%~1"=="parse"     set "PIPELINE_FLAGS="                                              & goto :eof
    if /i "%~1"=="dxgml"     set "PIPELINE_FLAGS=--host-pipeline=dxgml"                        & set "OUTPUT_EXT=dxgml"   & goto :eof
    if /i "%~1"=="highlevel" set "PIPELINE_FLAGS=--host-pipeline=dxgml,highlevel"                                     & set "OUTPUT_EXT=hlevel"  & goto :eof
    if /i "%~1"=="gpu"       set "PIPELINE_FLAGS=--host-pipeline=dxgml --kernel-pipeline=highlevel,gpu --arch=%ARCH%"           & set "OUTPUT_EXT=gpu"     & goto :eof
    if /i "%~1"=="rocdl"     set "PIPELINE_FLAGS=--host-pipeline=dxgml --kernel-pipeline=highlevel,gpu,rocdl --arch=%ARCH%"    & set "OUTPUT_EXT=rocdl"   & goto :eof
    if /i "%~1"=="binary"    set "PIPELINE_FLAGS=--host-pipeline=dxgml --kernel-pipeline=highlevel,gpu,binary --arch=%ARCH%"   & set "OUTPUT_EXT=bin"          & goto :eof
    if /i "%~1"=="full"      set "PIPELINE_FLAGS=--host-pipeline=dxgml --kernel-pipeline=full --arch=%ARCH%"                   & set "OUTPUT_EXT=bin"          & goto :eof
    echo Unknown pipeline: %~1
    echo Valid options: parse, dxgml, highlevel, gpu, rocdl, binary, full
    exit /b 1

REM ======================================
REM Subroutine: RunAll
REM Runs all DXML examples with current PIPELINE_FLAGS.
REM Note: conv_example (batch_norm+pooling) supports parse/dxgml/highlevel but NOT
REM       gpu/rocdl/binary/full (batch_norm/pooling lower to linalg, not rock.* GPU kernels).
REM ======================================
:RunAll
    echo --- Simple Models ---
    call :RunExample "model1\model.mlir"        "model1 (CNN with depth-to-space)"     "model1"
    call :RunExample "model2\model.mlir"        "model2 (CNN variant)"                 "model2"
    call :RunExample "model3\model.mlir"        "model3 (CNN variant)"                 "model3"
    call :RunExample "simple_gemm\model.mlir"   "simple_gemm (GEMM+bias+relu)"         "simple_gemm"
    REM conv_example: non-Rock linalg ops (batch_norm, pooling) are lowered via
    REM scf.parallel -> gpu.launch -> gpu.func outlining; all pipeline stages supported
    call :RunExample "conv_example\model.mlir"  "conv_example (Conv+BN+ReLU+MaxPool)"  "conv_example"
    echo.
    echo --- Audio/Vision Models ---
    call :RunExample "audio2face\model.mlir"  "audio2face (with reduce ops)"     "audio2face"
    echo.
    echo --- LLM Models ---
    call :RunExample "llama32\llama32_dxgml_static_decoder.mlir"  "llama32 decoder (GQA, dequantize)"   "llama32_decoder"
    call :RunExample "llama32\llama32_dxgml_static_pre-fill.mlir" "llama32 pre-fill (GQA, dequantize)"  "llama32_pre-fill"
    echo.
    echo --- Nemotron Models ---
    call :RunExample "nemotron\model_decoder.mlir"  "nemotron decoder"            "nemotron_decoder"
    call :RunExample "nemotron\model_pre-fill.mlir" "nemotron pre-fill"           "nemotron_pre-fill"
    echo.
    echo --- Phi Silica QDQ Models ---
    call :RunExample "phi_silica_qdq\model.mlir"  "phi_silica_qdq (quantized)"   "phi_silica_qdq"
    goto :eof

REM ======================================
REM Subroutine: RunExample <relative-path> <description> <testname>
REM   Writes output to %OUTPUT_DIR%\<testname>_<arch>.<ext> when OUTPUT_EXT is set.
REM ======================================
:RunExample
    set "MLIR_REL=%~1"
    set "DESC=%~2"
    set "TESTNAME=%~3"
    set "MLIR_FILE=%SCRIPT_DIR%%MLIR_REL%"

    if not exist "!MLIR_FILE!" (
        echo   [SKIP] !DESC! - file not found
        goto :eof
    )

    REM Determine output file path for pipelines that produce compiled output
    set "OUTPUT_FILE="
    if not "!OUTPUT_EXT!"=="" (
        if not "!TESTNAME!"=="" (
            set "OUTPUT_FILE=!OUTPUT_DIR!\!TESTNAME!_!ARCH!.!OUTPUT_EXT!"
        )
    )

    if "!PIPELINE_FLAGS!"=="" (
        "!DRIVER_PATH!" "!MLIR_FILE!" >nul 2>&1
    ) else if "!OUTPUT_FILE!"=="" (
        "!DRIVER_PATH!" "!MLIR_FILE!" !PIPELINE_FLAGS! >nul 2>&1
    ) else (
        "!DRIVER_PATH!" "!MLIR_FILE!" !PIPELINE_FLAGS! >"!OUTPUT_FILE!" 2>nul
    )

    if !errorlevel! == 0 (
        echo   [PASS] !DESC!
        if not "!OUTPUT_FILE!"=="" echo         -^> !OUTPUT_FILE!
        set /a PASS_COUNT+=1
    ) else (
        echo   [FAIL] !DESC!
        if "!PIPELINE_FLAGS!"=="" (
            "!DRIVER_PATH!" "!MLIR_FILE!" 2>&1
        ) else (
            "!DRIVER_PATH!" "!MLIR_FILE!" !PIPELINE_FLAGS! 2>&1
        )
        set /a FAIL_COUNT+=1
    )
    goto :eof
