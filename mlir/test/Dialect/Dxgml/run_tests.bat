@echo off
REM Script to run DXGML dialect tests on Windows
REM Usage: run_tests.bat [test_name] [build_config]
REM
REM Examples:
REM   run_tests.bat types WinDebug
REM   run_tests.bat model1 RelWithDebInfo
REM   run_tests.bat all WinDebug

setlocal enabledelayedexpansion

REM Default values
set TEST=%1
if "%TEST%"=="" set TEST=all

set BUILD_CONFIG=%2
if "%BUILD_CONFIG%"=="" set BUILD_CONFIG=WinDebug

REM Determine build directory
set SCRIPT_DIR=%~dp0
set ROOT_DIR=%SCRIPT_DIR%..\..\..
set BUILD_DIR=%ROOT_DIR%\build\%BUILD_CONFIG%

REM Check if build directory exists
if not exist "%BUILD_DIR%" (
    echo Error: Build directory not found at %BUILD_DIR%
    echo Please build rocMLIR first for configuration: %BUILD_CONFIG%
    exit /b 1
)

REM Check if rocmlir-driver exists
set ROCMLIR_OPT=%BUILD_DIR%\bin\rocmlir-opt.exe
set ROCMLIR_DRIVER=%BUILD_DIR%\bin\rocmlir-driver.exe

if not exist "%ROCMLIR_OPT%" (
    set ROCMLIR_OPT=%BUILD_DIR%\external\llvm-project\llvm\bin\mlir-opt.exe
)

if not exist "%ROCMLIR_OPT%" (
    echo Error: Could not find rocmlir-opt.exe or mlir-opt.exe
    echo Please build rocMLIR first
    exit /b 1
)

echo ======================================
echo DXGML Dialect Test Runner
echo ======================================
echo Test:         %TEST%
echo Build Config: %BUILD_CONFIG%
echo Build Dir:    %BUILD_DIR%
echo Tool:         %ROCMLIR_OPT%
echo ======================================
echo.

REM Run based on test selection
if /i "%TEST%"=="types" goto test_types
if /i "%TEST%"=="ops" goto test_ops
if /i "%TEST%"=="model1" goto test_model1
if /i "%TEST%"=="all" goto test_all
if /i "%TEST%"=="parse" goto test_parse_only

echo Unknown test: %TEST%
echo Available tests: types, ops, model1, all, parse
exit /b 1

:test_types
echo [TEST] Running types.mlir...
"%ROCMLIR_OPT%" "%SCRIPT_DIR%types.mlir"
if errorlevel 1 (
    echo [FAILED] types.mlir
    exit /b 1
)
echo [PASSED] types.mlir
goto end

:test_ops
echo [TEST] Running ops.mlir...
"%ROCMLIR_OPT%" "%SCRIPT_DIR%ops.mlir"
if errorlevel 1 (
    echo [FAILED] ops.mlir
    exit /b 1
)
echo [PASSED] ops.mlir
goto end

:test_model1
echo [TEST] Running model1.mlir...
"%ROCMLIR_OPT%" "%SCRIPT_DIR%model1.mlir"
if errorlevel 1 (
    echo [FAILED] model1.mlir
    exit /b 1
)
echo [PASSED] model1.mlir
goto end

:test_all
echo [TEST] Running all tests...
echo.

echo [1/3] Testing types.mlir...
"%ROCMLIR_OPT%" "%SCRIPT_DIR%types.mlir" > nul 2>&1
if errorlevel 1 (
    echo [FAILED] types.mlir
    set /a FAILED_COUNT+=1
) else (
    echo [PASSED] types.mlir
    set /a PASSED_COUNT+=1
)

echo [2/3] Testing ops.mlir...
"%ROCMLIR_OPT%" "%SCRIPT_DIR%ops.mlir" > nul 2>&1
if errorlevel 1 (
    echo [FAILED] ops.mlir
    set /a FAILED_COUNT+=1
) else (
    echo [PASSED] ops.mlir
    set /a PASSED_COUNT+=1
)

echo [3/3] Testing model1.mlir...
"%ROCMLIR_OPT%" "%SCRIPT_DIR%model1.mlir" > nul 2>&1
if errorlevel 1 (
    echo [FAILED] model1.mlir
    set /a FAILED_COUNT+=1
) else (
    echo [PASSED] model1.mlir
    set /a PASSED_COUNT+=1
)

echo.
echo ======================================
echo Test Summary
echo ======================================
echo Passed: %PASSED_COUNT%/3
echo Failed: %FAILED_COUNT%/3
echo ======================================

if %FAILED_COUNT% GTR 0 (
    exit /b 1
)
goto end

:test_parse_only
echo [TEST] Parse test (no verification)...
for %%f in ("%SCRIPT_DIR%*.mlir") do (
    echo Parsing %%~nxf...
    "%ROCMLIR_OPT%" "%%f" > nul 2>&1
    if errorlevel 1 (
        echo   [FAILED] %%~nxf
    ) else (
        echo   [PASSED] %%~nxf
    )
)
goto end

:end
echo.
echo Done!
