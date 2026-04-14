@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "BASEDIR="
for %%D in ("%CD%\.." "%CD%" "%CD%\build" "%~dp0..\..\..\build_vs_clang" "%USERPROFILE%\rocmlir\build_vs_clang") do (
  if not defined BASEDIR (
    if exist "%%~fD\bin\xmir-runner.exe" set "BASEDIR=%%~fD"
  )
)

if not defined BASEDIR (
  echo ERROR: Could not locate build directory with bin\xmir-runner.exe 1>&2
  exit /b 1
)

set "RUNNER=%BASEDIR%\bin\xmir-runner.exe"

call :ResolveLLVMLib mlir_rocm_runtime.dll MLIR_ROCM_RUNTIME
call :ResolveLLVMLib mlir_runner_utils.dll MLIR_RUNNER_UTILS
call :ResolveLLVMLib mlir_c_runner_utils.dll MLIR_C_RUNNER_UTILS
call :ResolveLLVMLib mlir_async_runtime.dll MLIR_ASYNC_RUNTIME
call :ResolveLLVMLib MLIRRocmExecutionEngineUtils.dll MLIR_ROCM_EXEC_ENGINE_UTILS
call :ResolveRocmLib conv-validation-wrappers.dll CONV_VALIDATION_WRAPPERS

if not defined MLIR_ROCM_RUNTIME goto :MissingLibs
if not defined MLIR_RUNNER_UTILS goto :MissingLibs
if not defined MLIR_C_RUNNER_UTILS goto :MissingLibs
if not defined MLIR_ASYNC_RUNTIME goto :MissingLibs
if not defined CONV_VALIDATION_WRAPPERS goto :MissingLibs

set "MLIR_ROCM_RUNTIME=%MLIR_ROCM_RUNTIME:\=/%"
set "MLIR_RUNNER_UTILS=%MLIR_RUNNER_UTILS:\=/%"
set "MLIR_C_RUNNER_UTILS=%MLIR_C_RUNNER_UTILS:\=/%"
set "MLIR_ASYNC_RUNTIME=%MLIR_ASYNC_RUNTIME:\=/%"
set "CONV_VALIDATION_WRAPPERS=%CONV_VALIDATION_WRAPPERS:\=/%"
if defined MLIR_ROCM_EXEC_ENGINE_UTILS set "MLIR_ROCM_EXEC_ENGINE_UTILS=%MLIR_ROCM_EXEC_ENGINE_UTILS:\=/%"

set "SHARED_LIBS=%MLIR_ROCM_RUNTIME%,%CONV_VALIDATION_WRAPPERS%,%MLIR_RUNNER_UTILS%,%MLIR_C_RUNNER_UTILS%,%MLIR_ASYNC_RUNTIME%"
if defined MLIR_ROCM_EXEC_ENGINE_UTILS set "SHARED_LIBS=%SHARED_LIBS%,%MLIR_ROCM_EXEC_ENGINE_UTILS%"

set "TEMP_INPUT="
set "INPUT_ARG="
if "%~1"=="" (
  set "TEMP_INPUT=%TEMP%\xmir-input-%RANDOM%-%RANDOM%.mlir"
  set "TMPFILE=!TEMP_INPUT!"
  "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -Command "$in=[Console]::OpenStandardInput();$out=[System.IO.File]::Open($env:TMPFILE,[System.IO.FileMode]::Create,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read);$in.CopyTo($out);$out.Close()" >nul
  set "TMPFILE="
  if errorlevel 1 (
    echo ERROR: Failed to capture piped input into temporary file. 1>&2
    exit /b 1
  )
  if not exist "!TEMP_INPUT!" (
    echo ERROR: Failed to capture piped input into temporary file. 1>&2
    exit /b 1
  )
  set "INPUT_ARG=!TEMP_INPUT!"
)

set "DLL_PATHS=%BASEDIR%\bin;%BASEDIR%\Debug\bin;%BASEDIR%\Release\bin;%BASEDIR%\RelWithDebInfo\bin;%BASEDIR%\MinSizeRel\bin;%BASEDIR%\external\llvm-project\llvm\bin;%BASEDIR%\external\llvm-project\llvm\Debug\bin;%BASEDIR%\external\llvm-project\llvm\Release\bin;%BASEDIR%\external\llvm-project\llvm\RelWithDebInfo\bin;%BASEDIR%\external\llvm-project\llvm\MinSizeRel\bin"
set "PATH=%DLL_PATHS%;%PATH%"

if defined INPUT_ARG (
  "%RUNNER%" "--shared-libs=%SHARED_LIBS%" --entry-point-result=void "%INPUT_ARG%"
) else (
  "%RUNNER%" "--shared-libs=%SHARED_LIBS%" --entry-point-result=void %*
)
set "EXIT_CODE=%ERRORLEVEL%"
if defined TEMP_INPUT if exist "%TEMP_INPUT%" del /f /q "%TEMP_INPUT%" >nul 2>&1
exit /b %EXIT_CODE%

:MissingLibs
echo ERROR: Missing required runtime libraries under %BASEDIR% 1>&2
echo        Expected: mlir_rocm_runtime.dll, mlir_runner_utils.dll, mlir_c_runner_utils.dll, mlir_async_runtime.dll, conv-validation-wrappers.dll 1>&2
exit /b 1

:ResolveLLVMLib
set "%~2="
for %%C in (Release RelWithDebInfo Debug MinSizeRel) do (
  if exist "%BASEDIR%\external\llvm-project\llvm\%%C\bin\%~1" (
    set "%~2=%BASEDIR%\external\llvm-project\llvm\%%C\bin\%~1"
    goto :eof
  )
)
if exist "%BASEDIR%\external\llvm-project\llvm\bin\%~1" (
  set "%~2=%BASEDIR%\external\llvm-project\llvm\bin\%~1"
)
goto :eof

:ResolveRocmLib
set "%~2="
for %%C in (Release RelWithDebInfo Debug MinSizeRel) do (
  if exist "%BASEDIR%\%%C\bin\%~1" (
    set "%~2=%BASEDIR%\%%C\bin\%~1"
    goto :eof
  )
)
if exist "%BASEDIR%\bin\%~1" (
  set "%~2=%BASEDIR%\bin\%~1"
)
goto :eof
