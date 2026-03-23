@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo DXML Dialect IR Validation Report
echo ==========================================
echo.
echo Timestamp: %date% %time%
echo.

REM Try Release driver first, fall back to Debug
set DRIVER_RELEASE=C:\Develop\rocMLIR.WML\build_vs\Release\bin\rocmlir-driver.exe
set DRIVER_DEBUG=C:\Develop\rocMLIR.WML\build_vs\bin\rocmlir-driver.exe

if exist "%DRIVER_RELEASE%" (
    set DRIVER=%DRIVER_RELEASE%
    echo Using Release driver: %DRIVER_RELEASE%
) else if exist "%DRIVER_DEBUG%" (
    set DRIVER=%DRIVER_DEBUG%
    echo Using Debug driver: %DRIVER_DEBUG%
) else (
    echo ERROR: rocmlir-driver.exe not found!
    echo   Expected at: %DRIVER_RELEASE%
    echo   Or at:       %DRIVER_DEBUG%
    echo.
    echo Build the project first:
    echo   cmake --build build_vs --config Release --target rocmlir-driver
    pause
    exit /b 1
)

set EXAMPLES=C:\Develop\rocMLIR.WML\examples\dxml-dialect
echo Driver: %DRIVER%
echo Examples: %EXAMPLES%
echo.

set PASS_COUNT=0
set FAIL_COUNT=0

echo Testing examples...
echo.

echo --- Simple CNN Models ---
call :TestFile "model1\model.mlir" "model1 (CNN with depth-to-space)"
call :TestFile "model2\model.mlir" "model2 (CNN variant)"
call :TestFile "model3\model.mlir" "model3 (CNN variant)"
echo.

echo --- Audio/Vision Models ---
call :TestFile "audio2face\model.mlir" "audio2face (with reduce ops)"
echo.

echo --- LLM Models ---
call :TestFile "llama32\llama32_dxgml_static_decoder.mlir" "llama32 decoder (GQA, dequantize)"
call :TestFile "llama32\llama32_dxgml_static_pre-fill.mlir" "llama32 pre-fill (GQA, dequantize)"
echo.

echo --- Nemotron Models ---
call :TestFile "nemotron\model_decoder.mlir" "nemotron decoder"
call :TestFile "nemotron\model_pre-fill.mlir" "nemotron pre-fill"
echo.

echo --- Phi Silica QDQ Models ---
call :TestFile "phi_silica_qdq\model.mlir" "phi_silica_qdq (quantized)"
echo.

echo ==========================================
echo Results: %PASS_COUNT% passed, %FAIL_COUNT% failed
echo ==========================================

if %FAIL_COUNT% GTR 0 (
    echo.
    echo To debug failures, run manually:
    echo   "%DRIVER%" "path\to\model.mlir"
    echo.
    pause
    exit /b 1
)

echo.
echo All tests passed!
echo.
pause
exit /b 0

REM ==========================================
REM Subroutine: TestFile <relative-path> <description>
REM ==========================================
:TestFile
    set "MLIR_FILE=%EXAMPLES%\%~1"
    set "DESC=%~2"
    if not exist "!MLIR_FILE!" (
        echo   [SKIP] !DESC! - file not found
        goto :eof
    )
    "!DRIVER!" "!MLIR_FILE!" >nul 2>nul
    if !errorlevel! == 0 (
        echo   [PASS] !DESC!
        set /a PASS_COUNT+=1
    ) else (
        echo   [FAIL] !DESC!
        "!DRIVER!" "!MLIR_FILE!" 2>&1 | more +0
        set /a FAIL_COUNT+=1
    )
    goto :eof
