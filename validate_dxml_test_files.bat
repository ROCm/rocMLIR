@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo DXML Dialect IR Validation Report
echo ==========================================
echo.
echo Testing with rocmlir-driver.exe (rebuilt 4:01 PM)
echo Timestamp: %date% %time%
echo.

set DRIVER=C:\Develop\rocMLIR.WML\build_vs\bin\rocmlir-driver.exe
set EXAMPLES=C:\Develop\rocMLIR.WML\examples\dxml-dialect

echo Testing clean _test.mlir files...
echo.

echo [1/8] Testing model1/model_test.mlir...
%DRIVER% %EXAMPLES%\model1\model_test.mlir > nul 2>&1
if %errorlevel% == 0 (
    echo   ✓ PASS - model1/model_test.mlir
) else (
    echo   ✗ FAIL - model1/model_test.mlir
)

echo [2/8] Testing model2/model_test.mlir...
%DRIVER% %EXAMPLES%\model2\model_test.mlir > nul 2>&1
if %errorlevel% == 0 (
    echo   ✓ PASS - model2/model_test.mlir
) else (
    echo   ✗ FAIL - model2/model_test.mlir
)

echo [3/8] Testing model3/model_test.mlir...
%DRIVER% %EXAMPLES%\model3\model_test.mlir > nul 2>&1
if %errorlevel% == 0 (
    echo   ✓ PASS - model3/model_test.mlir
) else (
    echo   ✗ FAIL - model3/model_test.mlir
)

echo [4/8] Testing audio2face/model_test.mlir...
%DRIVER% %EXAMPLES%\audio2face\model_test.mlir > nul 2>&1
if %errorlevel% == 0 (
    echo   ✓ PASS - audio2face/model_test.mlir
) else (
    echo   ✗ FAIL - audio2face/model_test.mlir
)

echo [5/8] Testing llama32/llama32_dxgml_static_decoder_test.mlir...
%DRIVER% %EXAMPLES%\llama32\llama32_dxgml_static_decoder_test.mlir > nul 2>&1
if %errorlevel% == 0 (
    echo   ✓ PASS - llama32_dxgml_static_decoder_test.mlir
) else (
    echo   ✗ FAIL - llama32_dxgml_static_decoder_test.mlir
)

echo [6/8] Testing llama32/llama32_dxgml_static_pre-fill_test.mlir...
%DRIVER% %EXAMPLES%\llama32\llama32_dxgml_static_pre-fill_test.mlir > nul 2>&1
if %errorlevel% == 0 (
    echo   ✓ PASS - llama32_dxgml_static_pre-fill_test.mlir
) else (
    echo   ✗ FAIL - llama32_dxgml_static_pre-fill_test.mlir
)

echo [7/8] Testing nemotron/model_decoder_test.mlir...
%DRIVER% %EXAMPLES%\nemotron\model_decoder_test.mlir > nul 2>&1
if %errorlevel% == 0 (
    echo   ✓ PASS - nemotron/model_decoder_test.mlir
) else (
    echo   ✗ FAIL - nemotron/model_decoder_test.mlir
)

echo [8/8] Testing nemotron/model_pre-fill_test.mlir...
%DRIVER% %EXAMPLES%\nemotron\model_pre-fill_test.mlir > nul 2>&1
if %errorlevel% == 0 (
    echo   ✓ PASS - nemotron/model_pre-fill_test.mlir
) else (
    echo   ✗ FAIL - nemotron/model_pre-fill_test.mlir
)

echo.
echo ==========================================
echo Validation Complete
echo ==========================================
echo.
echo Note: Original .mlir files contain test directives for lit testing
echo       and are not meant to be parsed directly by rocmlir-driver.
echo       The _test.mlir files are clean versions for validation.
echo.

endlocal
