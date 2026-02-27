@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo DXML Dialect IR Validation Report
echo ==========================================
echo.
echo Testing with rocmlir-driver.exe
echo Timestamp: %date% %time%
echo.

set DRIVER=C:\Develop\rocMLIR.WML\build_vs\bin\rocmlir-driver.exe
set EXAMPLES=C:\Develop\rocMLIR.WML\examples\dxml-dialect

echo Creating validation report...
echo.

echo [1/8] Testing model1/model.mlir...
%DRIVER% %EXAMPLES%\model1\model.mlir > nul 2>&1
if %errorlevel% == 0 (
    echo   PASS - model1/model.mlir parsed successfully
) else (
    echo   FAIL - model1/model.mlir failed to parse
)

echo [2/8] Testing model2/model.mlir...
%DRIVER% %EXAMPLES%\model2\model.mlir > nul 2>&1
if %errorlevel% == 0 (
    echo   PASS - model2/model.mlir parsed successfully
) else (
    echo   FAIL - model2/model.mlir failed to parse
)

echo [3/8] Testing model3/model.mlir...
%DRIVER% %EXAMPLES%\model3\model.mlir > nul 2>&1
if %errorlevel% == 0 (
    echo   PASS - model3/model.mlir parsed successfully
) else (
    echo   FAIL - model3/model.mlir failed to parse
)

echo [4/8] Testing audio2face/model.mlir...
%DRIVER% %EXAMPLES%\audio2face\model.mlir > nul 2>&1
if %errorlevel% == 0 (
    echo   PASS - audio2face/model.mlir parsed successfully
) else (
    echo   FAIL - audio2face/model.mlir failed to parse
)

echo [5/8] Testing llama32/llama32_dxgml_static_decoder.mlir...
%DRIVER% %EXAMPLES%\llama32\llama32_dxgml_static_decoder.mlir > nul 2>&1
if %errorlevel% == 0 (
    echo   PASS - llama32_dxgml_static_decoder.mlir parsed successfully
) else (
    echo   FAIL - llama32_dxgml_static_decoder.mlir failed to parse
)

echo [6/8] Testing llama32/llama32_dxgml_static_pre-fill.mlir...
%DRIVER% %EXAMPLES%\llama32\llama32_dxgml_static_pre-fill.mlir > nul 2>&1
if %errorlevel% == 0 (
    echo   PASS - llama32_dxgml_static_pre-fill.mlir parsed successfully
) else (
    echo   FAIL - llama32_dxgml_static_pre-fill.mlir failed to parse
)

echo [7/8] Testing nemotron/model_decoder.mlir...
%DRIVER% %EXAMPLES%\nemotron\model_decoder.mlir > nul 2>&1
if %errorlevel% == 0 (
    echo   PASS - nemotron/model_decoder.mlir parsed successfully
) else (
    echo   FAIL - nemotron/model_decoder.mlir failed to parse
)

echo [8/8] Testing nemotron/model_pre-fill.mlir...
%DRIVER% %EXAMPLES%\nemotron\model_pre-fill.mlir > nul 2>&1
if %errorlevel% == 0 (
    echo   PASS - nemotron/model_pre-fill.mlir parsed successfully
) else (
    echo   FAIL - nemotron/model_pre-fill.mlir failed to parse
)

echo.
echo ==========================================
echo Validation Complete
echo ==========================================

endlocal
pause
