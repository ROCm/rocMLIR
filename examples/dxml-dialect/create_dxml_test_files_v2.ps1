# Create clean test versions of all DXML dialect examples - Version 2
$examples = @(
    @{source="model1\model.mlir"; output="model1\model_test.mlir"; modname="model1"},
    @{source="model2\model.mlir"; output="model2\model_test.mlir"; modname="model2"},
    @{source="model3\model.mlir"; output="model3\model_test.mlir"; modname="model3"},
    @{source="audio2face\model.mlir"; output="audio2face\model_test.mlir"; modname="audio2face"},
    @{source="llama32\llama32_dxgml_static_decoder.mlir"; output="llama32\llama32_dxgml_static_decoder_test.mlir"; modname="llama32_decoder"},
    @{source="llama32\llama32_dxgml_static_pre-fill.mlir"; output="llama32\llama32_dxgml_static_pre-fill_test.mlir"; modname="llama32_prefill"},
    @{source="nemotron\model_decoder.mlir"; output="nemotron\model_decoder_test.mlir"; modname="nemotron_decoder"},
    @{source="nemotron\model_pre-fill.mlir"; output="nemotron\model_pre-fill_test.mlir"; modname="nemotron_prefill"}
)

$basePath = "C:\Develop\rocMLIR.WML\examples\dxml-dialect"

foreach ($ex in $examples) {
    $sourceFile = Join-Path $basePath $ex.source
    $testFile = Join-Path $basePath $ex.output
    
    # Read entire file as string
    $content = [System.IO.File]::ReadAllText($sourceFile, [System.Text.Encoding]::UTF8)
    
    # Remove RUN lines, CHECK lines, and markers
    $content = $content -creplace '(?m)^\s*//\s*(RUN|CHECK)[^\r\n]*[\r\n]+', ''
    $content = $content -creplace '(?m)^\s*\{-#[^\r\n]*[\r\n]+', ''
    $content = $content -creplace '(?m)^\s*#-\}[^\r\n]*[\r\n]+', ''
    
    # Remove leading blank lines
    $content = $content -creplace '(?ms)^\s+', ''
    
    # Add module name
    $content = $content -creplace '(?m)^dxgml\.module\s*\{', "dxgml.module @$($ex.modname) {"
    
    # Write with UTF8 no BOM
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($testFile, $content, $utf8NoBom)
    
    Write-Host "Created: $testFile"
}

Write-Host ""
Write-Host "All test files created successfully!"
