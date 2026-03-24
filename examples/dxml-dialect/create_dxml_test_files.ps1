# Create clean test versions of all DXML dialect examples
$examples = @(
    @{dir="model1"; name="model1"},
    @{dir="model2"; name="model2"},
    @{dir="model3"; name="model3"},
    @{dir="audio2face"; name="audio2face"}
)

$basePath = "C:\Develop\rocMLIR.WML\examples\dxml-dialect"

foreach ($ex in $examples) {
    $sourceFile = Join-Path $basePath "$($ex.dir)\model.mlir"
    $testFile = Join-Path $basePath "$($ex.dir)\model_test.mlir"
    
    # Read file and filter
    $lines = Get-Content $sourceFile -Encoding UTF8
    $filtered = $lines | Where-Object { 
        $_ -notmatch '^//\s*(RUN|CHECK)' -and 
        $_ -notmatch '^\{-#' -and 
        $_ -notmatch '^#-\}' 
    }
    
    # Remove leading/trailing blank lines
    $trimmed = $filtered | Where-Object { $_.Trim() }
    if ($trimmed.Count -eq 0) { $trimmed = $filtered }
    
    # Add module name
    $newContent = $trimmed -replace '^dxgml\.module\s*\{', "dxgml.module @$($ex.name) {"
    
    # Write with UTF8 no BOM
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllLines($testFile, $newContent, $utf8NoBom)
    
    Write-Host "Created: $testFile"
}

# Handle llama32 files
$llama32Files = @("llama32_dxgml_static_decoder", "llama32_dxgml_static_pre-fill")
foreach ($file in $llama32Files) {
    $sourceFile = Join-Path $basePath "llama32\$file.mlir"
    $testFile = Join-Path $basePath "llama32\${file}_test.mlir"
    
    $lines = Get-Content $sourceFile -Encoding UTF8
    $filtered = $lines | Where-Object { 
        $_ -notmatch '^//\s*(RUN|CHECK)' -and 
        $_ -notmatch '^\{-#' -and 
        $_ -notmatch '^#-\}' 
    }
    
    $trimmed = $filtered | Where-Object { $_.Trim() }
    if ($trimmed.Count -eq 0) { $trimmed = $filtered }
    
    $newContent = $trimmed -replace '^dxgml\.module\s*\{', "dxgml.module @llama32 {"
    
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllLines($testFile, $newContent, $utf8NoBom)
    
    Write-Host "Created: $testFile"
}

# Handle nemotron files  
$nemotronFiles = @("model_decoder", "model_pre-fill")
foreach ($file in $nemotronFiles) {
    $sourceFile = Join-Path $basePath "nemotron\$file.mlir"
    $testFile = Join-Path $basePath "nemotron\${file}_test.mlir"
    
    $lines = Get-Content $sourceFile -Encoding UTF8
    $filtered = $lines | Where-Object { 
        $_ -notmatch '^//\s*(RUN|CHECK)' -and 
        $_ -notmatch '^\{-#' -and 
        $_ -notmatch '^#-\}' 
    }
    
    $trimmed = $filtered | Where-Object { $_.Trim() }
    if ($trimmed.Count -eq 0) { $trimmed = $filtered }
    
    $newContent = $trimmed -replace '^dxgml\.module\s*\{', "dxgml.module @nemotron {"
    
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllLines($testFile, $newContent, $utf8NoBom)
    
    Write-Host "Created: $testFile"
}

Write-Host ""
Write-Host "All test files created successfully!"
