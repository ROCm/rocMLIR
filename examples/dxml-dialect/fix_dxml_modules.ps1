# Script to add module wrapper to DXML files
$files = @(
    "examples\dxml-dialect\model2\model.mlir",
    "examples\dxml-dialect\model3\model.mlir",
    "examples\dxml-dialect\audio2face\model.mlir",
    "examples\dxml-dialect\llama32\llama32_dxgml_static_decoder.mlir",
    "examples\dxml-dialect\llama32\llama32_dxgml_static_pre-fill.mlir",
    "examples\dxml-dialect\nemotron\model_decoder.mlir",
    "examples\dxml-dialect\nemotron\model_pre-fill.mlir"
)

foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "Processing $file..."
        $content = Get-Content $file -Raw
        
        # Check if already has module wrapper
        if ($content -notmatch '^module \{') {
            # Add module wrapper and indent dxgml.module
            $content = $content -replace '(?m)^dxgml\.module \{', '  dxgml.module {'
            $content = $content -replace '(?m)^  \}(\r?\n)\}$', '  }$1}'
            $content = "module {`n" + $content + "`n}"
            
            Set-Content -Path $file -Value $content -NoNewline
            Write-Host "  Added module wrapper to $file"
        } else {
            Write-Host "  $file already has module wrapper"
        }
    }
}

Write-Host "`nDone!"
