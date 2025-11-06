# Copyright (C) 2025 Advanced Micro Devices Inc.

param(
    [ValidateScript({ Test-Path -Path $_ })]
    [string]$sourceDir,
    [string]$buildDir,
    [string]$installDir,
#    [ValidateSet('Release','RelWithDebInfo','MinRelSize','Debug')]
    [string]$buildType,
    [string[]]$defines,
    [ValidateSet('default','hipSdk','clangCl')]
    [string]$toolchain,
    [switch]$force = $false,
    [string]$configJson,
    [switch]$minimal = $false,
    [switch]$binSkim = $false,
    [string]$targets,
    [ValidateScript({ Test-Path -Path $_ })]
    [string]$hipPath,
    [switch]$skipConfigure = $false,
    [switch]$skipInstall = $false,
    [int]$jobs = [Math]::Max([Environment]::ProcessorCount - 2, 1)
)

$envOrig = Get-ChildItem env:
$envVars = @()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$PSNativeCommandUseErrorActionPreference = $true

function Remove-File {
    param (
        [string]$BasePath,
        [string]$FileName
    )
    $Path = Join-Path -Path $BasePath -ChildPath $FileName
    if (Test-Path -Path $Path) {
        Remove-Item -Path $Path -Force -ProgressAction SilentlyContinue
    }
}

function Invoke-Call {
    param (
        [scriptblock]$ScriptBlock,
        [string]$ErrorCode = $ErrorActionPreference
    )
    & @ScriptBlock
    if (($LASTEXITCODE -ne 0) -and $ErrorAction -eq 'Stop') {
        exit $LASTEXITCODE
    }
}

function Invoke-Environment {
    param (
        [Parameter(Mandatory=$true)]
        [string] $scriptPath
    )
    $cmdLine = """$scriptPath"" $args & set"
    $Global:envVars = @()
    & $env:SystemRoot\system32\cmd.exe /c $cmdLine | Select-String '^([^=]*)=(.*)$' | ForEach-Object {
        $name = $_.Matches[0].Groups[1].Value
        $value = $_.Matches[0].Groups[2].Value
        Set-Item -Path Env:$name -Value $value
        $Global:envVars += Get-Item env:$name
    }
}

function Cleanup-Environment {
    try {
        if ($null -eq $Global:envVars -or $Global:envVars.Count -eq 0) {
            return
        }
        Write-Host "Cleaning up environment..."
        $global:envVars | ForEach-Object {
            $name = $_.Name
            $exists = $envOrig | Where-Object { $_.Name -eq $name } | Measure-Object | Select-Object -ExpandProperty Count
            if ($exists -eq 0) {
                Remove-Item Env:$name -ErrorAction SilentlyContinue
            }
        }
    }
    catch {
        Write-Warning "Cleanup failed: $_"

    }
}

if (-not $sourceDir -or $sourceDir.Trim() -eq '') {
    $sourceDir = (Get-Location).Path
}
if (-not $buildDir -or $buildDir.Trim() -eq '') {
    $buildDir = Join-Path -Path $sourceDir -ChildPath 'build'
}
if (-not $installDir -or $installDir.Trim() -eq '') {
    $installDir = Join-Path -Path $sourceDir -ChildPath 'install'
}
$configurations = @('Debug', 'Release')
if ($buildType -and $buildType -ne '') {
    $configurations = @($buildType -split ',' | Where-Object { $_.Trim() -ne '' })
}
$parentDir = Split-Path -Path $installDir -Parent
if ($minimal) {
    $configJsonDefault = 'minimal'
} else {
    $configJsonDefault = 'default'
}
if (-not $configJson -or $configJson.Trim() -eq '') {
    $configJson = $configJsonDefault
}
if ($binSkim) {
    $configJson = "$configJson.binskim"
}
$configJson = Join-Path -Path $sourceDir -ChildPath "$configJson.json"
if (-not $hipPath -or $hipPath.Trim() -eq '') {
    if ($env:HIP_PATH) {
        $hipPath = "$env:HIP_PATH"
    } elseif ($env:ROCM_PATH) {
        $hipPath = "$env:ROCM_PATH"
    } else {
        $hipPath = "$parentDir\rocm"
    }
}
if($skipConfigure -and $force) {
    Write-Error 'SkipConfigure and Force cannot be used together'
    Exit
}
$jsonContent = @{}
if (Test-Path -Path $configJson) {
    $jsonContent = Get-Content -Path $configJson -Raw | ConvertFrom-Json
}
$defaultToolchain = 'default'
if ($jsonContent -and $jsonContent.PSObject.Properties.Name -contains 'toolchain') {
    $defaultToolchain = $jsonContent.toolchain
}
if (-not $toolchain -or $toolchain.Trim() -eq '') {
    $toolchain = $defaultToolchain
}
$cmakeHipPath = $hipPath.Replace('\', '/')
$buildDict = @{}
if ($toolchain -eq 'hipSdk') {
    $buildDict['CMAKE_C_COMPILER'] = "$cmakeHipPath/bin/clang.exe"
    $buildDict['CMAKE_CXX_COMPILER'] = "$cmakeHipPath/bin/clang++.exe"
    $buildDict['CMAKE_RC_COMPILER'] = "$cmakeHipPath/bin/llvm-rc.exe"
} elseif ($toolchain -eq 'clangCl') {
    $installDir = "$installDir.cl"
    $buildDir = "$buildDir.cl"
    $buildDict['CMAKE_C_COMPILER'] = "$cmakeHipPath/bin/clang-cl.exe"
    $buildDict['CMAKE_CXX_COMPILER'] = "$cmakeHipPath/bin/clang-cl.exe"
    $buildDict['CMAKE_RC_COMPILER'] = "$cmakeHipPath/bin/llvm-rc.exe"
}
if ($binSkim) {
    $buildDir = "$buildDir.binskim"
    $installDir = "$installDir.binskim"
}
if ($jsonContent -and $jsonContent.PSObject.Properties.Name -contains 'compileWarningAsError') {
    if ($jsonContent.compileWarningAsError) {
        $buildDict["CMAKE_COMPILE_WARNING_AS_ERROR"] = "ON"
    }
}
$targetsDefault = @("all")
if (-not $targets -or $targets.Trim() -eq '') {
    if ($jsonContent -and $jsonContent.PSObject.Properties.Name -contains 'targets') {
        $listTargets = $jsonContent.targets
    } else {
        $listTargets = $targetsDefault
    }
} else {
    $listTargets = @($targets -split ',' | Where-Object { $_.Trim() -ne '' })
}
$depPrefix = ''
if ($binSkim) {
    $depPrefix = '.binskim'
}
Register-EngineEvent -SourceIdentifier([System.Management.Automation.PSEngineEvent]::Exiting) -Action { Cleanup-Environment } | Out-Null
try {
    if ($null -eq $env:VSINSTALLDIR) {
        $vs_install_dir = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath
        Invoke-Environment "$vs_install_dir\VC\Auxiliary\Build\vcvars64.bat"
    }
    $configurations | ForEach-Object {
        $buildType = $_
        Write-Host "Building configuration '$buildType'...";
        $buildPath = Join-Path -Path $buildDir -ChildPath $buildType
        if (-Not (Test-Path -Path $buildPath)) {
            New-Item -ItemType Directory -Path $buildPath -Force | Out-Null
        } elseif ($force) {
            Remove-Item -Path $buildPath -Recurse -Force -ProgressAction SilentlyContinue
        } elseif (-not $skipConfigure) {
            Remove-File -BasePath $buildPath -FileName "CMakeCache.txt"
            Remove-File -BasePath $buildPath -FileName "CMakeFiles\\cmake.check_cache"
        }
        $buildDict["CMAKE_BUILD_TYPE"] = "$buildType"
        if ($jsonContent -and $jsonContent.PSObject.Properties.Name -contains "cacheVariables") {
            foreach ($key in $jsonContent.cacheVariables.PSObject.Properties.Name) {
                $buildDict[$key] = $ExecutionContext.InvokeCommand.ExpandString($jsonContent.cacheVariables.$key)
            }
        }
        if ($defines -and $defines.Trim() -ne '') {
            $defines | ForEach-Object {
                $s = $_.Trim() -split "="
                $buildDict[$s[0].Trim()] = $ExecutionContext.InvokeCommand.ExpandString($s[-1].Trim())
            }
        }
        $cmakePrefixPath = "$sourceDir\depend\$buildType"
        if ($jsonContent -and $jsonContent.PSObject.Properties.Name -contains "depends") {
            $cmakePrefixPath += $jsonContent.depends.GetEnumerator() | ForEach-Object { ";$parentDir\$_$depPrefix\$buildType" }
        }
        $buildDict["CMAKE_PREFIX_PATH"] = $cmakePrefixPath
        $buildDefines = $buildDict.GetEnumerator() | ForEach-Object { "-D$($_.Key)=$($_.Value)" }
        Write-Output $buildDefines
        if (-not $skipConfigure) {
            Invoke-Call -ScriptBlock { cmake -S $sourceDir -B $buildPath -G Ninja $buildDefines -Wno-deprecated -Wno-dev }
        }
        Invoke-Call -ScriptBlock { cmake --build $buildPath --config $buildType -j $jobs --target $listTargets }
        if (-not $skipInstall) {
            $prefixPath = Join-Path -Path $installDir -ChildPath $buildType
            if (Test-Path -Path $prefixPath) {
                Remove-Item -Path $prefixPath -Recurse -Force -ProgressAction SilentlyContinue
            }
            Invoke-Call -ScriptBlock { cmake --install $buildPath --prefix $prefixPath --config $buildType }
        }
    }
}
finally {
    Cleanup-Environment
}

