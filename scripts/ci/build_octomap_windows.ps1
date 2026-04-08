# Build OctoMap + dynamicEDT3D with MSVC, install locally, stage DLLs/import libs for pyoctomap.
param(
    [Parameter(Mandatory = $true)]
    [string] $ProjectRoot
)

$ErrorActionPreference = "Stop"

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$Octo = Join-Path $ProjectRoot "src\octomap"
$LibStaging = Join-Path $Octo "lib"
$InstallPrefix = Join-Path $Octo "install"

Write-Host "==> OctoMap CI build (Windows)"
Write-Host "    ProjectRoot=$ProjectRoot"
Write-Host "    Octo=$Octo"

if (-not (Test-Path $Octo)) {
    Write-Error "Missing $Octo"
    exit 1
}

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    foreach ($dir in @(
            "${env:ProgramFiles}\CMake\bin",
            "${env:ProgramFiles(x86)}\CMake\bin"
        )) {
        if (Test-Path (Join-Path $dir "cmake.exe")) {
            $env:PATH = "$dir;$env:PATH"
            break
        }
    }
}
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Error "cmake not found (install CMake or add to PATH)"
    exit 1
}

Set-Location $Octo
foreach ($d in @("build", "install")) {
    if (Test-Path $d) {
        Remove-Item $d -Recurse -Force
    }
}
if (Test-Path $LibStaging) {
    Get-ChildItem -Path $LibStaging -Force -ErrorAction SilentlyContinue | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Force -Path $LibStaging | Out-Null

$cmakeArgs = @(
    "-B", "build",
    "-S", ".",
    "-A", "x64",
    "-DCMAKE_INSTALL_PREFIX=$InstallPrefix",
    "-DCMAKE_CXX_STANDARD=14",
    "-DBUILD_OCTOVIS_SUBPROJECT=OFF",
    "-DBUILD_DYNAMICETD3D_SUBPROJECT=ON"
)

& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& cmake --build build --config Release --parallel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& cmake --install build --config Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

function Copy-Staged {
    param([string] $SourceDir)
    if (-not (Test-Path $SourceDir)) { return }
    Get-ChildItem -Path $SourceDir -File -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item $_.FullName -Destination $LibStaging -Force
    }
}

Copy-Staged (Join-Path $InstallPrefix "bin")
Copy-Staged (Join-Path $InstallPrefix "lib")

Write-Host "==> Staged under $LibStaging`:"
Get-ChildItem $LibStaging

$libs = @(Get-ChildItem -Path $LibStaging -Filter "*.lib" -File -ErrorAction SilentlyContinue)
foreach ($pat in @("octomap", "octomath", "dynamicedt")) {
    $hit = $libs | Where-Object { $_.Name -match $pat }
    if (-not $hit) {
        Write-Error "No import library matching pattern '$pat' under $LibStaging"
        Get-ChildItem $LibStaging
        exit 1
    }
}

$dllCount = (Get-ChildItem -Path $LibStaging -Filter "*.dll" -File -ErrorAction SilentlyContinue).Count
if ($dllCount -lt 1) {
    Write-Error "No DLLs in lib — link/runtime will fail (expected octomap, octomath, dynamicEDT3D DLLs)."
    Get-ChildItem $LibStaging
    exit 1
}

Write-Host "==> OctoMap CI build OK"
