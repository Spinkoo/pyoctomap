# Build OctoMap + dynamicEDT3D with MSVC, install locally, stage DLLs/import libs for pyoctomap.
param(
    [Parameter(Mandatory = $true)]
    [string] $ProjectRoot
)

$ErrorActionPreference = "Stop"

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

Set-Location $Octo
foreach ($d in @("build", "install")) {
    if (Test-Path $d) {
        Remove-Item $d -Recurse -Force
    }
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

$requiredLibs = @("octomap.lib", "octomath.lib", "dynamicedt3d.lib")
foreach ($name in $requiredLibs) {
    $p = Join-Path $LibStaging $name
    if (-not (Test-Path $p)) {
        Write-Error "Missing required import library: $name"
        exit 1
    }
}

$dllCount = (Get-ChildItem -Path $LibStaging -Filter "*.dll" -File -ErrorAction SilentlyContinue).Count
if ($dllCount -lt 1) {
    Write-Error "No DLLs in lib — link/runtime will fail (expected octomap, octomath, dynamicEDT3D DLLs)."
    exit 1
}

Write-Host "==> OctoMap CI build OK"
