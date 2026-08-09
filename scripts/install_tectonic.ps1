$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$target = Join-Path $repoRoot 'tools\tectonic'
$archive = Join-Path $target 'tectonic.zip'
$expected = 'F61CE51F0B0ADE1015B7DE7EF368541C5424E9756ECBD0D7AF97D6D48030845F'
$url = 'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip'

New-Item -ItemType Directory -Force -Path $target | Out-Null
Invoke-WebRequest -Uri $url -OutFile $archive
$actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
if ($actual -ne $expected) {
    throw "Tectonic archive hash mismatch: expected $expected, received $actual"
}
Expand-Archive -LiteralPath $archive -DestinationPath $target -Force
& (Join-Path $target 'tectonic.exe') --version
