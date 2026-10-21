# Configures and builds the experiment drivers against the pinned OpenNN snapshot.
# Usage: pwsh code/build.ps1 [target ...]      (default: all drivers)
# Requires VS2022 (MSVC 14.39), its bundled CMake, and Ninja (Qt ships one).
param([string[]]$Targets = @("train_surrogates", "inverse_cases"))

$vs = "C:\Program Files\Microsoft Visual Studio\2022\Community"
$env:PATH = "$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Qt\Tools\Ninja;$env:PATH"
$source = $PSScriptRoot
$build = "C:\rai_onn\build"

$command = "call `"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && " +
           "cmake -S `"$source`" -B `"$build`" -G Ninja -DCMAKE_BUILD_TYPE=Release >nul && " +
           "cmake --build `"$build`" --target $($Targets -join ' ')"
cmd /c $command
exit $LASTEXITCODE
