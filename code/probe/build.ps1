# Configures and builds the probe driver against the newer OpenNN snapshot.
# Usage: pwsh code/probe/build.ps1
# Same toolchain and flags as code/build.ps1, a separate build tree and a separate bin.
$vs = "C:\Program Files\Microsoft Visual Studio\2022\Community"
$env:PATH = "$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Qt\Tools\Ninja;$env:PATH"
$source = $PSScriptRoot
$build = "C:\rai_onn_new\build"

$command = "call `"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && " +
           "cmake -S `"$source`" -B `"$build`" -G Ninja -DCMAKE_BUILD_TYPE=Release && " +
           "cmake --build `"$build`" --target probe_sample"
cmd /c $command
exit $LASTEXITCODE
