# Runs every OpenNN experiment of one case (seeds 1-10), skipping runs already logged in
# out/runs_<case>.csv (so an interrupted run resumes). Run from anywhere:
#   pwsh code/run_library.ps1 <concrete|hea|steel|kernel>
# Outputs: out/sample/*, out/solve/*, out/runs_<case>.csv.
param([Parameter(Mandatory)][string]$Case, [int]$Seeds = 10, [int]$Starts = 2000)

Set-Location $PSScriptRoot
$exe = Join-Path $PSScriptRoot "bin\inverse_cases.exe"
$log = Join-Path $PSScriptRoot "out\runs_$Case.csv"
$tasks = @("single")
# hea has one objective; the kernel case has no second criterion at fixed latency.
if ($Case -ne "hea" -and $Case -ne "kernel") { $tasks += "bi" }

# The kernel case states the specification three ways: as an equality (unsatisfiable on a
# catalogue), as a band around the target, and as a fixed objective.
$statements = @("constraint", "objective")
if ($Case -eq "kernel") { $statements = @("constraint", "band", "objective") }

function Done([string]$mode, [string]$solver, [string]$formulation, [string]$task, [int]$seed) {
    if (-not (Test-Path $log)) { return $false }
    return [bool](Import-Csv $log | Where-Object {
        $_.mode -eq $mode -and $_.solver -eq $solver -and $_.formulation -eq $formulation -and
        $_.task -eq $task -and [int]$_.seed -eq $seed })
}

foreach ($seed in 1..$Seeds) {
    if (-not (Done "sample" "repair" "constraint" "none" $seed)) { & $exe sample $Case $seed $Starts }
    if ($Case -eq "kernel" -and -not (Done "sample" "repair" "band" "none" $seed)) {
        & $exe sample $Case $seed $Starts band
    }
    foreach ($task in $tasks) {
        foreach ($formulation in $statements) {
            foreach ($solver in "dc", "ga") {
                if (Done "solve" $solver $formulation $task $seed) { continue }
                & $exe solve $Case $solver $formulation $task $seed 2>&1 | Where-Object { $_ -notmatch "local domains" }
            }
        }
    }
}
