# M6 F2 (Whisper-small full fine-tune) reproduction on the RTX 5070 Ti. Windows PowerShell 5.1 and PowerShell 7.
# Every step is idempotent: a finished stage is skipped, an unfinished one is moved aside (never deleted) and redone.
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('preflight', 'baseline', 'smoke', 'train', 'evaluate', 'pipeline', 'trainloop', 'evalloop', 'serial', 'status', 'compare')]
    [string]$Step,
    [string]$SeedList = '17,18,19',
    [string]$FinishBy = '05:55',
    [double]$EvalTailMinutes = 30,
    [double]$OverlapMaxPeakGiB = 8
)
$ErrorActionPreference = 'Continue'
$Seeds = @($SeedList.Split(',') | ForEach-Object { [int]$_ })
$Artifacts = 'C:\turas-artifacts'
if ($env:F2REPRO_ARTIFACTS) { $Artifacts = $env:F2REPRO_ARTIFACTS }  # local dry runs only
$R = "$Artifacts\verify-2026-11-22"
$M = "$R\main"
$D = "$R\pr26\data\cleaned\turas-tr-econ-legis-v0.1"
$O = "$Artifacts\m6-repro-f2"
$Rel = 'turas-tr-econ-legis-v0.1'
$Tool = "$O\f2repro.py"
$Self = "$O\f2repro.ps1"
$G = (Get-Content -LiteralPath "$O\gpu-python.txt" -TotalCount 1).Trim()
$Lean = "$R\venv-lean-3.13\Scripts\python.exe"
$ProtocolSha = '288aebd71d8a3574e13d436748d4161bc258a487e65066f1c3508bc8e835bec1'
$ValRefs = "$D\$Rel.references.validation.jsonl"
$ValRefSha = 'd0c8519ca6aed053f89786bc733541a70faee09cf8d8bdb6cc12065d97dbd958'
$TestRefSha = '31e372f0e35f504de59b49f1e0e1f3ed6fcc2772138c591a2cf2b054a4e7ef3f'
$Baseline = "$O\b1-val-baseline"

$env:PYTHONPATH = "$M\src"
$env:OMP_NUM_THREADS = '4'; $env:MKL_NUM_THREADS = '4'
$env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'; $env:TOKENIZERS_PARALLELISM = 'false'
$env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'
Set-Location -LiteralPath $M
New-Item -ItemType Directory -Force -Path "$O\logs" | Out-Null

function Write-Note([string]$Text) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Text"
    Write-Host $line
    Add-Content -LiteralPath "$O\pipeline.log" -Value $line -Encoding UTF8
}

# Run python; stream output to the console and a UTF-8 log; return the exit code.
function Invoke-Py([string]$Python, [string]$Log, [string[]]$PyArgs) {
    Write-Note "start $(Split-Path -Leaf $Log)"
    & $Python -u @PyArgs 2>&1 | ForEach-Object {
        $line = "$_"
        Add-Content -LiteralPath $Log -Value $line -Encoding UTF8
        Write-Host $line
    }
    $code = $LASTEXITCODE
    Write-Note "end $(Split-Path -Leaf $Log) exit=$code"
    return $code
}

function Get-Tool([string[]]$ToolArgs) {
    $out = & $G $Tool @ToolArgs
    if ($LASTEXITCODE -ne 0) { throw "f2repro.py $($ToolArgs[0]) failed" }
    return ($out | Select-Object -Last 1)
}

function Move-Aside([string]$Path) {
    if (Test-Path -LiteralPath $Path) {
        $name = (Split-Path -Leaf $Path) + '.stale-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
        Rename-Item -LiteralPath $Path -NewName $name
        Write-Note "moved aside $Path -> $name"
    }
}

function Read-Json([string]$Path) {
    try { return (Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json) } catch { return $null }
}

function Get-TestRefs {
    $hit = Get-ChildItem -LiteralPath "$Artifacts\m1-release" -Recurse -File -Filter "$Rel.references.test.jsonl" -ErrorAction SilentlyContinue |
        Where-Object { (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower() -eq $TestRefSha } | Select-Object -First 1
    if ($hit) { return $hit.FullName }
    Write-Note "no M1 archive copy with the payload hash under $Artifacts\m1-release; using the published copy in pr26"
    return "$D\$Rel.references.test.jsonl"
}

function Invoke-Scorer([string]$Split, [string]$Refs, [string]$RefSha, [string[]]$Runs, [string]$Output, [string]$Log) {
    $runArgs = @(); foreach ($r in $Runs) { $runArgs += @('--run', $r) }
    $a = @('scripts\m6_score_baselines.py', '--protocol', 'configs\evaluation\m45_final_protocol.json', '--protocol-sha256', $ProtocolSha,
        '--split', $Split, '--references', $Refs, '--references-sha256', $RefSha) + $runArgs + @('--output', $Output)
    return (Invoke-Py $Lean $Log $a)
}

function Step-Preflight {
    $refs = Get-TestRefs
    Set-Content -LiteralPath "$O\test-references.txt" -Value $refs -Encoding ASCII
    $c1 = Invoke-Py $G "$O\logs\preflight-gpu.log" @($Tool, 'preflight', '--data', $D, '--test-refs', $refs, '--gpu')
    $c2 = Invoke-Py $Lean "$O\logs\preflight-lean.log" @($Tool, 'preflight')
    & $Lean -c "import jiwer, numpy, pandas; print('lean venv imports jiwer, numpy, pandas: ok')"
    if ($c1 -ne 0 -or $c2 -ne 0 -or $LASTEXITCODE -ne 0) { throw 'preflight FAILED: fix before any GPU step' }
}

function Step-Baseline {
    if (Test-Path -LiteralPath "$Baseline\baseline-evidence.json") { Write-Note 'baseline: done, skipped' }
    else {
        Move-Aside $Baseline
        Get-Tool @('config', '--data', $D, '--baseline', "$Baseline\baseline-evidence.json", '--seed', '17', '--output', "$O\baseline-config.json") | Out-Null
        $c = Invoke-Py $G "$O\logs\b1-val-baseline.log" @('scripts\m45_validation_baseline.py', '--config', "$O\baseline-config.json",
            '--family', 'whisper', '--inference', "$D\$Rel.inference.validation.jsonl", '--warmup-inference', "$D\$Rel.inference.train.jsonl",
            '--audio-root', $D, '--output', $Baseline)
        if ($c -ne 0) { throw 'baseline validation decode failed' }
    }
    if (-not (Test-Path -LiteralPath "$O\b1-val-baseline-score\summary.json")) {
        Move-Aside "$O\b1-val-baseline-score"
        $c = Invoke-Scorer 'validation' $ValRefs $ValRefSha @("B1-whisper-small-m6repro=$Baseline") "$O\b1-val-baseline-score" "$O\logs\b1-val-baseline-score.log"
        if ($c -ne 0) { throw 'baseline scoring failed' }
    }
}

function Step-Smoke {
    $S = "$O\smoke"; New-Item -ItemType Directory -Force -Path $S | Out-Null
    Get-Tool @('config', '--data', $D, '--baseline', "$Baseline\baseline-evidence.json", '--seed', '17', '--smoke', '--output', "$S\config.json") | Out-Null
    if (-not (Test-Path -LiteralPath "$S\train\epoch-1\checkpoint-manifest.json")) {
        Move-Aside "$S\train"
        $c = Invoke-Py $G "$O\logs\smoke-train.log" @($Tool, 'smoke-train', '--config', "$S\config.json", '--output', "$S\train")
        if ($c -ne 0) { throw 'smoke training failed' }
    }
    Write-Note "smoke peak reserved CUDA GiB: $(Get-Tool @('peak', "$S\train\progress.json"))"
    $ck = "$S\train\epoch-1"
    $msha = Get-Tool @('sha', "$ck\checkpoint-manifest.json")
    # Validation decode: real command, stopped once its first (warm-up) item is decoded, i.e. after every input check.
    $V = "$S\val-e1-partial"; Move-Aside $V
    $out = "$O\logs\smoke-val.out.log"; $err = "$O\logs\smoke-val.err.log"
    $valArgs = @('-u', 'scripts\m45_validation_adapted.py', '--checkpoint', $ck, '--checkpoint-manifest-sha256', $msha,
        '--inference', "$D\$Rel.inference.validation.jsonl", '--warmup-inference', "$D\$Rel.inference.train.jsonl", '--audio-root', $D,
        '--split-manifest', "$M\data\splits\$Rel\split_manifest.json", '--output', $V, '--system-id', 'F2-m6repro-smoke-e1')
    $p = Start-Process -FilePath $G -ArgumentList $valArgs -WorkingDirectory $M -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
    $limit = (Get-Date).AddMinutes(10); $seen = $false
    while (-not $p.HasExited -and (Get-Date) -lt $limit -and -not $seen) {
        Start-Sleep -Seconds 2
        $seen = [bool](Select-String -LiteralPath $out -Pattern 'items_including_warmup' -Quiet -ErrorAction SilentlyContinue)
    }
    if ($p.HasExited) { Get-Content -LiteralPath $err -Tail 5; throw "smoke validation decode exited early ($($p.ExitCode))" }
    if (-not $seen) { Stop-Process -Id $p.Id -Force; throw 'smoke validation decode produced no item in 10 min' }
    Stop-Process -Id $p.Id -Force
    Write-Note 'smoke validation decode: all input checks passed and first item decoded; stopped by design'
    # Test decode: real m45_final_infer.py on the first 8 test IDs (derived smoke protocol). Never scored.
    $T = "$S\test-8"; Move-Aside $T
    $psha = Get-Tool @('protocol', '--data', $D, '--checkpoint', $ck, '--system-id', 'F2-m6repro-smoke', '--epoch', '1',
        '--output', "$S\protocol-test-smoke.json", '--test-ids-subset', '8')
    $c = Invoke-Py $G "$O\logs\smoke-test8.log" @('scripts\m45_final_infer.py', '--protocol', "$S\protocol-test-smoke.json",
        '--protocol-sha256', $psha, '--system', 'F2-m6repro-smoke', '--mode', 'accuracy', '--output', $T)
    if ($c -ne 0 -or -not (Test-Path -LiteralPath "$T\completed.json")) { throw 'smoke test decode failed' }
    Write-Note 'SMOKE PASS: training + epoch checkpoint + validation decode start + test decode/export'
}

function Test-TrainDone([string]$S) {
    $run = Read-Json "$S\train\run.json"
    return ($run -and $run.status -eq 'training_completed_selection_pending' -and @($run.checkpoints).Count -eq 3)
}

function Step-Train([int]$Seed) {
    $S = "$O\seed$Seed"; New-Item -ItemType Directory -Force -Path "$S\logs" | Out-Null
    if (Test-TrainDone $S) { Write-Note "seed $Seed training: done, skipped"; return 0 }
    if (Test-Path -LiteralPath "$S\train") {
        # fit() cannot resume mid-run: an interrupted run restarts from epoch 1, with every later stage of this seed.
        foreach ($n in @('train', 'val-e1', 'val-e2', 'val-e3', 'val-e1.ok.json', 'val-e2.ok.json', 'val-e3.ok.json', 'score-val', 'selection.json', 'protocol-test.json', 'test', 'score-test')) { Move-Aside "$S\$n" }
    }
    if (-not (Test-Path -LiteralPath "$Baseline\baseline-evidence.json")) { throw 'run -Step baseline first (training gate)' }
    Get-Tool @('config', '--data', $D, '--baseline', "$Baseline\baseline-evidence.json", '--seed', "$Seed", '--output', "$S\config.json") | Out-Null
    return (Invoke-Py $G "$S\logs\train.log" @('scripts\m4_train_whisper.py', '--config', "$S\config.json", '--output', "$S\train"))
}

function Wait-Checkpoint([int]$Seed, [int]$Epoch) {
    $T = "$O\seed$Seed\train"
    while ($true) {
        $run = Read-Json "$T\run.json"
        if ($run -and (@($run.checkpoints | Where-Object { $_.epoch -eq $Epoch }).Count -eq 1)) { return $true }
        if ($run -and $run.status -eq 'failed') { Write-Note "seed $Seed training failed; evaluation stops"; return $false }
        if ((Test-Path -LiteralPath "$O\trainloop.done") -and -not (Test-Path -LiteralPath "$O\trainloop.active")) { Write-Note "seed $Seed epoch $Epoch never saved; evaluation stops"; return $false }
        Start-Sleep -Seconds 30
    }
}

function Step-Evaluate([int]$Seed) {
    $S = "$O\seed$Seed"; New-Item -ItemType Directory -Force -Path "$S\logs" | Out-Null
    $redo = $false
    foreach ($e in 1, 2, 3) {
        if (-not (Wait-Checkpoint $Seed $e)) { return 1 }
        $ck = "$S\train\epoch-$e"; $V = "$S\val-e$e"
        $msha = Get-Tool @('sha', "$ck\checkpoint-manifest.json")
        $ok = Read-Json "$V.ok.json"
        if ($ok -and $ok.checkpoint_manifest_sha256 -eq $msha) { continue }
        Move-Aside $V; Move-Aside "$V.ok.json"; $redo = $true
        $c = Invoke-Py $G "$S\logs\val-e$e.log" @('scripts\m45_validation_adapted.py', '--checkpoint', $ck, '--checkpoint-manifest-sha256', $msha,
            '--inference', "$D\$Rel.inference.validation.jsonl", '--warmup-inference', "$D\$Rel.inference.train.jsonl", '--audio-root', $D,
            '--split-manifest', "$M\data\splits\$Rel\split_manifest.json", '--output', $V, '--system-id', "F2-m6repro-s$Seed-e$e")
        if ($c -ne 0) { Write-Note "seed $Seed val-e$e decode failed"; return 1 }
        Set-Content -LiteralPath "$V.ok.json" -Value ('{"checkpoint_manifest_sha256": "' + $msha + '"}') -Encoding ASCII
    }
    if ($redo) { foreach ($n in @('score-val', 'selection.json', 'protocol-test.json', 'test', 'score-test')) { Move-Aside "$S\$n" } }
    if (-not (Test-Path -LiteralPath "$S\score-val\summary.json")) {
        Move-Aside "$S\score-val"
        $runs = @(1, 2, 3 | ForEach-Object { "F2-m6repro-s$Seed-e$_=$S\val-e$_" })
        if ((Invoke-Scorer 'validation' $ValRefs $ValRefSha $runs "$S\score-val" "$S\logs\score-val.log") -ne 0) { return 1 }
    }
    if (-not (Test-Path -LiteralPath "$S\selection.json")) { Get-Tool @('select', '--seed-dir', $S, '--seed', "$Seed") | Write-Host }
    $sel = Read-Json "$S\selection.json"
    Write-Note "seed $Seed selected epoch $($sel.selected_epoch) on validation (before any test decode)"
    if (-not (Test-Path -LiteralPath "$S\test\completed.json")) {
        Move-Aside "$S\test"
        $psha = Get-Tool @('protocol', '--data', $D, '--checkpoint', $sel.checkpoint, '--system-id', "F2-m6repro-s$Seed",
            '--epoch', "$($sel.selected_epoch)", '--output', "$S\protocol-test.json")
        $c = Invoke-Py $G "$S\logs\test.log" @('scripts\m45_final_infer.py', '--protocol', "$S\protocol-test.json", '--protocol-sha256', $psha,
            '--system', "F2-m6repro-s$Seed", '--mode', 'accuracy', '--output', "$S\test")
        if ($c -ne 0) { Write-Note "seed $Seed test decode failed"; return 1 }
    }
    if (-not (Test-Path -LiteralPath "$S\score-test\summary.json")) {
        Move-Aside "$S\score-test"
        $refs = (Get-Content -LiteralPath "$O\test-references.txt" -TotalCount 1).Trim()
        if ((Invoke-Scorer 'test' $refs $TestRefSha @("F2-m6repro-s$Seed=$S\test") "$S\score-test" "$S\logs\score-test.log") -ne 0) { return 1 }
    }
    Write-Note "seed $Seed evaluation complete"
    return 0
}

# Start seed N only if training (as long as the last measured one) plus the evaluation tail ends by -FinishBy.
function Test-ClockGate([int]$Seed) {
    if ($Seed -eq $Seeds[0] -or (Test-TrainDone "$O\seed$Seed")) { return $true }
    $minutes = 45
    foreach ($s in $Seeds) {
        $run = Read-Json "$O\seed$s\train\run.json"
        if ($run -and $run.ended_at_utc) { $minutes = ([datetime]$run.ended_at_utc - [datetime]$run.started_at_utc).TotalMinutes }
    }
    $finish = [datetime]::Today.Add([timespan]::Parse($FinishBy))
    $eta = (Get-Date).AddMinutes($minutes + $EvalTailMinutes)
    if ($eta -gt $finish) { Write-Note "seed ${Seed}: not started (needs ~$([int]($minutes + $EvalTailMinutes)) min, ETA $($eta.ToString('HH:mm')) > $FinishBy)"; return $false }
    return $true
}

function Step-TrainLoop {
    Set-Content -LiteralPath "$O\trainloop.active" -Value $PID -Encoding ASCII
    try {
        foreach ($seed in $Seeds) {
            if (Test-Path -LiteralPath "$O\STOP") { Write-Note 'STOP file: no further training'; break }
            if (-not (Test-ClockGate $seed)) { break }
            if ((Step-Train $seed) -ne 0) { Write-Note "seed $seed training failed; train loop stops"; break }
        }
    } finally {
        Remove-Item -LiteralPath "$O\trainloop.active" -ErrorAction SilentlyContinue
        Set-Content -LiteralPath "$O\trainloop.done" -Value "train loop ended $(Get-Date -Format s)" -Encoding ASCII
    }
}

function Step-EvalLoop {
    try { foreach ($seed in $Seeds) { if ((Step-Evaluate $seed) -ne 0) { break } } }
    finally { Set-Content -LiteralPath "$O\evalloop.done" -Value "eval loop ended $(Get-Date -Format s)" -Encoding ASCII }
}

function Step-Serial {
    Set-Content -LiteralPath "$O\trainloop.active" -Value $PID -Encoding ASCII
    try {
        foreach ($seed in $Seeds) {
            if (Test-Path -LiteralPath "$O\STOP") { break }
            if (-not (Test-ClockGate $seed)) { break }
            if ((Step-Train $seed) -ne 0) { break }
            if ((Step-Evaluate $seed) -ne 0) { break }
        }
    } finally {
        Remove-Item -LiteralPath "$O\trainloop.active" -ErrorAction SilentlyContinue
        Set-Content -LiteralPath "$O\trainloop.done" -Value "serial loop ended $(Get-Date -Format s)" -Encoding ASCII
        Set-Content -LiteralPath "$O\evalloop.done" -Value "serial loop ended $(Get-Date -Format s)" -Encoding ASCII
    }
}

function Step-Pipeline {
    $pids = Read-Json "$O\pipeline-pids.json"
    if ($pids) { foreach ($id in @($pids.pids)) { if (Get-Process -Id $id -ErrorAction SilentlyContinue) { throw "pipeline already running (PID $id); use -Step status" } } }
    foreach ($n in @('trainloop.done', 'evalloop.done', 'trainloop.active')) { Remove-Item -LiteralPath "$O\$n" -ErrorAction SilentlyContinue }
    $shell = (Get-Process -Id $PID).Path
    $common = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $Self, '-SeedList', ($Seeds -join ','), '-FinishBy', $FinishBy)
    $peak = [double](Get-Tool @('peak', "$O\smoke\train\progress.json"))
    $started = @()
    $started += Start-Process -FilePath $G -ArgumentList @($Tool, 'keepawake', '--until-file', "$O\evalloop.done") -WindowStyle Hidden -PassThru
    if ($peak -gt 0 -and $peak -le $OverlapMaxPeakGiB) {
        Write-Note "pipeline: overlap mode (smoke peak $peak GiB): validation/test of seed N run while seed N+1 trains"
        Set-Content -LiteralPath "$O\trainloop.active" -Value 'starting' -Encoding ASCII
        $started += Start-Process -FilePath $shell -ArgumentList ($common + @('-Step', 'trainloop')) -WindowStyle Minimized -PassThru
        $started += Start-Process -FilePath $shell -ArgumentList ($common + @('-Step', 'evalloop')) -WindowStyle Minimized -PassThru
    } else {
        Write-Note "pipeline: serial mode (smoke peak $peak GiB)"
        $started += Start-Process -FilePath $shell -ArgumentList ($common + @('-Step', 'serial')) -WindowStyle Minimized -PassThru
    }
    $ids = ($started | ForEach-Object { $_.Id }) -join ', '
    Set-Content -LiteralPath "$O\pipeline-pids.json" -Value ('{"pids": [' + $ids + ']}') -Encoding ASCII
    Write-Note "pipeline started, PIDs $ids (keep-awake, loops). Poll with -Step status."
}

switch ($Step) {
    'preflight' { Step-Preflight }
    'baseline' { Step-Baseline }
    'smoke' { Step-Smoke }
    'train' { foreach ($s in $Seeds) { if ((Step-Train $s) -ne 0) { throw "seed $s training failed" } } }
    'evaluate' { foreach ($s in $Seeds) { if ((Step-Evaluate $s) -ne 0) { throw "seed $s evaluation failed" } } }
    'trainloop' { Step-TrainLoop }
    'evalloop' { Step-EvalLoop }
    'serial' { Step-Serial }
    'pipeline' { Step-Pipeline }
    'status' { & $G $Tool status --root $O; Get-Content -LiteralPath "$O\pipeline.log" -Tail 8 -ErrorAction SilentlyContinue }
    'compare' { & $G $Tool compare --root $O --seeds @Seeds }
}
