# M6 reproduction of F2 (and F3, F1) on the RTX 5070 Ti. Windows PowerShell 5.1 and PowerShell 7.
# One GPU job at a time. Every step is resumable: a finished stage is skipped, an unfinished one is
# moved aside (renamed *.stale-<time>, never deleted) and redone.
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('preflight', 'baseline', 'smoke', 'start', 'chain', 'status', 'compare')]
    [string]$Step,
    [string]$Jobs = 'F2:17,F3:17,F2:18,F1:17,F2:19',
    [string]$FinishBy = '05:45',
    [string]$Artifacts = 'C:\turas-artifacts'
)
$ErrorActionPreference = 'Continue'
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
$Train = @{ F1 = 'scripts\m4_train_whisper.py'; F2 = 'scripts\m4_train_whisper.py'; F3 = 'scripts\m5_train_ctc.py' }
$Family = @{ F1 = 'whisper'; F2 = 'whisper'; F3 = 'ctc' }
$Gate = @{ whisper = "$O\b1-val-baseline"; ctc = "$O\b4-val-baseline" }
# Minutes, estimates for the clock gate: training; per-epoch validation decode; test decode; scoring, selection and oracle.
$Est = @{ F1 = @(35, 7, 10, 4); F2 = @(35, 7, 10, 4); F3 = @(45, 16, 23, 4) }
$GateMinutes = @{ whisper = 9; ctc = 18 }

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

function Get-UtcStamp { return (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ss') + '+00:00' }

function Set-Status([hashtable]$Fields) {
    $Fields['updated_utc'] = Get-UtcStamp
    $tmp = "$O\status.tmp"
    Set-Content -LiteralPath $tmp -Value ($Fields | ConvertTo-Json -Compress) -Encoding ASCII
    Move-Item -LiteralPath $tmp -Destination "$O\status.json" -Force
}

# Run one python step as its own process, stdout/stderr to <Log>.out.log/<Log>.err.log; return its exit code.
function Invoke-Py([string]$Python, [string]$Log, [string[]]$PyArgs, [string]$Job = '', [string]$Name = '') {
    $out = "$Log.out.log"; $err = "$Log.err.log"
    foreach ($f in @($out, $err)) { if (Test-Path -LiteralPath $f) { Move-Aside $f } }
    $p = Start-Process -FilePath $Python -ArgumentList (@('-u') + $PyArgs) -WorkingDirectory $M -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
    $null = $p.Handle  # keeps ExitCode readable after exit
    $started = Get-UtcStamp
    Set-Status @{ state = 'running'; job = $Job; step = $Name; pid = $p.Id; step_started_utc = $started; log_out = $out; log_err = $err; job_dir = (Split-Path -Leaf (Split-Path -Parent (Split-Path -Parent $Log))) }
    Write-Note "start $Job $Name (PID $($p.Id)) -> $out"
    $p.WaitForExit()
    $code = $p.ExitCode
    $state = 'done'; if ($code -ne 0) { $state = 'failed' }
    Set-Status @{ state = $state; job = $Job; step = $Name; exit_code = $code; step_started_utc = $started; log_out = $out; log_err = $err; job_dir = '' }
    Write-Note "end $Job $Name exit=$code"
    if ($code -ne 0) { Get-Content -LiteralPath $err -Tail 3 -ErrorAction SilentlyContinue | ForEach-Object { Write-Note "  stderr: $_" } }
    return $code
}

function Get-Tool([string[]]$ToolArgs) {
    $result = & $G $Tool @ToolArgs
    if ($LASTEXITCODE -ne 0) { throw "f2repro.py $($ToolArgs[0]) failed" }
    return ($result | Select-Object -Last 1)
}

function Move-Aside([string]$Path) {
    if (Test-Path -LiteralPath $Path) {
        $name = (Split-Path -Leaf $Path) + '.stale-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
        Rename-Item -LiteralPath $Path -NewName $name -ErrorAction Stop
        Write-Note "moved aside $Path -> $name"
    }
}

function Read-Json([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try { return (Get-Content -LiteralPath $Path -Raw -Encoding UTF8 -ErrorAction Stop | ConvertFrom-Json) } catch { return $null }
}

function Get-JobDir([string]$Sys, [int]$Seed) {
    if ($Sys -eq 'F2') { return "$O\seed$Seed" }
    return "$O\$($Sys.ToLower())-seed$Seed"
}

function Get-TestRefs {
    $hit = Get-ChildItem -LiteralPath "$Artifacts\m1-release" -Recurse -File -Filter "$Rel.references.test.jsonl" -ErrorAction SilentlyContinue |
        Where-Object { (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower() -eq $TestRefSha } | Select-Object -First 1
    if ($hit) { return $hit.FullName }
    Write-Note "no M1 archive copy with the payload hash under $Artifacts\m1-release; using the published copy in pr26"
    return "$D\$Rel.references.test.jsonl"
}

function Invoke-Scorer([string]$Split, [string]$Refs, [string]$RefSha, [string[]]$Runs, [string]$Output, [string]$Log, [string]$Job) {
    $runArgs = @(); foreach ($item in $Runs) { $runArgs += @('--run', $item) }
    $a = @('scripts\m6_score_baselines.py', '--protocol', 'configs\evaluation\m45_final_protocol.json', '--protocol-sha256', $ProtocolSha,
        '--split', $Split, '--references', $Refs, '--references-sha256', $RefSha) + $runArgs + @('--output', $Output)
    return (Invoke-Py $Lean $Log $a $Job "score-$Split")
}

function Show-Tail([string]$Log) { Get-Content -LiteralPath "$Log.out.log" -Tail 4 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ } }

function Step-Preflight {
    $refs = Get-TestRefs
    Set-Content -LiteralPath "$O\test-references.txt" -Value $refs -Encoding ASCII
    $c1 = Invoke-Py $G "$O\logs\preflight-gpu" @($Tool, 'preflight', '--data', $D, '--test-refs', $refs, '--gpu') 'setup' 'preflight-gpu'
    Get-Content -LiteralPath "$O\logs\preflight-gpu.out.log"
    $c2 = Invoke-Py $Lean "$O\logs\preflight-lean" @($Tool, 'preflight') 'setup' 'preflight-lean'
    Show-Tail "$O\logs\preflight-lean"
    & $Lean -c "import jiwer, numpy, pandas; print('lean venv imports jiwer, numpy, pandas: ok')"
    if ($c1 -ne 0 -or $c2 -ne 0 -or $LASTEXITCODE -ne 0) { throw 'preflight FAILED: fix before any GPU step' }
}

# The unmodified-parent validation decode that run_training requires (inputs.baseline_report), then its M6 score.
function Step-Gate([string]$Sys) {
    $fam = $Family[$Sys]; $B = $Gate[$fam]; $label = @{ whisper = 'B1-whisper-small'; ctc = 'B4-xlsr-tr-cv7' }[$fam]
    if (-not (Test-Path -LiteralPath "$B\baseline-evidence.json")) {
        Move-Aside $B
        Get-Tool @('config', '--system', $Sys, '--data', $D, '--baseline', "$B\baseline-evidence.json", '--seed', '17', '--output', "$O\baseline-config-$fam.json") | Out-Null
        $c = Invoke-Py $G "$O\logs\$(Split-Path -Leaf $B)" @('scripts\m45_validation_baseline.py', '--config', "$O\baseline-config-$fam.json",
            '--family', $fam, '--inference', "$D\$Rel.inference.validation.jsonl", '--warmup-inference', "$D\$Rel.inference.train.jsonl",
            '--audio-root', $D, '--output', $B) 'gate' "baseline-$fam"
        if ($c -ne 0) { return $c }
    }
    if (-not (Test-Path -LiteralPath "$B-score\summary.json")) {
        Move-Aside "$B-score"
        $c = Invoke-Scorer 'validation' $ValRefs $ValRefSha @("$label-m6repro=$B") "$B-score" "$O\logs\$(Split-Path -Leaf $B)-score" 'gate'
        if ($c -ne 0) { return $c }
        Show-Tail "$O\logs\$(Split-Path -Leaf $B)-score"
    }
    return 0
}

function Step-Smoke {
    $S = "$O\smoke"; New-Item -ItemType Directory -Force -Path "$S\logs" | Out-Null
    if (-not (Test-Path -LiteralPath "$O\b1-val-baseline\baseline-evidence.json")) { throw 'run -Step baseline first (training gate)' }
    Get-Tool @('config', '--system', 'F2', '--data', $D, '--baseline', "$O\b1-val-baseline\baseline-evidence.json", '--seed', '17', '--smoke', '--output', "$S\config.json") | Out-Null
    if (-not (Test-Path -LiteralPath "$S\train\epoch-1\checkpoint-manifest.json")) {
        Move-Aside "$S\train"
        $c = Invoke-Py $G "$S\logs\train" @($Tool, 'smoke-train', '--config', "$S\config.json", '--output', "$S\train") 'smoke' 'train'
        Show-Tail "$S\logs\train"
        if ($c -ne 0) { throw 'smoke training failed' }
    }
    Write-Note "smoke peak reserved CUDA GiB: $(Get-Tool @('peak', "$S\train\progress.json"))"
    $ck = "$S\train\epoch-1"
    $msha = Get-Tool @('sha', "$ck\checkpoint-manifest.json")
    # Validation decode: the real command, stopped once its first (warm-up) item is decoded, i.e. after every input check.
    $V = "$S\val-e1-partial"; Move-Aside $V
    $out = "$S\logs\val-partial.out.log"; $err = "$S\logs\val-partial.err.log"
    foreach ($f in @($out, $err)) { if (Test-Path -LiteralPath $f) { Move-Aside $f } }
    $valArgs = @('-u', 'scripts\m45_validation_adapted.py', '--checkpoint', $ck, '--checkpoint-manifest-sha256', $msha,
        '--inference', "$D\$Rel.inference.validation.jsonl", '--warmup-inference', "$D\$Rel.inference.train.jsonl", '--audio-root', $D,
        '--split-manifest', "$M\data\splits\$Rel\split_manifest.json", '--output', $V, '--system-id', 'F2-m6repro-smoke-e1')
    $p = Start-Process -FilePath $G -ArgumentList $valArgs -WorkingDirectory $M -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
    $null = $p.Handle
    $limit = (Get-Date).AddMinutes(10); $seen = $false
    while (-not $p.HasExited -and (Get-Date) -lt $limit -and -not $seen) {
        Start-Sleep -Seconds 2
        $seen = [bool](Select-String -LiteralPath $out -Pattern 'items_including_warmup' -Quiet -ErrorAction SilentlyContinue)
    }
    if ($p.HasExited) { Get-Content -LiteralPath $err -Tail 5; throw "smoke validation decode exited early ($($p.ExitCode))" }
    if (-not $seen) { Stop-Process -Id $p.Id -Force; throw 'smoke validation decode produced no item in 10 min' }
    Stop-Process -Id $p.Id -Force
    Write-Note 'smoke validation decode: all input checks passed and the first item decoded; stopped by design'
    # Test decode: the real m45_final_infer.py on the first 8 test IDs (derived smoke protocol). Never scored.
    $T = "$S\test-8"; Move-Aside $T
    $psha = Get-Tool @('protocol', '--data', $D, '--checkpoint', $ck, '--system-id', 'F2-m6repro-smoke', '--family', 'whisper',
        '--epoch', '1', '--output', "$S\protocol-test-smoke.json", '--test-ids-subset', '8')
    $c = Invoke-Py $G "$S\logs\test-8" @('scripts\m45_final_infer.py', '--protocol', "$S\protocol-test-smoke.json",
        '--protocol-sha256', $psha, '--system', 'F2-m6repro-smoke', '--mode', 'accuracy', '--output', $T) 'smoke' 'test-8'
    Show-Tail "$S\logs\test-8"
    if ($c -ne 0 -or -not (Test-Path -LiteralPath "$T\completed.json")) { throw 'smoke test decode failed' }
    Write-Note 'SMOKE PASS: training + epoch checkpoint + validation decode start + test decode/export'
}

function Test-TrainDone([string]$S) {
    $run = Read-Json "$S\train\run.json"
    return ($run -and $run.status -eq 'training_completed_selection_pending' -and @($run.checkpoints).Count -eq 3)
}

# Remaining minutes of a job, from what is already finished and the latest measured whisper training time.
function Get-Remaining([string]$Sys, [int]$Seed) {
    $S = Get-JobDir $Sys $Seed; $e = $Est[$Sys]; $trainMinutes = $e[0]; $latest = $null
    foreach ($dir in (Get-ChildItem -LiteralPath $O -Directory -ErrorAction SilentlyContinue)) {
        $run = Read-Json "$($dir.FullName)\train\run.json"
        if ($Sys -ne 'F3' -and $dir.Name -match '^(seed|f1-seed)\d+$' -and $run -and $run.status -eq 'training_completed_selection_pending') {
            $ended = [datetime]$run.ended_at_utc
            if (-not $latest -or $ended -gt $latest) { $latest = $ended; $trainMinutes = ($ended - [datetime]$run.started_at_utc).TotalMinutes }
        }
    }
    $total = 0
    if (-not (Test-Path -LiteralPath "$($Gate[$Family[$Sys]])\baseline-evidence.json")) { $total += $GateMinutes[$Family[$Sys]] }
    if (-not (Test-TrainDone $S)) { $total += $trainMinutes }
    foreach ($ep in 1, 2, 3) { if (-not (Test-Path -LiteralPath "$S\val-e$ep.ok.json")) { $total += $e[1] } }
    if (-not (Test-Path -LiteralPath "$S\test\completed.json")) { $total += $e[2] }
    return ($total + $e[3])
}

function Test-Clock([string]$What, [double]$Minutes) {
    $finish = [datetime]::Today.Add([timespan]::Parse($FinishBy))
    $eta = (Get-Date).AddMinutes($Minutes)
    if ($eta -gt $finish) { Write-Note "skip ${What}: needs ~$([int]$Minutes) min, ETA $($eta.ToString('HH:mm')) after $FinishBy"; return $false }
    return $true
}

function Step-Train([string]$Sys, [int]$Seed) {
    $S = Get-JobDir $Sys $Seed; New-Item -ItemType Directory -Force -Path "$S\logs" | Out-Null
    if (Test-TrainDone $S) { return 0 }
    if (Test-Path -LiteralPath "$S\train") {
        # fit() cannot resume mid-run: an interrupted run restarts from epoch 1, with every later stage of this job.
        foreach ($n in @('train', 'val-e1', 'val-e2', 'val-e3', 'val-e1.ok.json', 'val-e2.ok.json', 'val-e3.ok.json', 'score-val', 'selection.json', 'protocol-test.json', 'test', 'score-test')) { Move-Aside "$S\$n" }
    }
    Get-Tool @('config', '--system', $Sys, '--data', $D, '--baseline', "$($Gate[$Family[$Sys]])\baseline-evidence.json", '--seed', "$Seed", '--output', "$S\config.json") | Out-Null
    return (Invoke-Py $G "$S\logs\train" @($Train[$Sys], '--config', "$S\config.json", '--output', "$S\train") "${Sys}:$Seed" 'train')
}

function Step-Validate([string]$Sys, [int]$Seed) {
    $S = Get-JobDir $Sys $Seed; $prefix = "$Sys-m6repro-s$Seed"; $job = "${Sys}:$Seed"; $redo = $false
    foreach ($ep in 1, 2, 3) {
        $ck = "$S\train\epoch-$ep"; $V = "$S\val-e$ep"
        $msha = Get-Tool @('sha', "$ck\checkpoint-manifest.json")
        $ok = Read-Json "$V.ok.json"
        if ($ok -and $ok.checkpoint_manifest_sha256 -eq $msha) { continue }
        if (-not (Test-Clock "$job val-e$ep" $Est[$Sys][1])) { return 9 }
        Move-Aside $V; Move-Aside "$V.ok.json"; $redo = $true
        $c = Invoke-Py $G "$S\logs\val-e$ep" @('scripts\m45_validation_adapted.py', '--checkpoint', $ck, '--checkpoint-manifest-sha256', $msha,
            '--inference', "$D\$Rel.inference.validation.jsonl", '--warmup-inference', "$D\$Rel.inference.train.jsonl", '--audio-root', $D,
            '--split-manifest', "$M\data\splits\$Rel\split_manifest.json", '--output', $V, '--system-id', "$prefix-e$ep") $job "val-e$ep"
        if ($c -ne 0) { return $c }
        Set-Content -LiteralPath "$V.ok.json" -Value ('{"checkpoint_manifest_sha256": "' + $msha + '"}') -Encoding ASCII
    }
    if ($redo) { foreach ($n in @('score-val', 'selection.json', 'protocol-test.json', 'test', 'score-test')) { Move-Aside "$S\$n" } }
    if (-not (Test-Path -LiteralPath "$S\score-val\summary.json")) {
        Move-Aside "$S\score-val"
        $runs = @(1, 2, 3 | ForEach-Object { "$prefix-e$_=$S\val-e$_" })
        $c = Invoke-Scorer 'validation' $ValRefs $ValRefSha $runs "$S\score-val" "$S\logs\score-val" $job
        if ($c -ne 0) { return $c }
    }
    if (-not (Test-Path -LiteralPath "$S\selection.json")) { Get-Tool @('select', '--job-dir', $S, '--prefix', $prefix) | ForEach-Object { Write-Note $_ } }
    return 0
}

function Step-Test([string]$Sys, [int]$Seed) {
    $S = Get-JobDir $Sys $Seed; $prefix = "$Sys-m6repro-s$Seed"; $job = "${Sys}:$Seed"
    $sel = Read-Json "$S\selection.json"
    if (-not (Test-Path -LiteralPath "$S\test\completed.json")) {
        if (-not (Test-Clock "$job test" $Est[$Sys][2])) { return 9 }
        Move-Aside "$S\test"
        $psha = Get-Tool @('protocol', '--data', $D, '--checkpoint', $sel.checkpoint, '--system-id', $prefix, '--family', $Family[$Sys],
            '--epoch', "$($sel.selected_epoch)", '--output', "$S\protocol-test.json")
        $c = Invoke-Py $G "$S\logs\test" @('scripts\m45_final_infer.py', '--protocol', "$S\protocol-test.json", '--protocol-sha256', $psha,
            '--system', $prefix, '--mode', 'accuracy', '--output', "$S\test") $job 'test'
        if ($c -ne 0) { return $c }
    }
    $refs = (Get-Content -LiteralPath "$O\test-references.txt" -TotalCount 1).Trim()
    if (-not (Test-Path -LiteralPath "$S\score-test\summary.json")) {
        Move-Aside "$S\score-test"
        $c = Invoke-Scorer 'test' $refs $TestRefSha @("$prefix=$S\test") "$S\score-test" "$S\logs\score-test" $job
        if ($c -ne 0) { return $c }
    }
    if (-not (Test-Path -LiteralPath "$S\test\oracle.json")) {
        $c = Invoke-Py $Lean "$S\logs\oracle" @('scripts\compute_nbest_oracle.py', '--references', $refs, '--nbest', "$prefix=$S\test\nbest.jsonl",
            '--output', "$S\test\oracle.json", '--view', 'tr-normalized-v1') $job 'oracle'
        if ($c -ne 0) { return $c }
    }
    return 0
}

function Step-Job([string]$Sys, [int]$Seed) {
    $job = "${Sys}:$Seed"; $S = Get-JobDir $Sys $Seed
    if (Test-Path -LiteralPath "$S\test\oracle.json") { Write-Note "$job complete, skipped"; return }
    if (-not (Test-Clock $job (Get-Remaining $Sys $Seed))) { return }
    if ($Sys -eq 'F3') {
        & $G $Tool cached --system F3 | Out-Null
        if ($LASTEXITCODE -ne 0) { Write-Note "skip ${job}: XLS-R parent not in the offline cache (download needs user approval)"; return }
    }
    if ((Step-Gate $Sys) -ne 0) { Write-Note "$job baseline gate failed; job skipped"; return }
    if ((Step-Train $Sys $Seed) -ne 0) { Write-Note "$job training failed; job skipped"; return }
    if ((Step-Validate $Sys $Seed) -ne 0) { Write-Note "$job validation stage stopped"; return }
    if ((Step-Test $Sys $Seed) -ne 0) { Write-Note "$job test stage stopped"; return }
    Write-Note "$job complete"
}

function Step-Chain {
    Set-Content -LiteralPath "$O\chain.pid" -Value $PID -Encoding ASCII
    Remove-Item -LiteralPath "$O\chain.done" -ErrorAction SilentlyContinue
    try {
        foreach ($item in $Jobs.Split(',')) {
            if (Test-Path -LiteralPath "$O\STOP") { Write-Note 'STOP file found: chain ends'; break }
            $sys, $seed = $item.Split(':')
            try { Step-Job $sys ([int]$seed) } catch { Write-Note "$item failed: $_" }
        }
    } finally {
        Set-Status @{ state = 'chain finished'; job = ''; step = ''; job_dir = '' }
        Set-Content -LiteralPath "$O\chain.done" -Value "chain ended $(Get-Date -Format s)" -Encoding ASCII
        Write-Note 'chain ended'
    }
}

function Step-Start {
    $old = Read-Json "$O\chain.pid"
    $proc = $null; if ($old) { $proc = Get-Process -Id ([int]$old) -ErrorAction SilentlyContinue }
    $st = Read-Json "$O\status.json"
    if ($st -and $st.state -eq 'running' -and $st.pid -and (Get-Process -Id ([int]$st.pid) -ErrorAction SilentlyContinue)) { throw "a chain step is still running (PID $($st.pid)); wait or stop it first" }
    if ($proc -and $proc.ProcessName -in @('powershell', 'pwsh') -and -not (Test-Path -LiteralPath "$O\chain.done")) { throw "chain already running (PID $old); use -Step status" }
    Remove-Item -LiteralPath "$O\chain.done" -ErrorAction SilentlyContinue
    $shell = (Get-Process -Id $PID).Path
    $awake = Start-Process -FilePath $G -ArgumentList @($Tool, 'keepawake', '--until-file', "$O\chain.done") -WindowStyle Hidden -PassThru
    $chain = Start-Process -FilePath $shell -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $Self, '-Step', 'chain', '-Jobs', $Jobs, '-FinishBy', $FinishBy, '-Artifacts', $Artifacts) `
        -WindowStyle Hidden -RedirectStandardOutput "$O\logs\chain.out.log" -RedirectStandardError "$O\logs\chain.err.log" -PassThru
    Write-Note "chain started: PID $($chain.Id) (keep-awake PID $($awake.Id)), jobs $Jobs, finish by $FinishBy"
}

switch ($Step) {
    'preflight' { Step-Preflight }
    'baseline' { if ((Step-Gate 'F2') -ne 0) { throw 'B1 baseline gate failed' } }
    'smoke' { Step-Smoke }
    'start' { Step-Start }
    'chain' { Step-Chain }
    'status' { & $G $Tool status --root $O; exit $LASTEXITCODE }
    'compare' { & $G $Tool compare --root $O --jobs $Jobs; exit $LASTEXITCODE }
}
