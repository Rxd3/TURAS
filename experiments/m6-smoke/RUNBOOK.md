# 5070 Ti smoke-test runbook

Run these commands on the RTX 5070 Ti laptop, in one PowerShell window, top to bottom.
Run them one at a time and wait for each to finish. Every path is absolute, so the
current folder does not matter. Do not edit any file in the repository.

Before you start: plug the laptop in, close other programs that use the GPU (games,
browsers with video, other Python sessions).

## 1. Get the branch

If `C:\Users\benda\Downloads\nlp\TURAS` does not exist yet:

```powershell
git clone https://github.com/Rxd3/TURAS C:\Users\benda\Downloads\nlp\TURAS
git -C C:\Users\benda\Downloads\nlp\TURAS switch feat/m6-smoke-tests
```

If it already exists:

```powershell
git -C C:\Users\benda\Downloads\nlp\TURAS fetch origin
git -C C:\Users\benda\Downloads\nlp\TURAS switch feat/m6-smoke-tests
git -C C:\Users\benda\Downloads\nlp\TURAS pull --ff-only
```

Check: `git -C C:\Users\benda\Downloads\nlp\TURAS status` says `On branch feat/m6-smoke-tests`.

## 2. Install the checker package into the existing environment

```powershell
C:\Users\benda\Downloads\nlp\project3-asr\.venv\Scripts\python.exe -m pip install --no-deps -e C:\Users\benda\Downloads\nlp\TURAS
C:\Users\benda\Downloads\nlp\project3-asr\.venv\Scripts\python.exe -c "import turkish_asr_rdl, torch, transformers, peft, pyarrow, huggingface_hub; print(turkish_asr_rdl.__file__, torch.cuda.is_available())"
```

Check: the second command prints a path under `C:\Users\benda\Downloads\nlp\TURAS\src` and `True`.
If it prints anything else, stop and send back its output.

## 3. Run the smoke tests, one model at a time

Each command downloads its model on first use (Hugging Face cache on `C:`), runs alone,
writes `C:\turas-artifacts\smoke\<NAME>.json` and prints one line: `<NAME>: ok -> ...` or
`<NAME>: FAILED: ... -> ...`. A FAILED line does not stop the run: go on with the next
command. The first command also downloads one 77 MB train shard and writes the clips to
`C:\turas-artifacts\smoke\audio\`; later commands reuse them.

Inference:

```powershell
C:\Users\benda\Downloads\nlp\project3-asr\.venv\Scripts\python.exe C:\Users\benda\Downloads\nlp\TURAS\experiments\m6-smoke\smoke.py --model whisper-small --artifacts C:\turas-artifacts --clips 20
C:\Users\benda\Downloads\nlp\project3-asr\.venv\Scripts\python.exe C:\Users\benda\Downloads\nlp\TURAS\experiments\m6-smoke\smoke.py --model whisper-large-v3 --artifacts C:\turas-artifacts --clips 20
C:\Users\benda\Downloads\nlp\project3-asr\.venv\Scripts\python.exe C:\Users\benda\Downloads\nlp\TURAS\experiments\m6-smoke\smoke.py --model qwen3-asr-1.7b --artifacts C:\turas-artifacts --clips 20
C:\Users\benda\Downloads\nlp\project3-asr\.venv\Scripts\python.exe C:\Users\benda\Downloads\nlp\TURAS\experiments\m6-smoke\smoke.py --model qwen3-asr-0.6b --artifacts C:\turas-artifacts --clips 20
C:\Users\benda\Downloads\nlp\project3-asr\.venv\Scripts\python.exe C:\Users\benda\Downloads\nlp\TURAS\experiments\m6-smoke\smoke.py --model xlsr-tr-cv7 --artifacts C:\turas-artifacts --clips 20
C:\Users\benda\Downloads\nlp\project3-asr\.venv\Scripts\python.exe C:\Users\benda\Downloads\nlp\TURAS\experiments\m6-smoke\smoke.py --model xlsr-tr-cv8 --artifacts C:\turas-artifacts --clips 20
```

Training memory (3 optimizer steps each; `train-whisper-large-v3-full` is expected to run
out of memory, and that still counts as `ok`):

```powershell
C:\Users\benda\Downloads\nlp\project3-asr\.venv\Scripts\python.exe C:\Users\benda\Downloads\nlp\TURAS\experiments\m6-smoke\smoke.py --model train-whisper-small-full --artifacts C:\turas-artifacts --clips 20
C:\Users\benda\Downloads\nlp\project3-asr\.venv\Scripts\python.exe C:\Users\benda\Downloads\nlp\TURAS\experiments\m6-smoke\smoke.py --model train-whisper-large-v3-lora --artifacts C:\turas-artifacts --clips 20
C:\Users\benda\Downloads\nlp\project3-asr\.venv\Scripts\python.exe C:\Users\benda\Downloads\nlp\TURAS\experiments\m6-smoke\smoke.py --model train-whisper-large-v3-full --artifacts C:\turas-artifacts --clips 20
```

PyTorch Geometric, last. Install it first and copy the `Successfully installed ...` line
that pip prints at the end; you will send it back:

```powershell
C:\Users\benda\Downloads\nlp\project3-asr\.venv\Scripts\python.exe -m pip install torch_geometric==2.8.1
C:\Users\benda\Downloads\nlp\project3-asr\.venv\Scripts\python.exe C:\Users\benda\Downloads\nlp\TURAS\experiments\m6-smoke\smoke.py --model pyg-hetero --artifacts C:\turas-artifacts --clips 20
```

`pyg-hetero` says FAILED only when the fp32 check fails; if only bf16 fails, it says ok and the
JSON shows `"outcome": "fp32 only"`. Only if the `pyg-hetero` line says FAILED: remove exactly the packages that pip named on
its `Successfully installed` line (expected: `torch_geometric`, `aiohttp`, `pyparsing` and
aiohttp's own dependencies). Write the names without their version numbers, separated by
spaces, in place of `<names>`:

```powershell
C:\Users\benda\Downloads\nlp\project3-asr\.venv\Scripts\python.exe -m pip uninstall -y <names>
```

If `pyg-hetero` says ok, leave `torch_geometric` installed.

## 4. Write the report

```powershell
C:\Users\benda\Downloads\nlp\project3-asr\.venv\Scripts\python.exe C:\Users\benda\Downloads\nlp\TURAS\experiments\m6-smoke\smoke.py --summarize C:\turas-artifacts
```

It prints `wrote C:\turas-artifacts\smoke\REPORT.md`.

## 5. Send back by message

1. The full text of `C:\turas-artifacts\smoke\REPORT.md`.
2. The full text of each of these ten files, each under its file name:
   `whisper-small.json`, `whisper-large-v3.json`, `qwen3-asr-1.7b.json`, `qwen3-asr-0.6b.json`,
   `xlsr-tr-cv7.json`, `xlsr-tr-cv8.json`, `train-whisper-small-full.json`,
   `train-whisper-large-v3-lora.json`, `train-whisper-large-v3-full.json`, `pyg-hetero.json`
   (all in `C:\turas-artifacts\smoke\`). A missing file means its command never ran: say so.
3. The one-line output of each `smoke.py` command, in order.
4. The `Successfully installed ...` line from the `torch_geometric` install, and whether you ran
   the uninstall command.

Do not send the audio clips or the `C:\turas-artifacts\smoke\runs\` folder; they stay on this
laptop.

## Re-run after the dedup fix

The export now removes duplicate N-best candidates, so only the four beam-search models run
again. First update the branch with the "If it already exists" commands in step 1. Each
command overwrites that model's JSON file; the other six result files stay and appear in the
new report. Send back the report and the four JSON files as in step 5.

```powershell
$py = "$env:USERPROFILE\Downloads\nlp\project3-asr\.venv\Scripts\python.exe"
$smoke = "$env:USERPROFILE\Downloads\nlp\TURAS\experiments\m6-smoke\smoke.py"
& $py $smoke --model whisper-small --artifacts C:\turas-artifacts --clips 20
& $py $smoke --model whisper-large-v3 --artifacts C:\turas-artifacts --clips 20
& $py $smoke --model qwen3-asr-1.7b --artifacts C:\turas-artifacts --clips 20
& $py $smoke --model qwen3-asr-0.6b --artifacts C:\turas-artifacts --clips 20
& $py $smoke --summarize C:\turas-artifacts
```
