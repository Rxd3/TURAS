"""Print the environment fingerprint and exit 0 (READY) or 1 (NOT READY)."""
try:
    from turkish_asr_rdl.utils.env_check import main
except ImportError:
    print("turkish_asr_rdl is not importable: activate .venv and run "
          "pip install --no-deps -e . from the repository root")
    raise SystemExit(1)

raise SystemExit(main())
