"""Safe replacement for the legacy in-place freeze command.

Use --workspace, --records, --adjudication-plan, --out, --variant and
--sampled-verification. Frozen splits/references are never overwritten.
Example: python scripts/m1_freeze_release.py --help
"""
import sys
from m1_verification import main


if __name__=='__main__':
    sys.argv.insert(1,'publish')
    raise SystemExit(main())
