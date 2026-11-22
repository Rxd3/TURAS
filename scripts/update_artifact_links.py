"""Insert the user-supplied public Drive folder URL without publishing private roles."""
import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- TURAS_ARTIFACT_LINKS_START -->"
END = "<!-- TURAS_ARTIFACT_LINKS_END -->"


def public_folder_url(value):
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.hostname != "drive.google.com" or not re.search(r"/folders/[A-Za-z0-9_-]+", parsed.path):
        raise argparse.ArgumentTypeError("Use the actual https://drive.google.com/.../folders/ID URL")
    return value


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--public-drive-url", type=public_folder_url, required=True)
    args = parser.parse_args()
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("README must contain exactly one artifact-link marker pair")
    block = (f"{START}\n## Reproduction downloads\n\n"
             f"[Public submission and presentation]({args.public_drive_url})\n\n"
             "Audio, references and model/run bundles use separate restricted role folders. "
             "See [the package map](docs/drive_packages.md) and [reproduction commands](docs/final_reproduction.md).\n"
             f"{END}")
    before, rest = text.split(START)
    _, after = rest.split(END)
    readme.write_text(before + block + after, encoding="utf-8")
    (ROOT / "docs/public_artifact_link.json").write_text(json.dumps({
        "public_submission_url": args.public_drive_url,
        "restricted_role_links_published": False,
        "upload_or_access_verification_claimed": False}, indent=2) + "\n", encoding="utf-8")
    print("README public folder link updated; Drive permissions are unchanged.")


if __name__ == "__main__":
    main()
