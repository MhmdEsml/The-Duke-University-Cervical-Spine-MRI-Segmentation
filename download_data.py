import os
import argparse
import subprocess
import zipfile
import shutil
import sys
from pathlib import Path


def run(cmd: str):
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(f"Command failed: {cmd}")


def ensure_gen3_client():
    if Path("gen3-client").exists():
        return

    run(
        "curl -s https://api.github.com/repos/uc-cdis/cdis-data-client/releases/latest "
        "| grep browser_download_url.*linux | cut -d '\"' -f 4 | wget -qi -"
    )

    with zipfile.ZipFile("dataclient_linux.zip", "r") as z:
        z.extractall(".")

    os.chmod("gen3-client", 0o755)


def organize_nifti_files(base_dir: Path):
    img_dir = base_dir / "images"
    mask_dir = base_dir / "masks"

    img_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    for root, _, files in os.walk(base_dir):
        for fname in files:
            if not fname.endswith(".nii.gz"):
                continue

            src = Path(root) / fname
            if src.parent in (img_dir, mask_dir):
                continue

            dst = mask_dir if "_SEG" in fname else img_dir
            shutil.move(str(src), str(dst / fname))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir", default="CSpineSeg")
    parser.add_argument("--profile", default="midrc")
    parser.add_argument("--api", default="https://data.midrc.org")
    parser.add_argument("--numparallel", type=int, default=8)

    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    ensure_gen3_client()

    run(
        f"./gen3-client configure "
        f"--profile={args.profile} "
        f"--cred={args.credentials} "
        f"--apiendpoint={args.api}"
    )

    run(f"./gen3-client auth --profile={args.profile}")

    if args.manifest:
        run(
            f"./gen3-client download-multiple "
            f"--profile={args.profile} "
            f"--manifest={args.manifest} "
            f"--download-path={outdir} "
            f"--numparallel={args.numparallel} "
            f"--skip-completed --no-prompt"
        )
    else:
        sys.exit("manifest.json is required")

    organize_nifti_files(outdir)


if __name__ == "__main__":
    main()
