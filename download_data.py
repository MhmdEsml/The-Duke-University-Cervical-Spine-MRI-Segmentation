import os
import shutil
import argparse
import subprocess
import sys


def run(cmd):
    print(f"> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(f"Command failed: {cmd}")


def download_midrc_data(
    credentials_path: str,
    manifest_path: str | None,
    output_dir: str,
    num_parallel: int,
):
    """Download MIDRC data using gen3-client"""
    if not os.path.isfile(credentials_path):
        sys.exit(f"credentials.json not found: {credentials_path}")

    if manifest_path and not os.path.isfile(manifest_path):
        sys.exit(f"manifest.json not found: {manifest_path}")

    os.makedirs(output_dir, exist_ok=True)

    shutil.copy(credentials_path, "credentials.json")
    if manifest_path:
        shutil.copy(manifest_path, "manifest.json")

    if not os.path.exists("gen3-client"):
        print("Downloading gen3-client...")
        run(
            "curl -s https://api.github.com/repos/uc-cdis/cdis-data-client/releases/latest "
            "| grep browser_download_url.*linux "
            "| cut -d '\"' -f 4 "
            "| wget -qi -"
        )

        run("unzip -o dataclient_linux.zip")
        run("chmod +x gen3-client")

    print("Configuring gen3-client...")
    run(
        "./gen3-client configure "
        "--profile=midrc "
        "--cred=credentials.json "
        "--apiendpoint=https://data.midrc.org"
    )

    run("./gen3-client auth --profile=midrc")

    if manifest_path:
        print("Downloading data using manifest...")
        run(
            "./gen3-client download-multiple "
            "--profile=midrc "
            "--manifest=manifest.json "
            f"--download-path={output_dir} "
            f"--numparallel={num_parallel} "
            "--no-prompt "
            "--skip-completed"
        )
    else:
        print("No manifest provided.")
        print(
            "Use:\n"
            "./gen3-client download-single "
            "--profile=midrc "
            "--guid=<GUID> "
            f"--download-path={output_dir} "
            "--no-prompt"
        )

    print(f"Files saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MIDRC data using gen3-client")

    parser.add_argument(
        "--credentials",
        required=True,
        help="Path to credentials.json",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to manifest.json (optional)",
    )
    parser.add_argument(
        "--output-dir",
        default="CSpineSeg",
        help="Directory to save downloaded data",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=8,
        help="Number of parallel downloads",
    )

    args = parser.parse_args()

    download_midrc_data(
        credentials_path=args.credentials,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        num_parallel=args.num_parallel,
    )
