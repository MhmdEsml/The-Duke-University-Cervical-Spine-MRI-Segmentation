import os
import shutil
import argparse
import subprocess
import sys
import platform


def run(cmd):
    print(f"> {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def download_midrc_data(
    credentials_path: str,
    manifest_path: str | None,
    output_dir: str,
    num_parallel: int,
):
    if not os.path.isfile(credentials_path):
        sys.exit(f"credentials.json not found: {credentials_path}")

    if manifest_path and not os.path.isfile(manifest_path):
        sys.exit(f"manifest.json not found: {manifest_path}")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists("credentials.json"):
        shutil.copy(credentials_path, "credentials.json")

    if manifest_path and not os.path.exists("manifest.json"):
        shutil.copy(manifest_path, "manifest.json")

    if platform.system() != "Linux":
        sys.exit("gen3-client auto-download only supported on Linux")

    if not (os.path.isfile("gen3-client") and os.access("gen3-client", os.X_OK)):
        run(
            "curl -s https://api.github.com/repos/uc-cdis/cdis-data-client/releases/latest "
            "| grep browser_download_url.*linux "
            "| cut -d '\"' -f 4 "
            "| wget -qi -"
        )
        run("unzip -o dataclient_linux.zip")
        run("chmod +x gen3-client")

    run(
        "./gen3-client configure "
        "--profile=midrc "
        "--cred=credentials.json "
        "--apiendpoint=https://data.midrc.org"
    )

    run("./gen3-client auth --profile=midrc")

    if manifest_path:
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
        sys.exit(
            "No manifest provided. Use download-single manually:\n"
            "./gen3-client download-single --profile=midrc --guid=<GUID> "
            f"--download-path={output_dir} --no-prompt"
        )

    img_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for root, _, files in os.walk(output_dir):
        for fname in files:
            if not fname.endswith(".nii.gz"):
                continue

            src = os.path.join(root, fname)

            if os.path.commonpath([src, img_dir]) == img_dir or \
               os.path.commonpath([src, mask_dir]) == mask_dir:
                continue

            dst_dir = mask_dir if "_SEG" in fname else img_dir
            dst = os.path.join(dst_dir, fname)

            if os.path.exists(dst):
                sys.exit(f"File already exists: {dst}")

            shutil.move(src, dst)

    n_images = len([f for f in os.listdir(img_dir) if f.endswith(".nii.gz")])
    n_masks = len([f for f in os.listdir(mask_dir) if f.endswith(".nii.gz")])

    if n_images == 0:
        sys.exit("No images downloaded — check manifest or credentials")

    print(f"Images: {n_images}")
    print(f"Masks: {n_masks}")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--credentials", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default="CSpineSeg")
    parser.add_argument("--num-parallel", type=int, default=8)

    args = parser.parse_args()

    download_midrc_data(
        credentials_path=args.credentials,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        num_parallel=args.num_parallel,
    )
