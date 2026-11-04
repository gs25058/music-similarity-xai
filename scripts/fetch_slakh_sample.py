"""Stream one official BabySlakh track; do not download the complete dataset."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/slakh_sample")
    args = parser.parse_args()
    out = Path(args.out)
    url = "https://zenodo.org/records/4603870/files/babyslakh_16k.tar.gz?download=1"
    files, chosen = [], None
    with urllib.request.urlopen(url, timeout=90) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                tracks = [i for i, part in enumerate(parts) if re.fullmatch(r"Track\d+", part)]
                if not tracks:
                    continue
                # The tar lists all track directories first. Choose the first
                # actual file, not a directory entry with no following content.
                if not member.isfile() or not member.name.endswith((".yaml", ".mid", ".wav", ".flac")):
                    continue
                index = tracks[0]
                track = parts[index]
                if chosen is None:
                    chosen = track
                if track != chosen:
                    break
                relative = PurePosixPath(*parts[index:])
                if relative.is_absolute() or ".." in relative.parts or member.size > 200_000_000:
                    raise ValueError("Unexpected archive member")
                data = archive.extractfile(member).read()
                path = out / str(relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                files.append(dict(path=str(path), size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    if not files:
        raise ValueError("No sample track extracted")
    metadata = dict(source=url, doi="10.5281/zenodo.4603870", version="BabySlakh v2, 16 kHz",
                    license="CC-BY-4.0", authors="Ethan Manilow, Gordon Wichern, Prem Seetharaman, Jonathan Le Roux",
                    selected_track=chosen, files=files, partial_archive=True)
    (out/"sample_sources.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(dict(selected_track=chosen, files=len(files), bytes=sum(f["size_bytes"] for f in files))))


if __name__ == "__main__":
    main()
