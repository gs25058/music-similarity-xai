"""Small public examples only; save original credits and provenance alongside."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def fetch(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    destination.write_bytes(data)
    return dict(url=url, path=str(destination), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    files = []
    for extension in ("ogg", "txt"):
        url = "https://librosa.org/data/audio/Kevin_MacLeod_-_Vibe_Ace."+extension
        files.append(fetch(url, out/("Vibe_Ace."+extension)))
    revision = "d83e6edba6872a704f5d3b8b32f5cb540088dae6"
    for index in ("001", "002", "003"):
        for name in (index+".mid", "beat_midi.txt", "chord_midi.txt", "key_audio.txt"):
            url = f"https://raw.githubusercontent.com/music-x-lab/POP909-Dataset/{revision}/POP909/{index}/{name}"
            files.append(fetch(url, out/"POP909"/index/name))
    files.append(fetch("https://raw.githubusercontent.com/ethman/slakh-utils/3f62d6a4b0e5952237dd5178ac3513e42161ec0a/splits/redux.json", out/"slakh_redux.json"))
    manifest = dict(files=files, pop909_revision=revision,
                    recording="Kevin MacLeod - Vibe Ace; license and credits in Vibe_Ace.txt",
                    note="Recording excerpts share one source; smoke test, not independent-song evaluation.")
    (out/"sources.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(downloaded_files=len(files))))


if __name__ == "__main__":
    main()
