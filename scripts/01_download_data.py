#!/usr/bin/env python
"""Download training/augmentation datasets into data/raw/<name>.

Sources come from configs/data.yaml (``sources:``) merged with built-in
defaults from the data-source research (HF repos via the ``hf`` CLI,
plain URLs via urllib). Idempotent: a ``.download_complete`` marker in
each destination directory makes reruns skip finished sources.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Built-in defaults; configs/data.yaml `sources:` entries override/extend by name.
DEFAULT_SOURCES: dict[str, dict] = {
    "common_voice_zhtw": {"kind": "hf_dataset", "id": "fsicoli/common_voice_22_0",
                          "include": ["audio/zh-TW/*", "transcript/zh-TW/*"],
                          "license": "cc0-1.0 (public domain; unofficial mirror of Common Voice 22.0)"},
    "yodas_zh": {"kind": "hf_dataset", "id": "espnet/yodas2", "include": ["data/zh000/*"],
                 "license": "cc-by-3.0 (attribution required; mixed zh-CN/zh-TW, filter + s2twp)"},
    "yodas_en": {"kind": "hf_dataset", "id": "espnet/yodas2", "include": ["data/en000/*"],
                 "license": "cc-by-3.0 (attribution required)"},
    "ivod_meta": {"kind": "hf_dataset", "id": "openfun/tw-ly-ivod",
                  "license": "cc-by-4.0 (Legislative Yuan open data; attribution required)"},
    "ivod_fine_tune": {"kind": "hf_dataset", "id": "openfun/ivod-fine-tune",
                       "license": "cc-by-4.0 (attribution required)"},
    "ascend": {"kind": "hf_dataset", "id": "CAiRE/ASCEND",
               "license": "cc-by-sa-4.0 (share-alike: derived DATASET redistribution must be CC-BY-SA)"},
    # AMI = ENGLISH gold meetings (diarization pretraining / pipeline validation only,
    # NOT zh-TW gold). Scope to ihm/ (headset mic, ~24.5 GB) to spare disk; add "sdm/*"
    # for the distant-mic condition (~17 GB more) if far-field robustness is needed.
    "ami": {"kind": "hf_dataset", "id": "edinburghcstr/ami", "include": ["ihm/*"],
            "license": "cc-by-4.0 (attribution required; English meetings)"},
    "musan": {"kind": "url", "id": "https://www.openslr.org/resources/17/musan.tar.gz",
              "license": "CC BY 4.0", "extract": True},
    "rirs_noises": {"kind": "url", "id": "https://www.openslr.org/resources/28/rirs_noises.zip",
                    "license": "Apache 2.0", "extract": True},
}


def load_cfg(path: Path) -> dict:
    if not path.exists():
        print(f"[warn] {path} not found; using built-in source list", file=sys.stderr)
        return {}
    import yaml
    return yaml.safe_load(path.read_text()) or {}


def merged_sources(cfg: dict) -> dict[str, dict]:
    sources = {k: dict(v) for k, v in DEFAULT_SOURCES.items()}
    for name, spec in (cfg.get("sources") or {}).items():
        base = sources.setdefault(name, {})
        base.update({"kind": spec.get("kind", base.get("kind", "hf_dataset")),
                     "id": spec.get("hf_id") or spec.get("url") or base.get("id"),
                     "license": spec.get("license", base.get("license", "UNKNOWN — verify!"))})
        if spec.get("include"):
            base["include"] = spec["include"]
    return sources


def download_hf(repo: str, dest: Path, include: list[str] | None) -> None:
    cmd = ["hf", "download", repo, "--repo-type", "dataset", "--local-dir", str(dest)]
    if include:
        cmd += ["--include", *include]
    subprocess.run(cmd, check=True)


def _has_payload(dest: Path) -> bool:
    """True if dest holds at least one real downloaded file.

    Guards against the hf CLI reporting success while fetching 0 files (e.g. an
    include glob that matches nothing): without this the .download_complete
    marker would be written for an empty dir and a later correct rerun skipped.
    Ignores the marker itself and hf's hidden .cache bookkeeping.
    """
    for p in dest.rglob("*"):
        if p.is_file() and p.name != ".download_complete" and ".cache" not in p.parts:
            return True
    return False


def download_url(url: str, dest: Path, extract: bool) -> None:
    fname = dest / url.rsplit("/", 1)[-1]
    if not fname.exists():
        print(f"    fetching {url}")
        urllib.request.urlretrieve(url, fname)  # noqa: S310 — https URL from vetted list
    if extract:
        import shutil
        print(f"    extracting {fname.name}")
        shutil.unpack_archive(str(fname), str(dest))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/data.yaml"))
    ap.add_argument("--only", default=None,
                    help="comma-separated source names (default: all)")
    ap.add_argument("--raw-dir", default=None, help="override data.yaml paths.raw")
    ap.add_argument("--include", nargs="+", default=None,
                    help="override each source's include glob(s) for a scoped/partial "
                         "fetch (hf_dataset sources only); use to grab a single split.")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    raw_dir = ROOT / (args.raw_dir or (cfg.get("paths") or {}).get("raw", "data/raw"))
    sources = merged_sources(cfg)
    wanted = [s.strip() for s in args.only.split(",")] if args.only else list(sources)
    unknown = [w for w in wanted if w not in sources]
    if unknown:
        ap.error(f"unknown source(s) {unknown}; available: {sorted(sources)}")

    failures: list[str] = []
    for name in wanted:
        spec = sources[name]
        dest = raw_dir / name
        marker = dest / ".download_complete"
        print(f"\n=== {name} ({spec['kind']}: {spec['id']})")
        print(f"    LICENSE: {spec['license']}")
        if not spec.get("id"):
            print("    no downloadable id (placeholder / self-recorded) — skipping")
            continue
        if marker.exists():
            print("    already complete — skipping")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        try:
            if spec["kind"] == "hf_dataset":
                download_hf(spec["id"], dest, args.include or spec.get("include"))
            else:
                download_url(spec["id"], dest, bool(spec.get("extract")))
            if not _has_payload(dest):
                raise RuntimeError(
                    "download reported success but no files landed in "
                    f"{dest} (include glob matched nothing?) — not marking complete")
            marker.write_text(f"{spec['id']}\n")
        except Exception as exc:  # keep going; report at the end
            print(f"    FAILED: {exc}", file=sys.stderr)
            failures.append(name)

    print("\nReminder: keep per-source LICENSE terms with any redistributed derivative.")
    if failures:
        print(f"FAILED sources: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
