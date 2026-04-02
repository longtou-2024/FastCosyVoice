#!/usr/bin/env python3
"""
Stage 0: 레퍼런스 오디오 데이터 준비
- egs/pbg/data/*.json 에서 duration 계산
- 8초 이상인 샘플 제외
- filelist.txt 생성
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="egs/pbg/data")
    parser.add_argument("--output", type=str, default="egs/pbg/filelist.txt")
    parser.add_argument("--max_duration", type=float, default=8.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    json_files = sorted(data_dir.glob("*.json"))

    kept = []
    skipped = 0

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            meta = json.load(f)

        duration = meta["end"] - meta["start"]
        if duration >= args.max_duration:
            skipped += 1
            continue

        # stem: e.g. "pbg_0_0"
        kept.append(jf.stem)

    # 파일리스트 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for name in kept:
            f.write(name + "\n")

    print(f"Total: {len(json_files)}, Kept: {len(kept)}, Skipped (>= {args.max_duration}s): {skipped}")
    print(f"Filelist saved to {output_path}")


if __name__ == "__main__":
    main()
