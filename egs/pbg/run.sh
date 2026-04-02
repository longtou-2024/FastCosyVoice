#!/usr/bin/env bash
# egs/pbg/run.sh — 레퍼런스 오디오 기반 TTS 합성 파이프라인
#
# Usage:
#   bash egs/pbg/run.sh                  # 전체 실행 (stage 0 → 1)
#   bash egs/pbg/run.sh --stage 0        # 데이터 준비만
#   bash egs/pbg/run.sh --stage 1        # 합성만

set -euo pipefail

# ── 프로젝트 루트 (이 스크립트가 egs/pbg/ 에 있다고 가정) ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── 기본값 ──
stage=0
stop_stage=1
data_dir="egs/pbg/data"
filelist="egs/pbg/filelist.txt"
output_dir="egs/pbg/output"
max_duration=8.0
config="config.yaml"

# ── 인자 파싱 ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)       stage=$2;        shift 2 ;;
        --stop_stage)  stop_stage=$2;   shift 2 ;;
        --data_dir)    data_dir=$2;     shift 2 ;;
        --output_dir)  output_dir=$2;   shift 2 ;;
        --max_duration) max_duration=$2; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── venv 활성화 ──
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# ══════════════════════════════════════════════════════════════════════════
# Stage 0: 데이터 준비 (8초 미만 샘플 필터링 → filelist.txt)
# ══════════════════════════════════════════════════════════════════════════
if [ "$stage" -le 0 ] && [ "$stop_stage" -ge 0 ]; then
    echo "═══ Stage 0: 데이터 준비 ═══"
    python egs/pbg/prepare_data.py \
        --data_dir "$data_dir" \
        --output "$filelist" \
        --max_duration "$max_duration"
    echo ""
fi

# ══════════════════════════════════════════════════════════════════════════
# Stage 1: TTS 합성
# ══════════════════════════════════════════════════════════════════════════
if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
    echo "═══ Stage 1: TTS 합성 ═══"
    python egs/pbg/synthesize.py \
        --filelist "$filelist" \
        --data_dir "$data_dir" \
        --output_dir "$output_dir" \
        --config "$config"
    echo ""
fi

echo "Done."
