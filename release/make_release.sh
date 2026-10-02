#!/usr/bin/env bash
# Build the robot-side release from the COMMITTED tree (dev box).
#
#   bash release/make_release.sh --runtime <dir> [--version 1.1.0] [--out <dir>]
#
# <dir> holds llm/ollama/ (Ollama v0.34.4 arm64 + jetpack5, unpacked) and
# llm/models/ (an Ollama model store containing qwen3:1.7b) -- i.e. what
# llm/make_runtime.sh builds, extracted.
#
# Writes to <out> (default ./dist):
#   release_v<V>.tar.gz                 code + policy bundle + Ollama runtime + READMEs
#   qwen3-1.7b_r1-robot-v<V>.tar.gz     the model, extracting into the same directory
#   SHA256SUMS
# Two archives because a GitHub release asset is limited to 2 GiB, and the
# runtime plus the model is ~2.3 GB.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V=1.1.0; RUNTIME=""; OUT="$REPO/dist"; MODEL=qwen3; TAGM=1.7b
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime) RUNTIME="$2"; shift 2 ;;
    --version) V="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done
[[ -x "$RUNTIME/llm/ollama/bin/ollama" ]] || { echo "--runtime: no llm/ollama/bin/ollama under '$RUNTIME'"; exit 2; }
MAN="$RUNTIME/llm/models/manifests/registry.ollama.ai/library/$MODEL/$TAGM"
[[ -f "$MAN" ]] || { echo "--runtime: no $MODEL:$TAGM manifest under $RUNTIME/llm/models"; exit 2; }
cd "$REPO"
git diff --quiet HEAD -- . ':!dist' || { echo "uncommitted changes: commit first, the release is built from HEAD"; exit 2; }
SHA="$(git rev-parse --short=12 HEAD)"
NAME="r1-robot-v$V"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
S="$WORK/main/$NAME"; mkdir -p "$S" "$OUT"
TAROPT=(--sort=name --owner=0 --group=0 --numeric-owner --mtime="@$(git log -1 --format=%ct)")

echo "== code from git HEAD $SHA"
git archive HEAD deploy mission_ctl llm policy_pack LICENSE NOTICE | tar -x -C "$S"
# dev-box-only or superseded files
rm -f "$S/deploy/setup.sh" "$S/deploy/tools/fetch_orin_snapshot.sh" "$S/llm/make_runtime.sh"
rm -rf "$S/deploy/docs"
cp release/README.md release/README_zh.md release/THIRD_PARTY_NOTICES.md "$S/"
install -m 755 release/selftest.sh "$S/selftest.sh"
printf 'r1-robot v%s\ngit %s\nbuilt %s\npolicy 2026-08-19_11-03-32_week04_nohead\nmodel %s:%s\n' \
  "$V" "$(git rev-parse HEAD)" "$(date -Iseconds)" "$MODEL" "$TAGM" > "$S/VERSION"

echo "== policy bundle"
python3 policy_pack/make_bundle.py --out "$S/bundles" >/dev/null
python3 policy_pack/verify_bundle.py "$S"/bundles/*/ | tail -1

echo "== Ollama runtime"
cp -a "$RUNTIME/llm/ollama" "$S/llm/ollama"
cp release/third_party/OLLAMA_LICENSE "$S/llm/ollama/LICENSE"
mkdir -p "$S/llm/models"          # the model archive fills this

( cd "$S" && find . -type f ! -name FILES.sha256 ! -path './llm/models/*' -printf '%P\n' | sort \
    | xargs -d '\n' sha256sum > FILES.sha256 )
echo "   $(wc -l < "$S/FILES.sha256") files"

echo "== model $MODEL:$TAGM"
M="$WORK/model/$NAME/llm/models"
mkdir -p "$M/blobs" "$(dirname "$M/manifests/registry.ollama.ai/library/$MODEL/$TAGM")"
cp "$MAN" "$M/manifests/registry.ollama.ai/library/$MODEL/$TAGM"
python3 - "$MAN" <<'PY' | while read -r d; do cp "$RUNTIME/llm/models/blobs/$d" "$M/blobs/$d"; done
import json, sys
m = json.load(open(sys.argv[1]))
for x in [m["config"]] + m["layers"]:
    print(x["digest"].replace(":", "-"))
PY
( cd "$M" && find . -type f ! -name MODEL.sha256 -printf '%P\n' | sort | xargs -d '\n' sha256sum > MODEL.sha256 )

echo "== archives"
MAIN="release_v$V.tar.gz"; MOD="$MODEL-${TAGM}_r1-robot-v$V.tar.gz"
tar -C "$WORK/main" "${TAROPT[@]}" -cf - "$NAME" | gzip -n -6 > "$OUT/$MAIN"
tar -C "$WORK/model" "${TAROPT[@]}" -cf - "$NAME" | gzip -n -6 > "$OUT/$MOD"
( cd "$OUT" && sha256sum "$MAIN" "$MOD" > SHA256SUMS && ls -l "$MAIN" "$MOD" && cat SHA256SUMS )
LIMIT=$((2*1024*1024*1024))
for f in "$OUT/$MAIN" "$OUT/$MOD"; do
  [[ $(stat -c %s "$f") -lt $LIMIT ]] || { echo "$f is over GitHub's 2 GiB asset limit"; exit 1; }
done
echo "done: $OUT"
