#!/usr/bin/env bash
# Semantic Motion — one-time setup.
#   * creates a private virtualenv in ~/.semantic-motion/venv (override with SM_HOME)
#   * installs the engine's Python dependencies
#   * downloads the small vision models (~25 MB) and the speech model (large-v3-turbo, ~1.6 GB)
# Re-running is safe (idempotent).  Flags:  --asr-model small   --skip-asr   --python /path/to/python3
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chmod +x "$HERE/sm" "$HERE/setup.sh" 2>/dev/null || true
SKILL="$(cd "$HERE/.." && pwd)"
SM_HOME="${SM_HOME:-$HOME/.semantic-motion}"
ASR_MODEL="large-v3-turbo"
SKIP_ASR=""
PYBIN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --asr-model) ASR_MODEL="$2"; shift 2;;
    --skip-asr) SKIP_ASR="--skip-asr"; shift;;
    --python) PYBIN="$2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done

say() { printf '\033[1;36m[semantic-motion]\033[0m %s\n' "$*"; }

# ---- ffmpeg
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  say "ffmpeg/ffprobe not found."
  case "$(uname -s)" in
    Darwin) say "install with:  brew install ffmpeg";;
    Linux) say "install with your package manager, e.g.  sudo apt-get install -y ffmpeg";;
  esac
  exit 1
fi

# ---- python >= 3.10
if [ -z "$PYBIN" ]; then
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
      PYBIN="$(command -v "$c")"; break
    fi
  done
fi
[ -n "$PYBIN" ] || { say "Python >= 3.10 is required"; exit 1; }
say "using $PYBIN ($("$PYBIN" --version))"

mkdir -p "$SM_HOME"
if [ ! -x "$SM_HOME/venv/bin/python" ]; then
  say "creating virtualenv $SM_HOME/venv"
  "$PYBIN" -m venv "$SM_HOME/venv"
fi
VPY="$SM_HOME/venv/bin/python"
"$VPY" -m pip install --upgrade pip wheel >/dev/null
say "installing Python dependencies (first run takes a few minutes)"
"$VPY" -m pip install -r "$SKILL/requirements.txt"
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  say "Apple Silicon detected: installing mlx-whisper for fast on-device ASR (optional)"
  "$VPY" -m pip install mlx-whisper || say "mlx-whisper unavailable; faster-whisper (CPU) will be used"
fi

# ---- Linux runtime libraries needed by MediaPipe / Skia (EGL + GLES client libs)
if [ "$(uname -s)" = "Linux" ]; then
  if ! ldconfig -p 2>/dev/null | grep -q "libGLESv2.so.2" || ! ldconfig -p 2>/dev/null | grep -q "libEGL.so.1"; then
    say "MediaPipe/Skia need libEGL and libGLESv2. Install e.g.:"
    say "  Debian/Ubuntu: sudo apt-get install -y libegl1 libgles2 libgl1"
    say "  Fedora/Amazon: sudo dnf install -y libglvnd-egl libglvnd-gles mesa-libEGL"
  fi
fi

say "downloading models"
SM_PYTHON="$VPY" "$HERE/sm" setup --asr-model "$ASR_MODEL" $SKIP_ASR
say "running doctor"
SM_PYTHON="$VPY" "$HERE/sm" doctor || { say "doctor reported problems (see above)"; exit 1; }
say "ready.  CLI:  $HERE/sm --help"
