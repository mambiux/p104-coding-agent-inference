#!/usr/bin/env bash
# install.sh — Ubuntu 22.04/24.04 + one NVIDIA P104-100 (8 GB) → llama.cpp server with the stock web UI,
# serving Bonsai 2 27B (PTQ1_0 ternary) with the kernels from this repository. Idempotent: rerun to resume.
# Read GUIDE.md first. Everything lands in ~/bonsai except the CUDA toolkit and the optional systemd units.
set -euo pipefail

MODEL_URL="${MODEL_URL:-https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf}"
CTX="${CTX:-32768}"
PORT="${PORT:-8080}"
POWER_W="${POWER_W:-128}"
SPEC="${SPEC:-ngram-mod}"
JOBS="${JOBS:-$(( $(nproc) < 8 ? $(nproc) : 8 ))}"
INSTALL_UNIT="${INSTALL_UNIT:-ask}"
FORK_URL="https://github.com/PrismML-Eng/llama.cpp.git"
FORK_COMMIT="d12ca13bb"
ROOT="$HOME/bonsai"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$REPO_DIR/patches/best8-cumulative-vs-prism-d12ca13bb.patch"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------- preflight ----------
say "Preflight"
[ -f /etc/os-release ] && . /etc/os-release
case "${VERSION_ID:-}" in 22.04|24.04) UBU="ubuntu${VERSION_ID/./}";; *) echo "Ubuntu ${VERSION_ID:-?} is untested; continuing with the 24.04 CUDA repo"; UBU="ubuntu2404";; esac
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: install the NVIDIA driver first (sudo ubuntu-drivers autoinstall; reboot)"
GPU_LINE="$(nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader | grep -i 'P104-100' | head -1 || true)"
[ -n "$GPU_LINE" ] || die "no P104-100 listed by nvidia-smi. Cards present: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',')"
GPU_UUID="$(echo "$GPU_LINE" | cut -d, -f1 | tr -d ' ')"
GPU_MEM="$(echo "$GPU_LINE" | cut -d, -f3 | tr -dc '0-9')"
[ "$GPU_MEM" -ge 8000 ] || die "the P104-100 exposes only ${GPU_MEM} MiB; this needs the 8 GB VBIOS (see GUIDE.md)"
echo "card: $GPU_UUID, ${GPU_MEM} MiB | Ubuntu ${VERSION_ID:-?} | jobs $JOBS | ctx $CTX | port $PORT"
FREE_GB=$(( $(df -Pk "$HOME" | awk 'NR==2{print $4}') / 1024 / 1024 ))
[ "$FREE_GB" -ge 25 ] || die "only ${FREE_GB} GB free in $HOME; need about 25 GB"
[ -f "$PATCH" ] || die "patch not found at $PATCH (run this script from the cloned repository)"
mkdir -p "$ROOT/models" "$ROOT/state"

# ---------- packages ----------
say "Packages"
if [ ! -f "$ROOT/state/apt.done" ]; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential python3 curl ca-certificates gnupg >/dev/null
  touch "$ROOT/state/apt.done"
fi
if [ ! -x "$CUDA_HOME/bin/nvcc" ] && ! command -v nvcc >/dev/null; then
  say "CUDA toolkit 12.6 (nvcc not found)"
  KEYRING="/tmp/cuda-keyring_1.1-1_all.deb"
  curl -fsSL -o "$KEYRING" "https://developer.download.nvidia.com/compute/cuda/repos/${UBU}/x86_64/cuda-keyring_1.1-1_all.deb"
  sudo dpkg -i "$KEYRING" >/dev/null
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-12-6 >/dev/null
fi
[ -x "$CUDA_HOME/bin/nvcc" ] || CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
echo "nvcc: $(nvcc --version | tail -1)"

# ---------- source ----------
say "Source: PrismML llama.cpp at $FORK_COMMIT + this repository's patch"
if [ ! -d "$ROOT/src/.git" ]; then
  git clone -q "$FORK_URL" "$ROOT/src"
fi
cd "$ROOT/src"
if [ ! -f "$ROOT/state/patched.done" ]; then
  git fetch -q origin "$FORK_COMMIT" 2>/dev/null || git fetch -q origin
  git checkout -q "$FORK_COMMIT"
  patch -p1 --dry-run -s < "$PATCH" || die "the patch does not apply cleanly (wrong commit?)"
  patch -p1 -s < "$PATCH"
  touch "$ROOT/state/patched.done"
fi
echo "source at $(git rev-parse --short HEAD), patched"

# ---------- build ----------
say "Build (sm_61 only; 20–40 minutes; retries on compiler crashes)"
if [ ! -f "$ROOT/state/build.done" ]; then
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
    -DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;q8_0-q4_0" \
    -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_COMPRESSION_MODE=size -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" > "$ROOT/cmake.log" 2>&1 \
    || { tail -20 "$ROOT/cmake.log"; die "cmake configure failed (see $ROOT/cmake.log)"; }
  for i in $(seq 1 40); do
    if cmake --build build --config Release -j "$JOBS" --target llama-server llama-bench > "$ROOT/build.log" 2>&1; then break; fi
    echo "build attempt $i failed (compiler crash or error); retrying in 5 s — tail of log:"; grep -E "error" "$ROOT/build.log" | grep -v Werror | tail -3; sleep 5
    [ "$i" -eq 40 ] && die "build keeps failing; see $ROOT/build.log"
  done
  [ -x build/bin/llama-server ] || die "llama-server was not produced; see $ROOT/build.log"
  touch "$ROOT/state/build.done"
fi
BIN="$ROOT/src/build/bin"; echo "binaries: $BIN"

# ---------- model ----------
say "Model"
MODEL="$ROOT/models/$(basename "$MODEL_URL")"
if [ ! -f "$ROOT/state/model.done" ] || [ ! -s "$MODEL" ]; then
  echo "downloading $(basename "$MODEL_URL") (about 5.5 GB, resumable)"
  curl -fL -C - --retry 5 --retry-delay 5 -o "$MODEL" "$MODEL_URL"
  [ "$(stat -c %s "$MODEL")" -ge 4000000000 ] || die "model file is too small; download incomplete?"
  touch "$ROOT/state/model.done"
fi
echo "model: $MODEL ($(( $(stat -c %s "$MODEL") / 1024 / 1024 )) MiB)"

# ---------- launcher ----------
say "Launcher"
SPEC_FLAGS="--spec-type $SPEC"
case "$SPEC" in *draft-mtp*) SPEC_FLAGS="$SPEC_FLAGS --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-p-min 0.75";; esac
cat > "$ROOT/serve.sh" <<EOF
#!/usr/bin/env bash
# Bonsai 2 27B on the P104-100 — generated by install.sh. Edit CTX/PORT here if you like.
export CUDA_VISIBLE_DEVICES=$GPU_UUID
exec env LD_LIBRARY_PATH=$BIN $BIN/llama-server -m "$MODEL" --host 0.0.0.0 --port $PORT -ngl 999 --ctx-size $CTX --parallel 1 \\
  -b 512 -ub 256 -fa on --cache-type-k q4_0 --cache-type-v q4_0 --jinja $SPEC_FLAGS \\
  --reasoning on --reasoning-effort medium --reasoning-budget 4000 \\
  --reasoning-budget-message "Time to stop thinking and write the final answer now." \\
  --temp 0.7 --top-p 0.95 --top-k 20 --cache-reuse 256
EOF
chmod +x "$ROOT/serve.sh"; echo "launcher: $ROOT/serve.sh"

# ---------- power cap ----------
if [ "$POWER_W" != "0" ]; then
  say "Power cap ${POWER_W} W"
  sudo nvidia-smi -i "$GPU_UUID" -pm 1 >/dev/null 2>&1 || true
  sudo nvidia-smi -i "$GPU_UUID" -pl "$POWER_W" >/dev/null 2>&1 && echo "cap applied (not persistent across reboots unless the unit below is installed)" || echo "could not set the power cap (driver refused); continuing at stock power"
fi

# ---------- systemd (optional) ----------
WANT_UNIT="$INSTALL_UNIT"
if [ "$WANT_UNIT" = "ask" ]; then read -r -p "Install a systemd unit so the server (and the power cap) start at boot? [y/N] " a; case "$a" in y|Y) WANT_UNIT=yes;; *) WANT_UNIT=no;; esac; fi
if [ "$WANT_UNIT" = "yes" ]; then
  say "systemd units"
  sudo tee /etc/systemd/system/p104-power-cap.service >/dev/null <<EOF
[Unit]
Description=P104-100 power cap ${POWER_W} W
After=multi-user.target
[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -i $GPU_UUID -pm 1
ExecStart=/usr/bin/nvidia-smi -i $GPU_UUID -pl $POWER_W
[Install]
WantedBy=multi-user.target
EOF
  sudo tee /etc/systemd/system/bonsai.service >/dev/null <<EOF
[Unit]
Description=Bonsai 2 27B llama.cpp server on the P104-100
After=network.target p104-power-cap.service
[Service]
User=$USER
ExecStart=$ROOT/serve.sh
Restart=on-failure
RestartSec=10
[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  [ "$POWER_W" != "0" ] && sudo systemctl enable --now p104-power-cap.service >/dev/null
  sudo systemctl enable --now bonsai.service >/dev/null
  echo "units installed and started: bonsai.service, p104-power-cap.service"
else
  say "Starting the server (foreground process detached; stop with: pkill -f llama-server)"
  nohup "$ROOT/serve.sh" > "$ROOT/server.log" 2>&1 < /dev/null &
fi

# ---------- ready ----------
say "Waiting for the model to load"
for i in $(seq 1 300); do
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then break; fi
  sleep 1
done
curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' || die "the server did not become healthy in 5 minutes; see $ROOT/server.log or: journalctl -u bonsai"
IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
say "Ready"
echo "Web UI:  http://${IP:-localhost}:$PORT/"
echo "API:     http://${IP:-localhost}:$PORT/v1/chat/completions  (OpenAI-compatible)"
echo "Card:    $(nvidia-smi -i "$GPU_UUID" --query-gpu=name,memory.used,power.limit --format=csv,noheader)"
echo "Files:   $ROOT  (serve.sh, src/, models/, logs)"
