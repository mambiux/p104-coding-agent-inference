# Guide: a 27B coding model on a €30 card, step by step

This gets you from a bare Ubuntu machine with a P104-100 to a running llama.cpp server with the
stock web UI, serving Bonsai 2 27B with the kernels from this repository. One script does the
work; this page tells you what you need and what the script does, so nothing is a surprise.

## What you need

**The card.** An NVIDIA **P104-100 with all 8 GB exposed**. Many P104-100s ship with a mining
VBIOS that exposes only 4 GB; `nvidia-smi` must show 8192 MiB. If it shows 4 GB, the card needs the
8 GB VBIOS from the card vendor before anything here applies (do not cross-flash a VBIOS from a
different vendor or board; a signed VBIOS for your exact board only). Everything in this repository
was measured on 8 GB.

**Cooling and power.** The P104-100 has no display outputs and usually no fan of its own; it is
meant to sit in a case with directed airflow. Give it a case fan blowing across it. It will draw
up to 180 W stock; the script caps it at 128 W, where it keeps ~93 % of its speed. One 8-pin PCIe
power connector.

**The host.** Any x86-64 machine with a free PCIe slot (x4 or better is nice; our host only gave it
Gen1 x2 and it still works, prompt loading is just slower), 16 GB RAM, 40 GB free disk (the model
is 5.5 GB, the build and the CUDA toolkit take the rest), and internet for the downloads.

**The OS.** Ubuntu 22.04 or 24.04, fresh is fine. The NVIDIA driver must be installed and working
before the script runs:

```
sudo ubuntu-drivers autoinstall     # or: sudo apt install nvidia-driver-580
sudo reboot
nvidia-smi                          # must list "P104-100" with 8192MiB
```

Pascal cards are supported by the 580 driver series used here; if a newer series drops Pascal,
pin the driver to 580.

**What the script installs.** `git`, `cmake`, `build-essential`, `python3`, `curl`; the CUDA 12.6
toolkit from NVIDIA's apt repository (only if `nvcc` is missing); the PrismML fork of llama.cpp at
the exact commit this work is based on, patched with `patches/best8-cumulative-vs-prism-d12ca13bb.patch`;
the model file from Hugging Face; a launcher and, if you say yes, a systemd unit so the server
starts at boot. Everything lands under `~/bonsai` except the toolkit and the unit.

## Install

```
git clone https://github.com/mambiux/p104-coding-agent-inference
cd p104-coding-agent-inference
bash install.sh
```

The script asks for `sudo` once for the apt packages, the power cap and the optional unit.
Expect 20–40 minutes: the build is the slow part (the ternary kernels compile for one GPU
architecture only, `sm_61`). On a weak CPU, or one that throws compiler errors under load, run
`JOBS=2 bash install.sh`; the build retries on its own.

When it finishes it prints the address of the web UI, by default `http://<your-ip>:8080/`.
Open it in a browser and talk to the model.

## What you get

With the official model file (`Ternary-Bonsai-2-27B-PTQ1_0.gguf`, 5.5 GB):

| | |
|---|---|
| prompt processing | ~180 tokens per second |
| generation | ~29 tokens per second, more with n-gram drafting on edits |
| context | 32k tokens by default (`CTX=49152` fits too; 64k does not on 8 GB) |
| power | 128 W cap |

With a model file that carries the native MTP draft head (community variants exist on Hugging
Face, for example the `...-PTQ1_0-MTP-...` files), generation on code goes to ~42 tokens per second.
Set `MODEL_URL` to that file and `SPEC=ngram-mod,draft-mtp` when you run the script; read the
model card first, since those variants also change the model's behaviour.

## Options

Set these as environment variables before `bash install.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_URL` | official PTQ1_0 file | any Bonsai 2 27B PTQ1_0 GGUF |
| `CTX` | 32768 | context size; 49152 is the most this card holds with headroom |
| `PORT` | 8080 | web UI and API port |
| `POWER_W` | 128 | power cap in watts; 0 leaves the card at stock |
| `SPEC` | ngram-mod | speculative decoding; use `ngram-mod,draft-mtp` with an MTP model file |
| `JOBS` | number of CPUs, max 8 | parallel compile jobs |
| `INSTALL_UNIT` | ask | `yes` installs and enables the systemd unit without asking |

The server is a standard llama.cpp server: the API is OpenAI-compatible at `/v1/chat/completions`,
so any agent framework that speaks that protocol can use it. For Hermes Agent and other tools that
require 64k context, see the README: on 8 GB that needs a smaller model (the 9B measured there
fits 64k with room to spare).

## If something goes wrong

* `nvidia-smi` does not show the card: driver not installed, or the card is not seated or powered.
* `nvidia-smi` shows 4096MiB: the 4 GB VBIOS, see above.
* The build fails with compiler crashes: run again with `JOBS=2`; the script resumes where it stopped.
* The server starts and then dies under load with "out of memory": lower `CTX` (try 24576).
* Speed is far below the table: check `nvidia-smi` for the power cap (128 W) and that nothing else
  is using the card; check the PCIe link (`nvidia-smi -q | grep -A2 "Link Width"`), a Gen1 x1 slot
  makes prompt loading slow but does not change generation speed.

Everything the script does is plain shell; read it before you run it, it is short.

## Many cards in one box (the fleet)

A mining-style chassis with 8–10 P104-100s (for example a Gigabyte G431-MM0: EPYC Embedded 3151,
four 2.5" SATA bays with no RAID controller, 8–10 PCIe x16 slots at Gen3 x1, three 1600 W PSUs) runs
one server per card. Gen3 x1 per card is enough: inference traffic over the link is tiny, model
loading takes a few seconds longer.

Before the OS: in the BIOS enable **Above 4G Decoding** (ten GPUs need the large BAR space; without
it some cards are missing or the board does not POST), leave SATA in AHCI, and update the BIOS and
BMC firmware while the vendor still hosts them. Give the chassis fans a real curve; the cards are
passive.

Sizing: 10 cards at the 128 W cap draw about 1.3 kW plus the host, which three 1600 W supplies carry
with redundancy. RAM: the model file is mapped once and shared, then about 1 GB per server; 16 GB is
tight for ten cards, 32 GB is comfortable. CPU: a 4-core embedded EPYC feeds ten servers at one
thread each (the work is on the GPUs), but do not build on it; build on another machine and copy
`~/bonsai` over, or let `install.sh` build overnight.

Then:

```
bash install.sh                 # once: toolchain, build, model, one server on card 0 (stop it after)
sudo systemctl stop bonsai      # if you installed the single-card unit
MEM_OFFSET=100 bash fleet/fleet-install.sh
```

`fleet-install.sh` finds every P104-100 with 8 GB, writes one environment file per card, installs
a template unit (`bonsai@N`), caps power and sets the memory offset on every card at boot, and puts
an nginx least-connections router on port 8080 in front of all of them. Agents talk to the router as
one OpenAI-compatible endpoint; ten agents run in parallel, each on its own card at full speed.
`~/bonsai/fleet/status.sh` shows temperatures, power, clocks and health per card; `journalctl -u
bonsai@3` shows card 3's log. Options: `CTX`, `SPEC`, `POWER_W`, `MEM_OFFSET`, `BASE_PORT`,
`ROUTER_PORT`, `THREADS`, `MODEL`.

What a fleet does not do: it does not make one request faster than one card. Splitting a larger
model across cards over x1 links works in llama.cpp (layer split) but runs at roughly one card's
bandwidth for the whole model; measure before planning around it.
