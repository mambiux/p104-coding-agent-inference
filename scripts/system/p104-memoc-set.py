#!/usr/bin/env python3
# PREPARED, NOT RUN. Sets (or resets) the P104 memory clock VF offset via NVML. Needs root. Volatile: lost on driver
# reload / reboot. Usage: sudo python3 -I memoc_set.py <offset_MHz>   (0 = rollback to stock)
import ctypes as C, sys
off = int(sys.argv[1]); assert -500 <= off <= 1000, "refusing offsets outside [-500, 1000]"
n = C.CDLL("libnvidia-ml.so.1"); assert n.nvmlInit_v2() == 0
h = C.c_void_p()
if len(sys.argv) > 2 and sys.argv[2].startswith("GPU-"):   # fleet use: select the card by UUID
    assert n.nvmlDeviceGetHandleByUUID(sys.argv[2].encode(), C.byref(h)) == 0
else:
    assert n.nvmlDeviceGetHandleByPciBusId_v2(b"00000000:04:00.0", C.byref(h)) == 0
name = C.create_string_buffer(96); n.nvmlDeviceGetName(h, name, 96); assert b"P104" in name.value, name.value
rc = n.nvmlDeviceSetMemClkVfOffset(h, C.c_int(off)); print("SetMemClkVfOffset", off, "rc", rc)
cur = C.c_int(); n.nvmlDeviceGetMemClkVfOffset(h, C.byref(cur)); print("offset now", cur.value)
clk = C.c_uint(); n.nvmlDeviceGetClockInfo(h, 2, C.byref(clk)); print("mem clock now", clk.value, "MHz")
n.nvmlShutdown()
