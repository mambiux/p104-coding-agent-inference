#!/usr/bin/env python3
# READ-ONLY NVML queries on the P104 (clock offsets/ranges, pstates). Calls only nvmlDeviceGet* functions.
import ctypes as C
n = C.CDLL("libnvidia-ml.so.1")
assert n.nvmlInit_v2() == 0
h = C.c_void_p()
r = n.nvmlDeviceGetHandleByPciBusId_v2(b"00000000:04:00.0", C.byref(h)); print("handle", r)
def q(name, *argtypes):
    f = getattr(n, name, None)
    if f is None: print(f"{name}: not in library"); return None
    vals = [t() for t in argtypes]
    rc = f(h, *[C.byref(v) for v in vals])
    msg = n.nvmlErrorString; msg.restype = C.c_char_p
    print(f"{name}: rc={rc} ({msg(rc).decode()}) -> {[v.value for v in vals]}")
    return vals
q("nvmlDeviceGetMemClkVfOffset", C.c_int)
q("nvmlDeviceGetMemClkMinMaxVfOffset", C.c_int, C.c_int)
q("nvmlDeviceGetGpcClkVfOffset", C.c_int)
q("nvmlDeviceGetGpcClkMinMaxVfOffset", C.c_int, C.c_int)
q("nvmlDeviceGetPerformanceState", C.c_int)
q("nvmlDeviceGetPowerManagementLimitConstraints", C.c_uint, C.c_uint)
q("nvmlDeviceGetEnforcedPowerLimit", C.c_uint)
q("nvmlDeviceGetCurrPcieLinkWidth", C.c_uint); q("nvmlDeviceGetMaxPcieLinkWidth", C.c_uint)
q("nvmlDeviceGetCurrPcieLinkGeneration", C.c_uint); q("nvmlDeviceGetMaxPcieLinkGeneration", C.c_uint)
q("nvmlDeviceGetGpuMaxPcieLinkGeneration", C.c_uint)
q("nvmlDeviceGetMemoryBusWidth", C.c_uint); q("nvmlDeviceGetNumGpuCores", C.c_uint)
# min/max clock per pstate (555+): nvmlDeviceGetMinMaxClockOfPState(dev, type, pstate, *min, *max)
f = getattr(n, "nvmlDeviceGetMinMaxClockOfPState", None)
if f:
    for ps in (0, 2, 5, 8):
        for ct, cn in ((0, "GRAPHICS"), (2, "MEM")):
            mn, mx = C.c_uint(), C.c_uint(); rc = f(h, ct, ps, C.byref(mn), C.byref(mx))
            print(f"MinMaxClockOfPState P{ps} {cn}: rc={rc} {mn.value}-{mx.value} MHz")
# nvmlDeviceGetClockOffsets (struct nvmlClockOffset_v1: version, type, pstate, clockOffsetMHz, min, max)
class CO(C.Structure): _fields_ = [("version", C.c_uint), ("type", C.c_int), ("pstate", C.c_int), ("off", C.c_int), ("mn", C.c_int), ("mx", C.c_int)]
f = getattr(n, "nvmlDeviceGetClockOffsets", None)
if f:
    for ps in (0, 2):
        for ct, cn in ((0, "GRAPHICS"), (2, "MEM")):
            s = CO(); s.version = C.sizeof(CO) | (1 << 24); s.type = ct; s.pstate = ps
            rc = f(h, C.byref(s)); print(f"ClockOffsets P{ps} {cn}: rc={rc} off={s.off} range=[{s.mn},{s.mx}]")
n.nvmlShutdown()
