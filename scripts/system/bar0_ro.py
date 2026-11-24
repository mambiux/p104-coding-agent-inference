#!/usr/bin/env python3
# READ-ONLY BAR0 probe of the P104 (0000:04:00.0). mmap PROT_READ only; never writes.
import mmap, os, struct, sys
dev = "/sys/bus/pci/devices/0000:04:00.0/resource0"
fd = os.open(dev, os.O_RDONLY | os.O_SYNC)
m = mmap.mmap(fd, 16 << 20, mmap.MAP_SHARED, mmap.PROT_READ)
def rd(a): return struct.unpack_from("<I", m, a)[0]
regs = {"PMC_BOOT_0":0x0, "PMC_BOOT_42":0xa00, "PMC_ENABLE":0x200, "PEXTDEV_BOOT_0(strap)":0x101000, "PEXTDEV_BOOT_3":0x10100c,
        "PTOP_NUM_GPCS":0x22430, "PTOP_TPC_PER_GPC":0x22434, "PTOP_NUM_FBPS":0x22438, "PTOP_LTC_PER_FBP":0x22450, "PTOP_SLICES_PER_LTC":0x22454,
        "PTOP_NUM_LTCS":0x22458, "FUSE_OPT_GPC":0x21c1c, "FUSE_OPT_FBIO":0x21c14, "FUSE_OPT_FBP":0x21d38,
        "FUSE_OPT_TPC_GPC0":0x21c38, "FUSE_OPT_TPC_GPC1":0x21c3c, "FUSE_OPT_TPC_GPC2":0x21c40, "FUSE_OPT_TPC_GPC3":0x21c44,
        "FUSE_ROP_L2_FBP0":0x21d70, "FUSE_ROP_L2_FBP1":0x21d74, "FUSE_ROP_L2_FBP2":0x21d78, "FUSE_ROP_L2_FBP3":0x21d7c}
for k,a in regs.items():
    v = rd(a); print(f"{k:28s} @0x{a:06x} = 0x{v:08x}  {v:032b}")
print("FUSE block 0x21c00-0x21dfc:")
for a in range(0x21c00, 0x21e00, 0x20):
    print(f"  0x{a:06x}: " + " ".join(f"{rd(a+i):08x}" for i in range(0, 0x20, 4)))
# PROM window (VBIOS) at 0x300000
sig = rd(0x300000) & 0xffff
print(f"PROM 0x300000 first u16 = 0x{sig:04x} (0xaa55 = valid option-ROM signature)")
if sig == 0xaa55:
    data = bytes(m[0x300000:0x300000 + (1 << 20)])
    open(sys.argv[1], "wb").write(data); print("PROM 1MB saved to", sys.argv[1])
else:
    print("PROM not readable this way; first 64 B:", bytes(m[0x300000:0x300040]).hex())
