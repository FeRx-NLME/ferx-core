import subprocess, sys, re
for b in sys.argv[1:]:
    out = subprocess.run(['size', b], capture_output=True, text=True).stdout.splitlines()[-1].split()
    print(f"{b}: text {int(out[0])/2**20:.1f} MB, total {int(out[3])/2**20:.1f} MB")
    nm = subprocess.run(['nm', '-S', '--size-sort', '-C', b], capture_output=True, text=True).stdout
    cats = [('DualV/vtape', r'DualV|vtape'), ('DualF/vfree', r'DualF|vfree'), ('DualH', r'DualH'),
            ('DualT/rtape', r'DualT|rtape'), ('DualMixed', r'DualMixed'), ('Dual2', r'Dual2\b|dual2::'),
            ('Dual1', r'Dual1\b|dual1::'), ('Taylor', r'Taylor')]
    acc = {k: 0 for k, _ in cats}
    acc['rest'] = 0
    for line in nm.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) < 4:
            continue
        try:
            sz = int(parts[1], 16)
        except ValueError:
            continue
        for k, pat in cats:
            if re.search(pat, parts[3]):
                acc[k] += sz
                break
        else:
            acc['rest'] += sz
    for k, v in sorted(acc.items(), key=lambda x: -x[1]):
        print(f"   {k:12s} {v/2**20:6.1f} MB")
