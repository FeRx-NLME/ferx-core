"""Exact -2LL (no 2*pi constant) for the naproxen MBMA model, linear in eta, with a
within-arm residual correlation C(rho) on the weighted residual: R_arm = W C W.
Because F is linear in (ETA1, ETA2) and R does not depend on eta, FOCE == exact
marginal: y ~ N(f0, G Om G' + R).  Pure python (no numpy)."""
import csv, math, sys, collections

def chol(A):
    n = len(A); L = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s = A[i][j] - sum(L[i][k]*L[j][k] for k in range(j))
            if i == j:
                if s <= 0: raise ValueError('not PD')
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L

def corr(struct, rho, times):
    n = len(times)
    C = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if struct == 'cs': C[i][j] = rho
            elif struct == 'ar1': C[i][j] = rho ** abs(i-j)
            elif struct == 'car': C[i][j] = rho ** abs(times[i]-times[j])
            elif struct == 'none': C[i][j] = 0.0
    return C

def m2ll(struct, rho, th, om_var, path='napx1081.csv'):
    E0, EMAX, ET50, BE0F, BEMF, BEMT, BLET = th
    rows = list(csv.DictReader(open(path)))
    by_id = collections.OrderedDict()
    for r in rows: by_id.setdefault(r['ID'], []).append(r)
    tot = 0.0
    for sid, rs in by_id.items():
        n = len(rs); res = []; G = []; W = []
        for r in rs:
            t = float(r['TIME']); trt = float(r['TRT']); fl = float(r['FLARE'])
            e0 = E0 + BE0F*fl; emax = EMAX + BEMF*fl + BEMT*trt; et50 = ET50*math.exp(BLET*trt)
            g2 = t/(t+et50)
            f = e0 - emax*g2
            res.append(float(r['DV']) - f); G.append((1.0, -g2)); W.append(float(r['WPSE']))
        V = [[G[i][0]*G[j][0]*om_var[0] + G[i][1]*G[j][1]*om_var[1] for j in range(n)] for i in range(n)]
        # arm blocks
        blocks = collections.OrderedDict()
        for i, r in enumerate(rs): blocks.setdefault(r['L2'], []).append(i)
        for idx in blocks.values():
            C = corr(struct, rho, [float(rs[i]['TIME']) for i in idx])
            for a, i in enumerate(idx):
                for b, j in enumerate(idx):
                    V[i][j] += W[i]*W[j]*C[a][b]
        L = chol(V)
        z = []
        for i in range(n):
            z.append((res[i] - sum(L[i][k]*z[k] for k in range(i)))/L[i][i])
        tot += 2*sum(math.log(L[i][i]) for i in range(n)) + sum(v*v for v in z)
    return tot

TH = [5.1972, 1.1580, 0.6901, 0.9629, 0.8197, 0.7930, -1.1747]
OM = [0.6215**2, 0.7387**2]
if __name__ == '__main__':
    for struct, rho in [('none',0.0),('cs',0.0),('cs',0.5),('cs',-0.2),('ar1',0.5),('ar1',-0.4),('car',0.8),('car',0.3)]:
        print(f'{struct:5s} rho={rho:+.2f}  -2LL = {m2ll(struct, rho, TH, OM):.9f}')
