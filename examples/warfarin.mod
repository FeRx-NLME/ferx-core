// One-compartment oral PK model of warfarin, written in mrgsolve syntax.
// Mirror of `examples/warfarin.ferx` — use `ferx translate examples/warfarin.mod`
// to convert it to ferx's native DSL.

$PROB warfarin one-cpt oral

$PARAM TVCL = 0.2, TVV = 10.0, TVKA = 1.5

$CMT GUT CENT

$OMEGA @labels ETA_CL ETA_V ETA_KA
0.09 0.04 0.30

$SIGMA @labels PROP_ERR
0.02

$MAIN
double CL = TVCL * exp(ETA_CL);
double V  = TVV  * exp(ETA_V);
double KA = TVKA * exp(ETA_KA);

$PKMODEL ncmt=1, depot=TRUE

$TABLE
capture IPRED = CENT/V;
capture Y     = IPRED * (1 + EPS(1));
