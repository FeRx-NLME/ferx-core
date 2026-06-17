; Provenance: Douglas Eleveld's NONMEM 7.6 reproduction from FeRx-NLME/ferx-r#154
; (the "DATASIM" PAGE blind-analysis dataset, 1-cpt oral, log-scale DV). This
; control file and the sibling datsim_oral.lst (its verbatim NONMEM output;
; #OBJV 70.851 WITHOUT CONSTANT) are the cross-engine reference for
; tests/structural_param_nonmem.rs. The $DATA line names data.ferx.csv as Doug
; ran it; the same data is committed here as data/datsim_oral.csv (identical
; cksum). Kept verbatim — do not edit to "fix" paths.
$PROB Datasim
$INPUT NSIM ID TIME RDV DV MDV AMT EVID
$DATA "data.ferx.csv" IGNORE=@
$SUBROUTINES ADVAN2 TRANS2
$PK
	V  = EXP(THETA(1) + ETA(1));
	KE = EXP(THETA(2) + ETA(2));
	KA = KE + EXP(THETA(3) + ETA(3));
	CL = KE * V;
$ERROR
	CP = A(2)/V;
	Y  = log(CP) + ERR(1);
$THETA
	( 1.61, 3.40, 4.1) ;	/* V */
	( -4.6, -1.2, 0.7) ;	/* KE */
	( -4.6, -0.7, 0.7) ;	/* KA */
$OMEGA BLOCK(3)
	1
	0.001 1
	0.001 0.001 1
$SIGMA
	1
$ESTM SIG=5 MAX=5000 METHOD=1 INTERACT NOABORT POSTHOC PRINT=1
