"""Render the field matrix. Sets come from extract.py; R_WRITE_EXTRA/R_NOT_WRITTEN and the
list-name map are hand-checked against ferx_save_fit.R / ferx_load_fit.R."""
import json,sys
S=sys.argv[1]; k=json.load(open(S+'/an/keys.json'))
L=set(k['Rlist']); top=k['FitWire']
rw=set(k['Rwrite_top'])|{'reader_settings','population_fingerprint'}
rr=set(k['Rread_w'])
LIST={'ofv_data':'ofv_data','covariance_matrix':'cov_matrix (+`cov_matrix_dim`)','optimizer':'optimizer_label','bloq_method':'bloq_method_label','sir_seed':'sir_seed_used',
 'theta':'see nested','omega':'see nested','sigma':'see nested','sir':'see nested','iov':'see nested','vi':'vi','eta_param_info':'eta_param_types, eta_linked_theta',
 'error_model':None,'environment':None,'method_wall_times_secs':None,'total_ebe_nm_polished':None,'saem_mh_accept_tail':None,
 'npde_seed':None,'data_bindings':'theta_levels, covariate_stats','omega_init':'omega_init (+`omega_init_dim`)'}
NOTE={'ofv_data':'core writes only for a priored fit','ofv_prior':'core writes only for a priored fit','prior_summary':'core writes only for a priored fit',
 'error_model':'R derives it from `sigma_types`; R never reads it','model_text':'R writes the same text as `model.ferx` only; core overwrites from `model.ferx` on load',
 'neural_networks':'**R drops it on every save**, live fit included','warning_details':'R writes only when a warning has details',

 'saem_mu_ref_m_step_evals_saved':'','sir_seed':'R writes only beside `sir.settings`','bloq_method':'R omits when empty',
 'data_bindings':'R rebuilds it from `theta_levels`','reader_settings':'R carries it as a JSON string','population_fingerprint':'R carries it as a JSON string; O(N) in size',
 'eta_param_info':'R writes `individual_param_name` as `""`','trace_path':'dangling temp path after the run','environment':'includes `username`'}
def m(b): return 'yes' if b else '**no**'
rows=['| `fit.json` key | Core W | Core R | R W | R R | R list key | Note |','|---|---|---|---|---|---|---|']
for f in top:
    lk=LIST.get(f,f if f in L else None) if f in LIST else (f if f in L else None)
    W=f in rw; R=f in rr
    rows.append(f"| `{f}` | yes | yes | {m(W)} | {m(R)} | {('`'+lk+'`') if lk and lk!='see nested' else (lk or '**none**')} | {NOTE.get(f,'')} |")
open(S+'/an/matrix_top.md','w').write('\n'.join(rows)+'\n')
nw=[f for f in top if f not in rw]; nr=[f for f in top if f not in rr]
print('not R-written',nw); print('not R-read',nr); print('no list',[f for f in top if not (LIST.get(f) if f in LIST else f in L)])
print('in list, not wire top:',sorted(L-set(top)))
