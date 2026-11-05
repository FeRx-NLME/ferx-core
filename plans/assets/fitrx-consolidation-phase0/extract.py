import re,json,sys
S=sys.argv[1]
CORE=sys.argv[2]
core=open(CORE+'/src/io/fitrx.rs').read()
lib=open(S+'/repos/ferx-r/src/rust/src/lib.rs').read().split('\n')
sv=open(S+'/repos/ferx-r/R/ferx_save_fit.R').read()
ld=open(S+'/repos/ferx-r/R/ferx_load_fit.R').read()
def fields(name):
    m=re.search(r'struct '+name+r' \{(.*?)\n\}',core,re.S)
    return re.findall(r'^\s{4}(?:pub )?([a-z_0-9]+): ',m.group(1),re.M)
out={}
for s in 'FitWire ViWire ViPosteriorWire ThetaWire OmegaWire SigmaWire SirWire SirSettingsWire ScoringSettingsWire IovWire LevelWire EtaParamInfoWire MatrixWire DataBindingsWire'.split():
    out[s]=fields(s)
# R list keys: last list!( in fit_result_to_list
start=[i for i,line in enumerate(lib) if line.startswith('fn fit_result_to_list')][0]
okl=[i for i in range(start,len(lib)) if lib[i].startswith('    Ok(list!(')][0]
end=[i for i in range(okl,len(lib)) if lib[i]=='    ))'][0]
keys=[]
for line in lib[okl:end]:
    m=re.match(r'^ {8}([a-zA-Z_0-9]+)\s*(?:=|,$)',line)
    if m:
        keys.append(m.group(1))
out['Rlist']=keys
# R written: body of .fitrx_write_fit_json
b=sv[sv.index('.fitrx_write_fit_json <- function'):sv.index('.fitrx_write_ebes_csv <- function')]
top=set(re.findall(r'^    ([a-z_0-9]+)\s*=\s',b,re.M))|set(re.findall(r'wire\$([a-z_0-9]+)\s*<-',b))
out['Rwrite_top']=sorted(top)
out['Rread_w']=sorted(set(re.findall(r'\bw\$([a-z_0-9]+)',ld))|set(re.findall(r'\bw\[\["([a-z_0-9]+)"\]\]',ld))|set(re.findall(r'\bwire\$([a-z_0-9]+)',ld)))
json.dump(out,open(S+'/an/keys.json','w'),indent=0)
ft=set(out['FitWire'])
print('nFitWire',len(ft),'nRlist',len(keys))
print('core top not R-written:',sorted(ft-top))
print('R-written not core:',sorted(top-ft))
print('core top not R-read:',sorted(ft-set(out['Rread_w'])))
print('R-read not core:',sorted(set(out['Rread_w'])-ft))
print('R-written not R-read:',sorted(top-set(out['Rread_w'])))
print('R-read not R-written:',sorted(set(out['Rread_w'])-top))
print('Rlist:',keys)
