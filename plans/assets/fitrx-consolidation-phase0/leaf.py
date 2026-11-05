import json,zipfile,sys
def leaves(o,p=''):
    s=set()
    if isinstance(o,dict):
        for k,v in o.items(): s|=leaves(v,(p+'.'+k) if p else k)
        if not o: s.add(p)
    elif isinstance(o,list):
        if o and isinstance(o[0],dict):
            for x in o: s|=leaves(x,p+'[]')
        else: s.add(p)
    else: s.add(p)
    return s
def rd(f): return json.loads(zipfile.ZipFile(f).read('fit.json'))
S=sys.argv[1]
for name,r,c in [('warfarin','rt/warfarin.r.fitrx','corefix/warfarin.core.fitrx'),('dcm','rt/dcm.r.fitrx','corefix/warfarin_dcm.core.fitrx'),('iov','rt/iov.r.fitrx','corefix/warfarin_iov.core.fitrx')]:
    a,b=rd(S+'/'+r),rd(S+'/'+c)
    la,lb=leaves(a),leaves(b)
    print('##',name)
    print(' core-only leaves:',sorted(x for x in lb-la if not x.startswith('neural_networks[]') ))
    print(' R-only leaves   :',sorted(x for x in la-lb if not x.startswith('r_extras') and not x.startswith('exclusions.')))
    print(' nn leaves:',sorted(x for x in lb if x.startswith('neural_networks'))[:20])
    print(' environment:',b.get('environment'))
    z=zipfile.ZipFile(S+'/'+c)
    print(' core entries',[(i.filename,i.file_size,i.compress_size) for i in z.infolist()])
