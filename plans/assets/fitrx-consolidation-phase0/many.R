args <- commandArgs(TRUE); S <- args[1]; .libPaths(c(file.path(S,"rlib"), .libPaths()))
suppressMessages({library(ferx);library(jsonlite)}); setwd(file.path(S,"cost"))
ex <- system.file("examples",package="ferx"); m<-file.path(ex,"models/warfarin.ferx"); d<-file.path(ex,"data/warfarin.csv")
sink(tempfile()); K<-50
tt<-numeric(K); fits<-vector("list",K)
for(i in 1:K) tt[i]<-system.time(fits[[i]]<-suppressWarnings(suppressMessages(ferx_fit(m,d,method="foce",covariance=FALSE,verbose=FALSE))))[["elapsed"]]
p<-tempfile(fileext=".fitrx"); ts<-system.time(for(i in 1:20) suppressWarnings(ferx_save_fit(fits[[1]],p)))[["elapsed"]]/20
b<-readBin("../corefix/warfarin.core.fitrx","raw",file.size("../corefix/warfarin.core.fitrx"))
fb<-lapply(fits,function(f){f$.bundle<-b;f})
r1<-tempfile();r2<-tempfile();saveRDS(fits,r1);saveRDS(fb,r2)
# real fit on the 1000-subject data, as the denominator for a large-ish save
sed<-readLines("w0.ferx"); sed<-sub("maxiter = 0","maxiter = 300",sed); writeLines(sed,"w300.ferx")
t1000<-system.time(f1<-suppressWarnings(suppressMessages(ferx_fit("w300.ferx","w1000.csv",verbose=FALSE))))[["elapsed"]]
# SIR pool
fs<-suppressWarnings(suppressMessages(ferx_fit(m,d,method="foce",covariance=TRUE,sir=TRUE,verbose=FALSE,settings=list(sir_samples=10000,sir_resamples=5000,sir_keep_samples=TRUE))))
ps<-tempfile(fileext=".fitrx");tsv<-system.time(suppressWarnings(ferx_save_fit(fs,ps)))[["elapsed"]]
fn<-suppressWarnings(suppressMessages(ferx_fit(m,d,method="foce",covariance=TRUE,verbose=FALSE)));pn<-tempfile(fileext=".fitrx");suppressWarnings(ferx_save_fit(fn,pn))
sink()
out<-list(K=K,fit_med=median(tt),fit_min=min(tt),r_save=ts,obj_one=as.numeric(object.size(fits[[1]])),list_obj=as.numeric(object.size(fits)),list_obj_b=as.numeric(object.size(fb)),rds=file.size(r1),rds_b=file.size(r2),bundle=length(b),t1000=t1000,n_iter1000=f1$n_iterations,conv1000=f1$converged,sir_n=fs$sir_resamples_n,sir_dim=fs$sir_resamples_dim,sir_fitrx=file.size(ps),nosir_fitrx=file.size(pn),sir_save=tsv,sir_obj=as.numeric(object.size(fs)))
write_json(out,"many.json",auto_unbox=TRUE,digits=6); print(unlist(out))
