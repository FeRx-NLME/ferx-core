args <- commandArgs(TRUE); S <- args[1]; .libPaths(c(file.path(S,"rlib"), .libPaths()))
suppressMessages({library(ferx);library(jsonlite)}); setwd(file.path(S,"cost"))
tm <- function(e,n=3) { v<-numeric(n); ex<-substitute(e); pf<-parent.frame(); for(i in seq_len(n)) v[i]<-system.time(eval(ex,pf))[["elapsed"]]; median(v) }
q <- function(e) suppressWarnings(suppressMessages(capture.output(r<-e))) -> z
rows <- list()
for (N in c(10,1000,10000,50000)) {
  invisible(capture.output(t_fit <- system.time(fit <- suppressWarnings(suppressMessages(ferx_fit("w0.ferx",sprintf("w%d.csv",N),verbose=FALSE))))[["elapsed"]]))
  b <- readBin(sprintf("core%d.fitrx",N),"raw",file.size(sprintf("core%d.fitrx",N)))
  fb <- fit; fb$.bundle <- b
  n <- if (N>=50000) 1 else 3
  p <- tempfile(fileext=".rds"); p2 <- tempfile(fileext=".rds"); pf <- tempfile(fileext=".fitrx")
  t_rds <- tm(saveRDS(fit,p),n); t_rds2 <- tm(saveRDS(fb,p2),n)
  t_rr <- tm(readRDS(p),n); t_rr2 <- tm(readRDS(p2),n)
  t_ser <- tm(serialize(fit,NULL),n); t_ser2 <- tm(serialize(fb,NULL),n)
  t_sv <- tm(suppressWarnings(ferx_save_fit(fit,pf)),n); t_ld <- tm(suppressWarnings(ferx_load_fit(pf)),n)
  sz <- sort(vapply(unclass(fit),function(x) as.numeric(object.size(x)),0),decreasing=TRUE)[1:5]
  rows[[as.character(N)]] <- list(N=N,obs=fit$n_obs,r_fit_s=t_fit,bundle=length(b),obj=as.numeric(object.size(fit)),ser=length(serialize(fit,NULL)),
    rds=file.size(p),rds_b=file.size(p2),t_rds=t_rds,t_rds_b=t_rds2,t_read=t_rr,t_read_b=t_rr2,t_ser=t_ser,t_ser_b=t_ser2,r_save_s=t_sv,r_load_s=t_ld,r_fitrx=file.size(pf),top=as.list(sz))
  str(rows[[as.character(N)]],give.attr=FALSE,max.level=1,vec.len=2); rm(fit,fb); gc(FALSE)
}
write_json(rows,file.path(S,"cost","r_cost.json"),auto_unbox=TRUE,digits=6)
