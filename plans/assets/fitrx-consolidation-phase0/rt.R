args <- commandArgs(TRUE); S <- args[1]
.libPaths(c(file.path(S,"rlib"), .libPaths()))
suppressMessages({library(ferx); library(jsonlite)})
ex <- system.file("examples",package="ferx")
M <- function(n) file.path(ex,"models",paste0(n,".ferx")); D <- function(n) file.path(ex,"data",paste0(n,".csv"))
fx <- list(
 warfarin=list("warfarin","warfarin",list(method="foce",covariance=TRUE)),
 sir=list("warfarin","warfarin",list(method="foce",covariance=TRUE,sir=TRUE,settings=list(sir_samples=300,sir_resamples=100,sir_keep_samples=TRUE))),
 iov=list("warfarin_iov","warfarin_iov",list(method="foce",covariance=FALSE,settings=list(maxiter=2))),
 saem=list("warfarin_saem","warfarin",list(covariance=FALSE,settings=list(maxiter=3))),
 dcm=list("warfarin_dcm","warfarin_dcm",list(covariance=FALSE,settings=list(maxiter=0))),
 nested=list("nested_levels","nested_levels",list(covariance=FALSE,settings=list(maxiter=1))),
 tte=list("pktte_joint","pktte_joint",list(covariance=FALSE,settings=list(maxiter=1))),
 binary=list("binary_logistic","binary_logistic",list(covariance=FALSE,settings=list(maxiter=1))))
sel <- if (length(args)>1) args[-1] else names(fx)
same <- function(a,b) isTRUE(identical(a,b))
res <- list()
for (n in sel) { f <- fx[[n]]; cat("==",n,"\n")
 r <- tryCatch({
  fit <- suppressWarnings(suppressMessages(do.call(ferx_fit,c(list(M(f[[1]]),D(f[[2]])),f[[3]]))))
  p <- file.path(S,"rt",paste0(n,".r.fitrx")); suppressWarnings(ferx_save_fit(fit,p,include_data=TRUE))
  saveRDS(unclass(fit),file.path(S,"rt",paste0(n,".rds")))
  l <- suppressWarnings(ferx_load_fit(p))
  p2 <- file.path(S,"rt",paste0(n,".r2.fitrx")); suppressWarnings(ferx_save_fit(l,p2,include_data=TRUE))
  p3 <- file.path(S,"rt",paste0(n,".core.fitrx"))
  e <- tryCatch({utils::getFromNamespace("ferx_rust_fitrx_engine_resave","ferx")(p,p3);"ok"},error=function(e) conditionMessage(e))
  rd <- function(p,ent="fit.json") { d<-tempfile();dir.create(d);utils::unzip(p,exdir=d);jsonlite::read_json(file.path(d,ent)) }
  fn <- names(unclass(fit)); ln <- names(unclass(l))
  common <- intersect(fn,ln)
  diffs <- common[!vapply(common,function(k) same(fit[[k]],l[[k]]),NA)]
  list(ok=TRUE,live_only=setdiff(fn,ln),load_only=setdiff(ln,fn),changed=diffs,
       nn_live=length(fit[["neural_networks"]]),nn_loaded=length(l[["neural_networks"]]),
       r_keys=names(rd(p)), r2_keys=names(rd(p2)), rx=names(rd(p)$r_extras),
       entries=utils::unzip(p,list=TRUE)$Name, entries_core=if(file.exists(p3)) utils::unzip(p3,list=TRUE)$Name,
       core_keys=if(file.exists(p3)) names(rd(p3)), resave=e, live_names=fn)
 },error=function(e) list(ok=FALSE,err=conditionMessage(e)))
 res[[n]] <- r; if(!isTRUE(r$ok)) cat("  FAIL",r$err,"\n") else cat("  live_only:",r$live_only,"\n  load_only:",r$load_only,"\n  changed:",r$changed,"\n  nn",r$nn_live,r$nn_loaded,"resave",substr(r$resave,1,100),"\n")
}
write_json(res,file.path(S,"rt",paste0("rt-",paste(sel,collapse="_"),".json")),auto_unbox=TRUE,null="null")
