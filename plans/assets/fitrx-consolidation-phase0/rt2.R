args <- commandArgs(TRUE); S <- args[1]
.libPaths(c(file.path(S,"rlib"), .libPaths())); suppressMessages({library(ferx);library(jsonlite)})
j <- read_json(list.files(file.path(S,"rt"),"^rt-.*json$",full.names=TRUE)[1])
nz <- function(x) !(is.null(x) || length(x)==0L)
agg <- list()
for (n in names(j)) { r <- j[[n]]; cat("\n##",n,if(!isTRUE(r$ok)) paste("FAIL:",r$err),"\n"); if(!isTRUE(r$ok)) next
 fit <- readRDS(file.path(S,"rt",paste0(n,".rds")))
 p <- file.path(S,"rt",paste0(n,".r.fitrx")); l <- unclass(suppressWarnings(ferx_load_fit(p)))
 livek <- names(fit)[vapply(fit,nz,NA)]; lk <- names(l)[vapply(l,nz,NA)]
 lost <- setdiff(livek,lk); gained <- setdiff(lk,livek)
 cm <- intersect(livek,lk); ch <- cm[!vapply(cm,function(k) isTRUE(all.equal(fit[[k]],l[[k]],check.attributes=FALSE)),NA)]
 chid <- setdiff(cm[!vapply(cm,function(k) identical(fit[[k]],l[[k]]),NA)],ch)
 cat("lost(non-empty live, empty after R load):",lost,"\ngained:",gained,"\nvalue changed:",ch,"\ntype/attr-only changed:",chid,"\n")
 rk <- unlist(r$r_keys); ck <- unlist(r$core_keys); r2 <- unlist(r$r2_keys)
 cat("R fit.json keys not in core re-save:",setdiff(rk,ck),"\ncore re-save keys not in R fit.json:",setdiff(ck,rk),"\nr_extras:",unlist(r$rx),"\nkeys lost on R re-save:",setdiff(rk,r2),"\nentries R:",unlist(r$entries),"\nentries core:",unlist(r$entries_core),"\nresave:",substr(r$resave,1,160),"\n")
 for (k in lost) agg[[k]] <- c(agg[[k]],n)
 for (k in ch) agg[[paste0("~",k)]] <- c(agg[[paste0("~",k)]],n)
}
write_json(agg,file.path(S,"rt","agg.json"),auto_unbox=FALSE)
if ("dcm" %in% names(j)) { f<-readRDS(file.path(S,"rt","dcm.rds")); cat("\nDCM live neural_networks:\n"); str(f$neural_networks,max.level=2,list.len=8)
 d<-tempfile();dir.create(d);utils::unzip(file.path(S,"rt","dcm.r.fitrx"),exdir=d);w<-read_json(file.path(d,"fit.json")); cat("in R fit.json:", "neural_networks"%in%names(w),"\n"); print(file.size(file.path(S,"rt","dcm.r.fitrx")))
 cat(grep("neural_network|TYPICAL_PK",readLines(file.path(d,"model.ferx")),value=TRUE)[1:3],sep="\n")}
