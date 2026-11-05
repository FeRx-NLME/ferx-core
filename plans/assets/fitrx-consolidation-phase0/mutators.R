args <- commandArgs(TRUE); rdir <- args[1]; out <- args[2]
options(keep.source=TRUE)
rows <- list()
add <- function(...) rows[[length(rows)+1L]] <<- data.frame(..., stringsAsFactors=FALSE)
lhs_info <- function(e) {
  # returns root var, path string, wrapper funs
  path <- character(); wrap <- character()
  while (is.call(e)) {
    f <- if (is.symbol(e[[1]])) as.character(e[[1]]) else paste(deparse(e[[1]]),collapse="")
    if (f %in% c("$","@")) { path <- c(as.character(e[[3]]), path); e <- e[[2]] }
    else if (f %in% c("[[","[")) {
      lab <- if (length(e)<3 || (is.symbol(e[[3]]) && !nzchar(as.character(e[[3]])))) "<all>" else if (is.character(e[[3]])) e[[3]] else paste0("<",paste(deparse(e[[3]]),collapse=""),">")
      path <- c(lab, path); e <- e[[2]]
    } else { wrap <- c(f, wrap); if (length(e)<2) return(NULL); e <- e[[2]] }
  }
  if (!is.symbol(e)) return(NULL)
  list(root=as.character(e), path=path, wrap=wrap)
}
walk <- function(e, file, fn, line) {
  if (is.call(e)) {
    h <- e[[1]]
    hs <- if (is.symbol(h)) as.character(h) else ""
    if (hs=="{") {
      sr <- attr(e,"srcref")
      for (i in seq_along(e)[-1]) { l <- if (!is.null(sr) && length(sr)>=i) sr[[i]][1] else line; walk(e[[i]],file,fn,l) }
      return(invisible())
    }
    if (hs %in% c("<-","<<-","=")) {
      l <- e[[2]]
      if (is.call(l)) {
        info <- lhs_info(l)
        if (!is.null(info)) add(file=file,fn=fn,line=line,root=info$root,path=paste(info$path,collapse="$"),
          wrap=paste(info$wrap,collapse=","),op=hs,rhs=substr(paste(deparse(e[[3]],width.cutoff=120)[1],collapse=""),1,70))
      }
      if (is.symbol(l) && is.call(e[[3]])) {
        r <- e[[3]]; rf <- if (is.symbol(r[[1]])) as.character(r[[1]]) else paste(deparse(r[[1]]),collapse="")
        if (grepl("modifyList|^within$|^transform$|^structure$|^unclass$|^c$|^append$|^replace$|^setdiff$",rf) && length(r)>=2 && is.symbol(r[[2]]))
          add(file=file,fn=fn,line=line,root=as.character(l),path=paste0("<",rf,"(",as.character(r[[2]]),")>"),wrap="",op="call",rhs=substr(paste(deparse(r,width.cutoff=120)[1],collapse=""),1,70))
      }
      if (is.symbol(l) && is.call(e[[3]]) && identical(e[[3]][[1]], as.name("function"))) { walk(e[[3]],file,paste0(fn,"/",as.character(l)),line); return(invisible()) }
    }
    if (hs=="function") { walk(e[[3]],file,fn,line); return(invisible()) }
    for (i in seq_along(e)) { if (is.symbol(e[[i]])) next; if (is.call(e[[i]])||is.pairlist(e[[i]])) walk(e[[i]],file,fn,line) }
  } else if (is.pairlist(e)) for (i in seq_along(e)) { if (is.symbol(e[[i]])) next; if (is.call(e[[i]])) walk(e[[i]],file,fn,line) }
  invisible()
}
for (f in list.files(rdir,"\\.R$",full.names=TRUE)) {
  ex <- parse(f,keep.source=TRUE); sr <- attr(ex,"srcref")
  for (i in seq_along(ex)) {
    e <- ex[[i]]; fn <- "<top>"
    if (is.call(e) && as.character(e[[1]])[1] %in% c("<-","=") && is.symbol(e[[2]])) fn <- as.character(e[[2]])
    walk(e,basename(f),fn,sr[[i]][1])
  }
}
d <- do.call(rbind,rows); write.csv(d,out,row.names=FALSE); cat(nrow(d),"assignments into a compound target\n")
