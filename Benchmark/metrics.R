#' C-index calculation
#' 
#' Calculating Concordance Indices for the evaluation results of Cancer Prognosis methods
#' 
#' @import genefu
#' @import survcomp
#' @param data A dataframe object with rows for samples and columns represent corresponding methods. A return value from CancerPrognosis_xxx() function
#' @param survival A dataframe object which contains variables (columns) representing for survival time and event. The rows are the samples.
#' @param outputFolder (Optional) A desired folder to put the results or "output" by default
#' @param PValue if ture, return the object of function concordance.index().
#' 
#' @return
#' This function is used for its side-effect.
#' A plot for overall Concordance Index. For C-Index for each method, please refer in the outputFolder
#' 
#' 
#' @examples 
#' library("breastCancerMAINZ")
#' data("mainz")
#' methods <- c("AURKA", "ESR1", "ERBB2", "GGI", "GENIUS", "Endopredict", "OncotypeDx", 
#'             "TAMR13", "PIK3CAGS", "GENE70", "rorS", "RNAmodel", "Ensemble")
#' sampleInfo= pData(mainz)           
#' survival=data.frame(time=sampleInfo$t.dmfs,event=sampleInfo$e.dmfs, row.names=sampleInfo$samplename)
#' \dontrun{
#' res = CancerPrognosis_RNAData(data=mainz, platform="custom", methods=methods)
#' CIs = Cindex(data=res, survival,outputFolder="./mainz")
#' }
#' @export
#' 
Cindex <- function(data, survival, PValue = FALSE, outputFolder=NULL) {
  
  methods = toupper(colnames(data)[1:ncol(data)])
  
  samples = intersect(rownames(data), rownames(survival))
  data = data[samples, ]
  survival = survival[samples, ]
  
  cat("Calculating on",nrow(data), "samples \n")
  
  setT = survival$time
  setE = survival$event
    
  cindex <- apply(X = as.matrix(data), MARGIN = 2, 
                  FUN = function(x, y, z) { 
                    tt = survcomp::concordance.index(x = x, surv.time = y, surv.event = z, method = "noether", na.rm = TRUE)
                    if (PValue) return (tt)
                    else return(c("cindex" = tt$c.index, "cindex.se" = tt$se, "lower" = tt$lower, "upper" = tt$upper))},
                    y = setT, z = setE )
  if(! is.null(outputFolder)) {
    dir.create(file.path(outputFolder), showWarnings = FALSE)
    write.csv(cindex,file = paste(outputFolder, "Cindex.csv", sep = "/"),row.names = TRUE)
  }
  else {
    outputFolder = "./output"
    dir.create(file.path(outputFolder), showWarnings = FALSE)
    write.csv(cindex,file = paste(outputFolder, "Cindex.csv", sep = "/"),row.names = TRUE)
  }
  return(cindex)
}

#data: resMatrix
IBSC <- function(dd.tr, dd.ts, method = "cox") {
  
  methods = toupper(colnames(dd.ts$resMatrix)[1:ncol(dd.ts$resMatrix)])
  
  samples = intersect(rownames(dd.ts$resMatrix), rownames(dd.ts$survival))
  dd.ts$resMatrix = dd.ts$resMatrix[samples, ]
  dd.ts$survival = dd.ts$survival[samples, ]
  
  cat("Calculating on",nrow(dd.ts$resMatrix), "samples \n")
  
  i_bsc = NULL
  for(i in 1:length(methods)){
    data.tr <- data.frame("time"=dd.tr$survival$time, "event"=dd.tr$survival$event, "score"=dd.tr$resMatrix[,i])
    ok <- complete.cases(data.tr)
    data.tr = data.tr[ok, ]
    data.ts <- data.frame("time"=dd.ts$survival$time, "event"=dd.ts$survival$event, "score"=dd.ts$resMatrix[,i])
    ok <- complete.cases(data.ts)
    data.ts = data.ts[ok, ]
    if(ncol(data.tr)==0||nrow(data.tr)==0||ncol(data.ts)==0||nrow(data.ts)==0) i_bsc = c(i_bsc, "NA")
    else {
      tt = survcomp::sbrier.score2proba(data.tr=data.tr, data.ts=data.ts, method="cox")
      i_bsc = c(i_bsc, tt$bsc.integrated)
    }
  }
  names(i_bsc)=methods
  return(i_bsc)
}

####---- hazard ratio---------------
#We use the following function to rescale the gene expression values to lie approximately in [-1,1],
#robust to extreme values (possibly outliers).

#Therefore we use the following function to rescale the gene expression values to lie approximately in [-1,1], robust to extreme values (possibly outliers).
rescale <- function(x, na.rm=TRUE, q=0.05) {
  ma <- quantile(x, probs=1-(q/2), na.rm=na.rm)
  mi <- quantile(x, probs=q/2, na.rm=na.rm)
  x <- (x - mi) / (ma - mi)
  return((x - 0.5) * 2)
}

hr <- function(data, survival, PValue = FALSE, outputFolder=NULL) {
  
  methods = toupper(colnames(data)[1:ncol(data)])
  
  samples = intersect(rownames(data), rownames(survival))
  data = data[samples, ]
  survival = survival[samples, ]
  
  cat("Calculating on",nrow(data), "samples \n")
  
  setT = survival$time
  setE = survival$event
  
  hratio <- apply(X = as.matrix(data), MARGIN = 2, 
                  FUN = function(x, y, z) { 
                    tt = survcomp::hazard.ratio(x = binarize(rescale(x)), surv.time = y, surv.event = z, na.rm = TRUE)
                    if (PValue) return ("p.value" = tt$p.value)
                    else return(c("hr" = tt$hazard.ratio, "hr.se" = tt$se, "lower" = tt$lower, "upper" = tt$upper))},
                  y = setT, z = setE )
  if(! is.null(outputFolder)) {
    dir.create(file.path(outputFolder), showWarnings = FALSE)
    write.csv(hratio,file = paste(outputFolder, "hr.csv", sep = "/"),row.names = TRUE)
  }
  else {
    outputFolder = "./output"
    dir.create(file.path(outputFolder), showWarnings = FALSE)
    write.csv(hratio,file = paste(outputFolder, "hr.csv", sep = "/"),row.names = TRUE)
  }
  return(hratio)
}


