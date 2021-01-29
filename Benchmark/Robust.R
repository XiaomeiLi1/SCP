Robust = signatures$gene[signatures$Robust == "yes"] #127
index = which(Robust %in% rownames(datasets$METABRIC))
Robust = Robust[index]
library(Biobase)

makeDS <- function(dd = NULL, pd = NULL, list = NULL){
  temp = t(dd)
  expd = NULL
  for (i in list) {
    idx = which(colnames(temp) == i)
    if (length(idx)) expd = cbind(expd, temp[,idx])
    else expd = cbind(expd, rep(0, nrow(temp)))
  }
  colnames(expd) = list
  expd = data.frame(expd, stringsAsFactors = F)
  t1 = as.numeric(pd[,1])
  t1[is.na(t1)] <- 0
  t1[t1<0] <- 0
  expd$time = t1
  t2 = as.numeric(pd[,2])
  expd$status = t2
  # pd = data.frame(cbind(t1,t2))
  # colnames(pd) = c("time", "status")
  # if(nrow(expd) == 1) {data = cbind(t(expd), pd)
  # colnames(data) = c("G1","time", "status")
  # }
  # else data = cbind(pd,expd)
  # datasets <- data.frame(data,stringsAsFactors = F)
  return(expd)
}

#training
pd = pData(datasets$METABRIC)[,c(5,6)]
train_data = makeDS(exprs(datasets$METABRIC),pd,Robust)
coxph1 <- coxph(Surv(time,status)~.,data=train_data)
predict1 <- predict(coxph1,newdata=train_data)
# s = Surv(pd[,1], pd[,2])
# a = Hmisc::rcorr.cens(-1 * predict1, s)[["C Index"]]
data.tr <- data.frame("time"=pd[,1], "event"=pd[,2], "score"=predict1)
ok <- complete.cases(data.tr)
data.tr = data.tr[ok, ]
i_bsc = NULL
hr = NULL
pvs = NULL

dn = c("transbig", "unt", "upp", "mainz", "nki","GSE6532", "GEO", "TCGA753", "TCGA500",
       "UK", "HEL", "GSE19783")
for(i in dn){
  sampleInfo= pData(datasets[[i]])
  if (i %in% c("transbig", "unt", "mainz", "nki")) {
    survival = data.frame(time=sampleInfo$t.dmfs/365,event=sampleInfo$e.dmfs, row.names=rownames(sampleInfo))
  }
  else if (i %in% c("GSE6532", "upp")) {
    survival = data.frame(time=sampleInfo$t.rfs/365,event=sampleInfo$e.rfs, row.names=rownames(sampleInfo))
  }
  else if(i == "GSE19783") {
    survival=data.frame(time=sampleInfo$`disease free survival time (months):ch1`/12,event=sampleInfo$`death status:ch1`, row.names=rownames(sampleInfo))
  }
  else if(i == "HEL") {
    survival = data.frame(time=sampleInfo$`bddm followup time (months):ch1`/12,event=sampleInfo$`bddm status:ch1`, row.names=rownames(sampleInfo))
  }
  else if(i %in% c("TCGA753", "TCGA500")) {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs/365),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
    survival[survival[,"time"]<0,"time"] <- 0
  }
  else if(i %in% c("METABRIC")) {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs/12),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
  }
  else {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
  }
  test_data = makeDS(exprs(datasets[[i]]),survival,Robust)
  predict1 <- predict(coxph1,newdata=test_data)
  s = Surv(survival[,1], survival[,2])
  #a = c(a,Hmisc::rcorr.cens(-1 * predict1, s)[["C Index"]])
  
  #IBS
  data.ts <- data.frame("time"=survival[,1], "event"=survival[,2], "score"=predict1)
  ok <- complete.cases(data.ts)
  data.ts = data.ts[ok, ]
  if(ncol(data.tr)==0||nrow(data.tr)==0||ncol(data.ts)==0||nrow(data.ts)==0) i_bsc = c(i_bsc, "NA")
  else {
    tt = survcomp::sbrier.score2proba(data.tr=data.tr, data.ts=data.ts, method="cox")
    i_bsc = c(i_bsc, tt$bsc.integrated)
  }
  
  #hazard ratio
  tt = survcomp::hazard.ratio(x = rescale(predict1), surv.time = survival[,1], surv.event = survival[,2], na.rm = TRUE)
  hr = c(hr,tt$hazard.ratio)
  
  #p-value
  group = binarize(predict1)
  clusterNum=length(unique(group))
  if (clusterNum>1){
    sdf=survdiff(Surv(survival[,1], survival[,2]) ~ group)
    p_value=1 - pchisq(sdf$chisq, length(sdf$n) - 1)
  }
  else
  {
    p_value=1
  }
  pvs = c(pvs, p_value)
}

temp = rbind(R_ci, i_bsc, hr, pvs)
write.csv(temp, file = "temp.csv")
