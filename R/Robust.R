library(Biobase)
library(survival)
library(survcomp)
#library(Hmisc)

Robust = ConvertSymbol(signatures$gene[signatures$Robust == "yes"]) #127
Robust[30] = "HJURP"
index = which(Robust %in% ConvertSymbol(rownames(datasets$METABRIC1980))) #119
Robust = Robust[index]

# Build dataset
makeDS <- function(dd = NULL, pd = NULL, list = NULL){
  # Imputation
  #dd = data.imputation(dd, fun = "median")
  # Convert to sample by gene
  temp = t(dd)
  #temp = dd
  colnames(temp) = ConvertSymbol(colnames(temp))
  expd = NULL
  for (i in list) {
    idx = which(colnames(temp) == i)
    if (length(idx)==1) expd = cbind(expd, temp[,idx])
    else if(length(idx) > 1) {
      ttemp=apply(temp[,idx], 1, mean)
      expd = cbind(expd, ttemp)
    }
    else expd = cbind(expd, rep(0, nrow(temp)))
  }
  print(dim(expd))
  colnames(expd) = list
  expd = data.frame(expd, stringsAsFactors = F)
  t1 = as.numeric(pd[,1])
  t1[t1<0] <- 0
  #t1[is.na(t1)] <- 0
  expd$time = t1
  t2 = as.numeric(pd[,2])
  t2[is.na(t2)] <- 0
  expd$status = t2
  na_index = which(is.na(expd$time))
  if(length(na_index) > 0) expd = expd[-na_index,]
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
pd = pData(datasets$METABRIC1980)
survival=data.frame(time=pd$DFS_MONTHS/12,
                    event=as.numeric(pd$DFS_STATUS)-1,
                    row.names=rownames(pd))
train_data = makeDS(exprs(datasets$METABRIC1980),survival,Robust)
coxph <- coxph(Surv(time,status)~.,data=train_data)

hr = NULL
pvs = NULL
R_ci = NULL

# dn = c("transbig", "unt", "upp", "mainz", "nki","GSE6532", "GEO", "TCGA753", "TCGA500",
#        "UK", "HEL")
for(i in dn){
  print(i)
  ddata=exprs(datasets[[i]])
  sampleInfo= pData(datasets[[i]])
  na_index = NULL
  if (i %in% c("transbig", "unt", "mainz", "nki")) {
    survival = data.frame(time=sampleInfo$t.dmfs/365,event=sampleInfo$e.dmfs, row.names=rownames(sampleInfo))
    na_index = which(is.na(survival$time))
  }
  else if (i %in% c("GSE6532", "upp")) {
    survival = data.frame(time=sampleInfo$t.rfs/365,event=sampleInfo$e.rfs, row.names=rownames(sampleInfo))
    na_index = which(is.na(survival$time))
  }
  else if(i == "HEL") {
    survival = data.frame(time=sampleInfo$`bddm followup time (months):ch1`/12,event=sampleInfo$`bddm status:ch1`, row.names=rownames(sampleInfo))
    na_index = which(is.na(survival$time))
  }
  else if(i %in% c("TCGA753", "TCGA500")) {
    #survival=data.frame(time=as.numeric(sampleInfo$t.rfs/365),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
    temp = survival_data[rownames(sampleInfo),c(3,4)] 
    survival=data.frame(time=temp[,1]/365,event=temp[,2], row.names=rownames(sampleInfo))
    #survival[survival[,"time"]<0,"time"] <- 0
    na_index = which(is.na(survival$time))
  }
  else if(i %in% c("TCGA1093")) {
    survival=data.frame(time=sampleInfo$new_time/365,event=sampleInfo$new_event, row.names=rownames(sampleInfo))
    na_index = which(is.na(survival$time))
  }
  else {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
    na_index = which(is.na(survival$time))
  }
  # remove the samples do not have survival time
  if(length(na_index) > 0) {
    ddata = ddata[,-na_index]
    survival = survival[-na_index,]
  }
  survival[survival$time<0,"time"] <- 0
  survival[is.na(survival$event),"event"] <- 0
  test_data = makeDS(ddata,survival,Robust)
  predict1 <- predict(coxph,newdata=test_data)
  s = Surv(survival[,1], survival[,2])
  tt = survcomp::concordance.index(x = predict1, surv.time = survival[,1], surv.event = survival[,2], method = "noether", na.rm = TRUE)
  R_ci = c(R_ci,tt$c.index)
  
  #hazard ratio
  tt = survcomp::hazard.ratio(x = binarize(rescale(predict1)), 
                              surv.time = survival[,1], surv.event = survival[,2], na.rm = TRUE)
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

temp = rbind(R_ci, hr, pvs)
colnames(temp) = dn
write.csv(temp, file = "Robust.csv")
