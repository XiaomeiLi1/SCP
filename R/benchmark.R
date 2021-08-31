library(CancerSubtypesPrognosis)
library(reshape2)
library(ggplot2)
library(irr)
library(gridExtra)
library(Unicode)

###calculate the risk scores
dn = c("transbig", "unt", "upp", "mainz", "nki","GSE6532", "GEO", "TCGA753", "TCGA500",
       "METABRIC", "UK", "HEL")
mRNA_methods <- c("AURKA", "ESR1", "ERBB2", "GGI", "GENIUS", "Endopredict", "OncotypeDx",
                  "TAMR13", "PIK3CAGS", "GENE70", "rorS")


resMatrix <- as.list(NULL)

for (i in 1:length(dn)){
  print(dn[i])
  res = CancerPrognosis_RNAData(data=get(dn[i]), platform="custom", method=mRNA_methods)
  resMatrix[[i]] = res
}
names(resMatrix) = dn
save(resMatrix, file = "resMatrix.rda")


#AURKA for UK
resMatrix$UK$AURKA= exprs(datasets$UK)["STK6",]

# OncotypeDx for UK
fd = featureData(datasets$UK)@data
fd["STK6",2] = "6790"
fd["CTSL2",2] = "1515"
fd["GAPD",2] = "2597"
pd = pData(datasets$UK)
ddata=exprs(datasets$UK)
eset = ExpressionSet(ddata, AnnotatedDataFrame(pd), AnnotatedDataFrame(fd))

res = CancerPrognosis_RNAData(data=eset, platform="custom", methods="OncotypeDx")
resMatrix$UK$OncotypeDx = res

save(resMatrix, file = "resMatrix.rda")

# Calculate cindex for each predictor
## cindex and p-value computation per algorithm
ciMatrix = as.list(NULL)
for(i in 1:length(dn)){
  sampleInfo= pData(datasets[[i]])
  if (dn[i] %in% c("transbig", "unt", "mainz", "nki")) {
    survival = data.frame(time=sampleInfo$t.dmfs/365,event=sampleInfo$e.dmfs, row.names=rownames(sampleInfo))
  }
  else if (dn[i] %in% c("GSE6532", "upp")) {
    survival = data.frame(time=sampleInfo$t.rfs/365,event=sampleInfo$e.rfs, row.names=rownames(sampleInfo))
  }
  else if(dn[i] == "HEL") {
    survival = data.frame(time=sampleInfo$`bddm followup time (months):ch1`/12,event=sampleInfo$`bddm status:ch1`, row.names=rownames(sampleInfo))
  }
  else if(dn[i] %in% c("TCGA753", "TCGA500")) {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs/365),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
    survival[survival[,"time"]<0,"time"] <- 0
  }
  else {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
  }
  ciMatrix[[i]] = Cindex(resMatrix[[i]], survival)
}
names(ciMatrix) = dn
# resMatrix = lapply(resMatrix, function(x){return(x[,mRNA_methods])})
citemp = lapply(ciMatrix, function(x){return(x["cindex",mRNA_methods])})

write.csv(citemp,file = "Cindex.csv")


# Calculate hazard ratio for each predictor
hrMatrix = as.list(NULL)
for(i in 1:length(dn)){
  sampleInfo= pData(datasets[[dn[i]]])
  if (dn[i] %in% c("transbig", "unt", "mainz", "nki")) {
    survival = data.frame(time=sampleInfo$t.dmfs/365,event=sampleInfo$e.dmfs, row.names=rownames(sampleInfo))
  }
  else if (dn[i] %in% c("GSE6532", "upp")) {
    survival = data.frame(time=sampleInfo$t.rfs/365,event=sampleInfo$e.rfs, row.names=rownames(sampleInfo))
  }
  else if(dn[i] == "HEL") {
    survival = data.frame(time=sampleInfo$`bddm followup time (months):ch1`/12,event=sampleInfo$`bddm status:ch1`, row.names=rownames(sampleInfo))
  }
  else if(dn[i] %in% c("TCGA753", "TCGA500")) {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs/365),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
    survival[survival[,"time"]<0,"time"] <- 0
  }
  else {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
  }
  hrMatrix[[i]] = hr(resMatrix[[dn[i]]], survival)
}
names(hrMatrix) = dn
save(hrMatrix, file = "hrMatrix.rda")


# Calculate Log-rank P-value for each predictor
## P-value computation per algorithm
pvMatrix <- NULL
for(i in 1:length(dn)){
  ddata=datasets[[dn[i]]]
  sampleInfo= pData(ddata)
  if (dn[i] %in% c("transbig", "unt", "mainz", "nki")) {
    survival = data.frame(time=sampleInfo$t.dmfs/365,event=sampleInfo$e.dmfs, row.names=rownames(sampleInfo))
  }
  else if (dn[i] %in% c("GSE6532", "upp")) {
    survival = data.frame(time=sampleInfo$t.rfs/365,event=sampleInfo$e.rfs, row.names=rownames(sampleInfo))
  }
  else if(dn[i] == "HEL") {
    survival = data.frame(time=sampleInfo$`bddm followup time (months):ch1`/12,event=sampleInfo$`bddm status:ch1`, row.names=rownames(sampleInfo))
  }
  else if(dn[i] %in% c("TCGA753", "TCGA500")) {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs/365),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
    survival[survival[,"time"]<0,"time"] <- 0
  }
  else {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
  }
  pvMatrix = rbind(pvMatrix, LogRank(data=resMatrix[[dn[i]]], survival))
}

rownames(pvMatrix) = dn


write.csv(pvMatrix,file = "LogrankPV.csv",row.names = TRUE)