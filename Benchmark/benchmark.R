library(CancerSubtypesPrognosis)
library(reshape2)
library(ggplot2)
library(irr)
library(gridExtra)
library(Unicode)

###calculate the risk scores
dn = c("transbig", "unt", "upp", "mainz", "nki","GSE6532", "GEO", "TCGA753", "TCGA500",
       "METABRIC", "UK", "HEL", "GSE19783")
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

# Endopredict for METABRIC
ddata = exprs(datasets$METABRIC)
DHCR7 = runif(1283)
ddata = rbind(ddata, DHCR7)
ddata[is.na(ddata)] <- 0
fd = featureData(datasets$METABRIC)@data
fd = rbind(fd, c("DHCR7","1717"))
rownames(fd)[17759] = "DHCR7"
# library(scales)
# ddata = apply(ddata, 1, rescale)
pd = pData(datasets$METABRIC)
eset = ExpressionSet(ddata, AnnotatedDataFrame(pd), AnnotatedDataFrame(fd))
res = CancerPrognosis_RNAData(data=eset, platform="custom", methods="Endopredict")
resMatrix$METABRIC$Endopredict = res

# OncotypeDx for METABRIC
BAG1 = runif(1283)
ddata = rbind(ddata, BAG1)
fd = rbind(fd, c("BAG1","573"))
rownames(fd)[17760] = "BAG1"
eset = ExpressionSet(ddata, AnnotatedDataFrame(pd), AnnotatedDataFrame(fd))
res = CancerPrognosis_RNAData(data=eset, platform="custom", methods="OncotypeDx")
resMatrix$METABRIC$OncotypeDx = res

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

#get the training data: METABRIC
dd.tr = list("resMatrix"= resMatrix$METABRIC)
for(i in 10:10){
  sampleInfo= pData(datasets[[i]])
  if(dn[i] %in% c("METABRIC")) {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs/12),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
  }
  else {
    print("Wrong datasets.")
  }
  dd.tr$survival = survival
}

samples = intersect(rownames(dd.tr$resMatrix), rownames(dd.tr$survival))
dd.tr$resMatrix = dd.tr$resMatrix[samples, ]
dd.tr$survival = dd.tr$survival[samples, ]

# Calculate IBS for each predictor
## IBS and p-value computation per algorithm
int_bsc = as.list(NULL)
for(i in 1:length(dn)){
  sampleInfo= pData(datasets[[i]])
  if (dn[i] %in% c("transbig", "unt", "mainz", "nki")) {
    survival = data.frame(time=sampleInfo$t.dmfs/365,event=sampleInfo$e.dmfs, row.names=rownames(sampleInfo))
  }
  else if (dn[i] %in% c("GSE6532", "upp")) {
    survival = data.frame(time=sampleInfo$t.rfs/365,event=sampleInfo$e.rfs, row.names=rownames(sampleInfo))
  }
  else if(dn[i] == "GSE19783") {
    survival=data.frame(time=sampleInfo$`disease free survival time (months):ch1`/12,event=sampleInfo$`death status:ch1`, row.names=rownames(sampleInfo))
  }
  else if(dn[i] == "HEL") {
    survival = data.frame(time=sampleInfo$`bddm followup time (months):ch1`/12,event=sampleInfo$`bddm status:ch1`, row.names=rownames(sampleInfo))
  }
  else if(dn[i] %in% c("TCGA753", "TCGA500")) {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs/365),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
    survival[survival[,"time"]<0,"time"] <- 0
  }
  else if(dn[i] %in% c("METABRIC")) {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs/12),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
  }
  else {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
  }
  dd.ts = list("resMatrix"=resMatrix[[i]],"survival"=survival)
  #int_bsc[[i]] = IBSC(data=resMatrix[[i]], survival)
  int_bsc[[i]] = IBSC(dd.tr, dd.ts, "cox")
}
names(int_bsc) = dn

write.csv(int_bsc,file = "ibsc.csv")

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
  else if(dn[i] == "GSE19783") {
    survival=data.frame(time=sampleInfo$`disease free survival time (months):ch1`/12,event=sampleInfo$`death status:ch1`, row.names=rownames(sampleInfo))
  }
  else if(dn[i] == "HEL") {
    survival = data.frame(time=sampleInfo$`bddm followup time (months):ch1`/12,event=sampleInfo$`bddm status:ch1`, row.names=rownames(sampleInfo))
  }
  else if(dn[i] %in% c("TCGA753", "TCGA500")) {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs/365),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
    survival[survival[,"time"]<0,"time"] <- 0
  }
  else if(dn[i] %in% c("METABRIC")) {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs/12),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
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
## cindex and p-value computation per algorithm
hrMatrix = as.list(NULL)
for(i in 1:length(dn)){
  sampleInfo= pData(datasets[[dn[i]]])
  if (dn[i] %in% c("transbig", "unt", "mainz", "nki")) {
    survival = data.frame(time=sampleInfo$t.dmfs/365,event=sampleInfo$e.dmfs, row.names=rownames(sampleInfo))
  }
  else if (dn[i] %in% c("GSE6532", "upp")) {
    survival = data.frame(time=sampleInfo$t.rfs/365,event=sampleInfo$e.rfs, row.names=rownames(sampleInfo))
  }
  else if(dn[i] == "GSE19783") {
    survival=data.frame(time=sampleInfo$`disease free survival time (months):ch1`/12,event=sampleInfo$`death status:ch1`, row.names=rownames(sampleInfo))
  }
  else if(dn[i] == "HEL") {
    survival = data.frame(time=sampleInfo$`bddm followup time (months):ch1`/12,event=sampleInfo$`bddm status:ch1`, row.names=rownames(sampleInfo))
  }
  else if(dn[i] %in% c("TCGA753", "TCGA500")) {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs/365),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
    survival[survival[,"time"]<0,"time"] <- 0
  }
  else if(dn[i] %in% c("METABRIC")) {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs/12),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
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
  else if(dn[i] == "GSE19783") {
    survival=data.frame(time=sampleInfo$`disease free survival time (months):ch1`/12,event=sampleInfo$`death status:ch1`, row.names=rownames(sampleInfo))
  }
  else if(dn[i] == "HEL") {
    survival = data.frame(time=sampleInfo$`bddm followup time (months):ch1`/12,event=sampleInfo$`bddm status:ch1`, row.names=rownames(sampleInfo))
  }
  else if(dn[i] %in% c("TCGA753", "TCGA500")) {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs/365),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
    survival[survival[,"time"]<0,"time"] <- 0
  }
  else if(dn[i] %in% c("METABRIC")) {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs/12),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
  }
  else {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
  }
  pvMatrix = rbind(pvMatrix, LogRank(data=resMatrix[[dn[i]]], survival))
}

rownames(pvMatrix) = dn


write.csv(pvMatrix,file = "LogrankPV.csv",row.names = TRUE)