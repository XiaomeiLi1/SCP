# Step 1: change the format of data (genes expressed by gene symbol)
pd = pData(GSE6532) #sample * clincal
ddata = exprs(GSE6532) #gene * sample
fd = featureData(GSE6532) # gene * featurecolumns
temp = ddata
rownames(temp) = fd@data[,"Gene.symbol"] #EntrezGene.ID
## Merge the same probles of mRNAs with average value.
temp=apply(temp, 2, function(x) tapply(x, rownames(temp), mean))
ddata = temp
GSE6532 <- ExpressionSet(assayData = ddata,
            phenoData = AnnotatedDataFrame(pd))
save(GSE6532,file = "GSE6532.rda")

pd = pData(nki) #sample * clincal
ddata = exprs(nki) #gene * sample
fd = featureData(nki) # gene * featurecolumns
temp = ddata
rownames(temp) = fd@data[,"HUGO.gene.symbol"] #EntrezGene.ID
## Merge the same probles of mRNAs with average value
temp=apply(temp, 2, function(x) tapply(x, rownames(temp), mean))
ddata = temp
nki <- ExpressionSet(assayData = ddata,
                         phenoData = AnnotatedDataFrame(pd))
save(nki,file = "nki.rda")

pd = pData(mainz) #sample * clincal
ddata = exprs(mainz) #gene * sample
fd = featureData(mainz) # gene * featurecolumns
temp = ddata
rownames(temp) = fd@data[,"Gene.symbol"] #EntrezGene.ID
## Merge the same probles of mRNAs with average value
temp=apply(temp, 2, function(x) tapply(x, rownames(temp), mean))
ddata = temp
mainz <- ExpressionSet(assayData = ddata,
                     phenoData = AnnotatedDataFrame(pd))
save(mainz,file = "mainz.rda")

pd = pData(unt) #sample * clincal
ddata = exprs(unt) #gene * sample
fd = featureData(unt) # gene * featurecolumns
temp = ddata
rownames(temp) = fd@data[,"Gene.symbol"]
## Merge the same probles of mRNAs with average value
temp=apply(temp, 2, function(x) tapply(x, rownames(temp), mean))
ddata = temp
unt <- ExpressionSet(assayData = ddata,
                       phenoData = AnnotatedDataFrame(pd))
save(unt,file = "unt.rda")

pd = pData(upp) #sample * clincal
ddata = exprs(upp) #gene * sample
fd = featureData(upp) # gene * featurecolumns
temp = ddata
rownames(temp) = fd@data[,"Gene.symbol"]
## Merge the same probles of mRNAs with average value
temp=apply(temp, 2, function(x) tapply(x, rownames(temp), mean))
ddata = temp
upp <- ExpressionSet(assayData = ddata,
                     phenoData = AnnotatedDataFrame(pd))
save(upp,file = "upp.rda")

pd = pData(transbig) #sample * clincal
ddata = exprs(transbig) #gene * sample
fd = featureData(transbig) # gene * featurecolumns
temp = ddata
rownames(temp) = fd@data[,"Gene.symbol"]
## Merge the same probles of mRNAs with average value
temp=apply(temp, 2, function(x) tapply(x, rownames(temp), mean))
ddata = temp
transbig <- ExpressionSet(assayData = ddata,
                     phenoData = AnnotatedDataFrame(pd))
save(transbig,file = "transbig.rda")

#TCGA 1093
#death_time vital_status --> OS
#new_time, new_event --> replase
load("data/BRCA-Xiaomei-mRNA-Clinical-1093.RData")
ddata = RNASeq_data #gene * sample
fd = names(rownames(RNASeq_data))
#split gene symbols and ENtrez ID
EID <- sapply(fd, function(x) unlist(strsplit(x,"\\|"))[[2]])
gsym <- sapply(fd, function(x) unlist(strsplit(x,"\\|"))[[1]])
EID[which(duplicated(gsym))]
gsym[which(duplicated(gsym))]
# convert ENTREZID to SYMBOL
library(DOSE)
library(org.Hs.eg.db)
library(topGO)
library(clusterProfiler)
library(pathview)

test = bitr(EID[which(duplicated(gsym))], fromType="ENTREZID", toType="SYMBOL", OrgDb="org.Hs.eg.db")
temp = read.csv("temp1.csv",header = F)
gsym[which(duplicated(gsym))] = temp$V3
gsym[14140] = "SLC35E2A"
na_index = which(is.na(gsym))
RNASeq_data = RNASeq_data[-na_index,]
rownames(RNASeq_data) = gsym[-na_index]
 
TCGA_mRNA_Anno = data.frame("Gene.Symbol" = gsym[-na_index], 
                            "EntrezGene.ID"=EID[-na_index]) 
rownames(TCGA_mRNA_Anno) = gsym[-na_index]

which(duplicated(rownames(RNASeq_data)))

TCGA1093 <- ExpressionSet(assayData = RNASeq_data,
                          phenoData = AnnotatedDataFrame(survival_data))

# TCGA1093 <- ExpressionSet(assayData = RNASeq_data,
#                           phenoData = AnnotatedDataFrame(survival_data),
#                           featureData = AnnotatedDataFrame(TCGA_mRNA_Anno))

save( TCGA1093, file = "TCGA1093.rda")

#METABRIC 1980
# OS_MONTHS OS_STATUS
# DFS_MONTHS  DFS_STATUS
load("data/METABRIC_mRNA_clin_1980.rda")

all(rownames(MBR_mRData)==MBR_clin$PATIENT_ID)
all(rownames(MBR_mRData)==rownames(MBR_clin))
rownames(MBR_clin) = MBR_clin$PATIENT_ID

METABRIC1980 <- ExpressionSet(assayData = t(MBR_mRData),
                          phenoData = AnnotatedDataFrame(MBR_clin))

save(METABRIC1980, file = "METABRIC1980.rda")


#Step 2: Convert Gene Aliases to Official Gene Symbols for all the datasets
library(limma)
## Convert gene symbols to HUGO, if no HUGO gene symbol was found, then keep the original one
ConvertSymbol <- function(GeneSymbol = NULL, species = "Hs"){
  res = alias2SymbolTable(GeneSymbol,species = "Hs")
  na_index = which(is.na(res))
  res[na_index] = GeneSymbol[na_index]
  return(res) 
}

datasets = vector("list",13)
names(datasets) = c("transbig", "unt", "upp", "mainz", "nki","GSE6532", "GEO", "TCGA753", "TCGA500",
                    "UK", "HEL","TCGA1093","METABRIC1980")

##TCGA753
OfficialGeneSymbol = ConvertSymbol(rownames(TCGA753),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
index = c(1:446, na_index)
index = unique(index)
temp = TCGA753[-index,]
TCGA753 = temp
datasets$TCGA753 = TCGA753

##transbig
OfficialGeneSymbol = ConvertSymbol(rownames(transbig),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
if(length(na_index) > 0) {
  temp = transbig[-na_index,]
  transbig = temp}
datasets$transbig = transbig

##unt
OfficialGeneSymbol = ConvertSymbol(rownames(unt),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
if(length(na_index) > 0) {
  temp = unt[-na_index,]
  unt = temp}
datasets$unt = unt

##upp
OfficialGeneSymbol = ConvertSymbol(rownames(upp),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
if(length(na_index) > 0) {
  temp = upp[-na_index,]
  upp = temp}
datasets$upp = upp

##mainz
OfficialGeneSymbol = ConvertSymbol(rownames(mainz),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
if(length(na_index) > 0) {
  temp = mainz[-na_index,]
  mainz = temp}
datasets$mainz = mainz

##nki
OfficialGeneSymbol = ConvertSymbol(rownames(nki),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
if(length(na_index) > 0) {
  temp = nki[-na_index,]
  nki = temp}
datasets$nki = nki

##GSE6532
OfficialGeneSymbol = ConvertSymbol(rownames(GSE6532),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
if(length(na_index) > 0) {
  temp = GSE6532[-na_index,]
  GSE6532 = temp}
datasets$GSE6532 = GSE6532

##GEO
OfficialGeneSymbol = ConvertSymbol(rownames(GEO),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
if(length(na_index) > 0) {
  temp = GEO[-na_index,]
  GEO = temp}
datasets$GEO = GEO

##TCGA500
OfficialGeneSymbol = ConvertSymbol(rownames(TCGA500),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
if(length(na_index) > 0) {
  temp = TCGA500[-na_index,]
  TCGA500 = temp}
datasets$TCGA500 = TCGA500

##UK
OfficialGeneSymbol = ConvertSymbol(rownames(UK),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
if(length(na_index) > 0) {
  temp = UK[-na_index,]
  UK = temp}
datasets$UK = UK

##HEL
OfficialGeneSymbol = ConvertSymbol(rownames(HEL),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
if(length(na_index) > 0) {
  temp = HEL[-na_index,]
  HEL = temp}
datasets$HEL = HEL

##METABRIC1980
OfficialGeneSymbol = ConvertSymbol(rownames(METABRIC1980),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
if(length(na_index) > 0) {
  temp = METABRIC1980[-na_index,]
  METABRIC1980 = temp}
datasets$METABRIC1980 = METABRIC1980

##TCGA1093
OfficialGeneSymbol = ConvertSymbol(rownames(TCGA1093),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
if(length(na_index) > 0) {
  temp = TCGA1093[-na_index,]
  TCGA1093 = temp}
datasets$TCGA1093 = TCGA1093

#common genes in all the datasets
intersect_genes=Reduce(intersect,  list(ConvertSymbol(rownames(datasets$transbig)),ConvertSymbol(rownames(datasets$unt)),
                                        ConvertSymbol(rownames(datasets$upp)), ConvertSymbol(rownames(datasets$mainz)),
                                        ConvertSymbol(rownames(datasets$nki)),ConvertSymbol(rownames(datasets$GSE6532)),
                                        ConvertSymbol(rownames(datasets$GEO)),ConvertSymbol(rownames(datasets$TCGA753)),
                                        ConvertSymbol(rownames(datasets$TCGA500)),ConvertSymbol(rownames(datasets$HEL)),
                                        ConvertSymbol(rownames(datasets$METABRIC)),ConvertSymbol(rownames(datasets$UK)),
                                        ConvertSymbol(rownames(datasets$GSE19783)))) #7913


#Step 3: prepare datasets
dn = c("transbig", "unt", "upp", "mainz", "nki","GSE6532", "GEO", "TCGA753", "TCGA500",
       "UK", "HEL", "TCGA1093","METABRIC1980")

library(CancerSubtypes)
library(scales)

for(i in 1:length(dn)){
  ddata=exprs(datasets[[i]])
  OfficialGeneSymbol = ConvertSymbol(rownames(ddata),species = "Hs")
  rownames(ddata) = OfficialGeneSymbol
  sampleInfo= pData(datasets[[i]])
  na_index = NULL
  if (dn[i] %in% c("transbig", "unt", "mainz", "nki")) {
    survival = data.frame(time=sampleInfo$t.dmfs/365,event=sampleInfo$e.dmfs, row.names=rownames(sampleInfo))
    na_index = which(is.na(survival$time))
  }
  else if (dn[i] %in% c("GSE6532", "upp")) {
    survival = data.frame(time=sampleInfo$t.rfs/365,event=sampleInfo$e.rfs, row.names=rownames(sampleInfo))
    na_index = which(is.na(survival$time))
  }
  else if(dn[i] == "HEL") {
    survival = data.frame(time=sampleInfo$`bddm followup time (months):ch1`/12,event=sampleInfo$`bddm status:ch1`, row.names=rownames(sampleInfo))
    na_index = which(is.na(survival$time))
  }
  else if(dn[i] %in% c("TCGA753", "TCGA500")) {
    #survival=data.frame(time=as.numeric(sampleInfo$t.rfs/365),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
    temp = survival_data[rownames(sampleInfo),c(3,4)] 
    survival=data.frame(time=temp[,1]/365,event=temp[,2], row.names=rownames(sampleInfo))
    #survival[survival[,"time"]<0,"time"] <- 0
    na_index = which(is.na(survival$time))
  }
  else if(dn[i] %in% c("TCGA1093")) {
    survival=data.frame(time=sampleInfo$new_time/365,event=sampleInfo$new_event, row.names=rownames(sampleInfo))
    na_index = which(is.na(survival$time))
  }
  else if(dn[i] %in% c("METABRIC1980")) {
    survival=data.frame(time=sampleInfo$DFS_MONTHS/12,event=as.numeric(sampleInfo$DFS_STATUS)-1, row.names=rownames(sampleInfo))
    na_index = which(is.na(survival$time))
  }
  else {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
    na_index = which(is.na(survival$time))
  }
  # remove the samples do not have survival time
  if(length(na_index) > 0) {
    ddata = ddata[intersect_genes,-na_index]
    survival = survival[-na_index,]
  }
  else ddata = ddata[intersect_genes,]
  survival[survival[,"time"]<0,"time"] <- 0
  survival[is.na(survival$event),"event"] <- 0
  # imputation of gene expression profiles
  ddata = data.imputation(ddata, fun = "median")
  # normalized data in the range 0-1, and transpose the matrix
  data = apply(ddata, 1, rescale)
  data = cbind(data, survival)
  print(dim(data))
  write.csv(data, file = paste0(dn[i],".csv"), row.names = F)
}

#Step 4: Analyse the survival time

survival_time_quantile = NULL

for(i in 1:length(dn)){
  sampleInfo= pData(datasets[[i]])
  na_index = NULL
  if (dn[i] %in% c("transbig", "unt", "mainz", "nki")) {
    survival = data.frame(time=sampleInfo$t.dmfs/365,event=sampleInfo$e.dmfs, row.names=rownames(sampleInfo))
    na_index = which(is.na(survival$time))
  }
  else if (dn[i] %in% c("GSE6532", "upp")) {
    survival = data.frame(time=sampleInfo$t.rfs/365,event=sampleInfo$e.rfs, row.names=rownames(sampleInfo))
    na_index = which(is.na(survival$time))
  }
  else if(dn[i] == "HEL") {
    survival = data.frame(time=sampleInfo$`bddm followup time (months):ch1`/12,event=sampleInfo$`bddm status:ch1`, row.names=rownames(sampleInfo))
    na_index = which(is.na(survival$time))
  }
  else if(dn[i] %in% c("TCGA753", "TCGA500")) {
    #survival=data.frame(time=as.numeric(sampleInfo$t.rfs/365),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
    temp = survival_data[rownames(sampleInfo),c(3,4)] 
    survival=data.frame(time=temp[,1]/365,event=temp[,2], row.names=rownames(sampleInfo))
    #survival[survival[,"time"]<0,"time"] <- 0
    na_index = which(is.na(survival$time))
  }
  else if(dn[i] %in% c("TCGA1093")) {
    survival=data.frame(time=sampleInfo$new_time/365,event=sampleInfo$new_event, row.names=rownames(sampleInfo))
    na_index = which(is.na(survival$time))
  }
  else if(dn[i] %in% c("METABRIC1980")) {
    survival=data.frame(time=sampleInfo$DFS_MONTHS/12,event=as.numeric(sampleInfo$DFS_STATUS)-1, row.names=rownames(sampleInfo))
    na_index = which(is.na(survival$time))
  }
  else {
    survival=data.frame(time=as.numeric(sampleInfo$t.rfs),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
    na_index = which(is.na(survival$time))
  }
  # remove the samples do not have survival time
  if(length(na_index) > 0) {
    survival = survival[-na_index,]
  }
  survival[survival[,"time"]<0,"time"] <- 0
  survival_time_quantile = rbind(survival_time_quantile,quantile(survival[,"time"],na.rm = T))
}
rownames(survival_time_quantile) = dn
survival_time_quantile
write.csv(survival_time_quantile, file = "survival_time_quantile.csv")
