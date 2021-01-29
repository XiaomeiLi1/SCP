# Step 1: change the format of data (genes expressed by gene symbol)
pd = pData(GSE6532) #sample * clincal
ddata = exprs(GSE6532) #gene * sample
fd = featureData(GSE6532) # gene * featurecolumns
temp = ddata
rownames(temp) = fd@data[,"Gene.symbol"]
## Merge the same probles of miRNAs with average value
temp=apply(temp, 2, function(x) tapply(x, rownames(temp), mean))
ddata = temp
GSE6532 <- ExpressionSet(assayData = ddata,
            phenoData = AnnotatedDataFrame(pd))
save(GSE6532,file = "GSE6532.rda")

pd = pData(nki) #sample * clincal
ddata = exprs(nki) #gene * sample
fd = featureData(nki) # gene * featurecolumns
temp = ddata
rownames(temp) = fd@data[,"HUGO.gene.symbol"]
## Merge the same probles of miRNAs with average value
temp=apply(temp, 2, function(x) tapply(x, rownames(temp), mean))
ddata = temp
nki <- ExpressionSet(assayData = ddata,
                         phenoData = AnnotatedDataFrame(pd))
save(nki,file = "nki.rda")

pd = pData(mainz) #sample * clincal
ddata = exprs(mainz) #gene * sample
fd = featureData(mainz) # gene * featurecolumns
temp = ddata
rownames(temp) = fd@data[,"Gene.symbol"]
## Merge the same probles of miRNAs with average value
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
## Merge the same probles of miRNAs with average value
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
## Merge the same probles of miRNAs with average value
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
## Merge the same probles of miRNAs with average value
temp=apply(temp, 2, function(x) tapply(x, rownames(temp), mean))
ddata = temp
transbig <- ExpressionSet(assayData = ddata,
                     phenoData = AnnotatedDataFrame(pd))
save(transbig,file = "transbig.rda")

# print the dimentions of datasets
for(i in 1:length(dn)){
  ddata=get(dn[i])
  print(dim(ddata))
}

#Step 2: Convert Gene Aliases to Official Gene Symbols for all the datasets
library(limma)
datasets = vector("list",13)
names(datasets) = c("transbig", "unt", "upp", "mainz", "nki","GSE6532", "GEO", "TCGA753", "TCGA500",
                    "METABRIC", "UK", "HEL", "GSE19783")
##METABRIC
OfficialGeneSymbol = alias2SymbolTable(rownames(METABRIC),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
index = c(1:823, na_index)
index = unique(index)
temp = METABRIC[-index,]
OfficialGeneSymbol = OfficialGeneSymbol[-index]
METABRIC = temp
datasets$METABRIC = METABRIC

##TCGA753
OfficialGeneSymbol = alias2SymbolTable(rownames(TCGA753),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
index = c(1:446, na_index)
index = unique(index)
temp = TCGA753[-index,]
TCGA753 = temp
datasets$TCGA753 = TCGA753

##transbig
OfficialGeneSymbol = alias2SymbolTable(rownames(transbig),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
temp = transbig[-na_index,]
transbig = temp
datasets$transbig = transbig

##unt
OfficialGeneSymbol = alias2SymbolTable(rownames(unt),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
temp = unt[-na_index,]
unt = temp
datasets$unt = unt

##upp
OfficialGeneSymbol = alias2SymbolTable(rownames(upp),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
temp = upp[-na_index,]
upp = temp
datasets$upp = upp

##mainz
OfficialGeneSymbol = alias2SymbolTable(rownames(mainz),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
temp = mainz[-na_index,]
mainz = temp
datasets$mainz = mainz

##nki
OfficialGeneSymbol = alias2SymbolTable(rownames(nki),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
temp = nki[-na_index,]
nki = temp
datasets$nki = nki

##GSE6532
OfficialGeneSymbol = alias2SymbolTable(rownames(GSE6532),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
temp = GSE6532[-na_index,]
GSE6532 = temp
datasets$GSE6532 = GSE6532

##GEO
OfficialGeneSymbol = alias2SymbolTable(rownames(GEO),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
temp = GEO[-na_index,]
GEO = temp
datasets$GEO = GEO

##TCGA500
OfficialGeneSymbol = alias2SymbolTable(rownames(TCGA500),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
temp = TCGA500[-na_index,]
TCGA500 = temp
datasets$TCGA500 = TCGA500

##UK
OfficialGeneSymbol = alias2SymbolTable(rownames(UK),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
temp = UK[-na_index,]
UK = temp
datasets$UK = UK

##HEL
OfficialGeneSymbol = alias2SymbolTable(rownames(HEL),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
temp = HEL[-na_index,]
HEL = temp
datasets$HEL = HEL

##GSE19783
OfficialGeneSymbol = alias2SymbolTable(rownames(GSE19783),species = "Hs")
na_index = which(is.na(OfficialGeneSymbol))
temp = GSE19783[-na_index,]
GSE19783 = temp
datasets$GSE19783 = GSE19783

# print the dimentions of datasets
for(i in 1:length(datasets)){
  print(dim(datasets[[i]]))
}

##save datasets
save(datasets,file = "datasets.rda")

#common genes in all the datasets
library(limma)
intersect_genes=Reduce(intersect,  list(alias2SymbolTable(rownames(datasets$transbig)),alias2SymbolTable(rownames(datasets$unt)),
                                        alias2SymbolTable(rownames(datasets$upp)), alias2SymbolTable(rownames(datasets$mainz)),
                                         alias2SymbolTable(rownames(datasets$nki)),alias2SymbolTable(rownames(datasets$GSE6532)),
                                        alias2SymbolTable(rownames(datasets$GEO)),alias2SymbolTable(rownames(datasets$TCGA753)),
                                        alias2SymbolTable(rownames(datasets$TCGA500)),alias2SymbolTable(rownames(datasets$HEL)),
                                        alias2SymbolTable(rownames(datasets$METABRIC)),alias2SymbolTable(rownames(datasets$UK)),
                                        alias2SymbolTable(rownames(datasets$GSE19783)))) #7658

save(intersect_genes, file = "intersect_genes.rda")

#Step 3: prepare training dataset (METABRIC 1283)
ddata=exprs(datasets$METABRIC)
#rmean = rowMeans(ddata, na.rm = T)
OfficialGeneSymbol = alias2SymbolTable(rownames(datasets$METABRIC),species = "Hs")
rownames(ddata) = OfficialGeneSymbol
ddata = ddata[intersect_genes,]
sampleInfo= pData(datasets$METABRIC)
survival=data.frame(time=as.numeric(sampleInfo$t.rfs),event=as.numeric(sampleInfo$e.rfs), row.names=rownames(sampleInfo))
library(scales)
data = apply(ddata, 1, rescale)
data = cbind(data, survival)
##imputation
library(CancerSubtypes)
data = data.imputation(t(data), fun = "median")
##survival time (year)
data["time",] = data["time",]/12
write.csv(t(data), file = paste0(dn,".csv"), row.names = F)

#Step 4: prepare testing dataset
dn = c("transbig", "unt", "upp", "mainz", "nki","GSE6532", "GEO", "TCGA753", "TCGA500",
       "METABRIC", "UK", "HEL", "GSE19783")

library(CancerSubtypes)

for(i in 1:length(dn)){
  ddata=exprs(datasets[[i]])
  OfficialGeneSymbol = alias2SymbolTable(rownames(ddata),species = "Hs")
  rownames(ddata) = OfficialGeneSymbol
  ddata = ddata[intersect_genes,]
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
  data = apply(ddata, 1, rescale)
  data = cbind(data, survival)
  data = data.imputation(t(data), fun = "median")
  write.csv(t(data), file = paste0(dn[i],".csv"), row.names = F)
}

#Step 5: Analyse the survival time
dn = c("transbig", "unt", "upp", "mainz", "nki","GSE6532", "GEO", "TCGA753", "TCGA500",
       "METABRIC", "UK", "HEL", "GSE19783")

library(CancerSubtypes)
survival_time_quantile = NULL

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
  survival_time_quantile = rbind(survival_time_quantile,quantile(survival[,"time"],na.rm = T))
}
rownames(survival_time_quantile) = dn
survival_time_quantile
write.csv(survival_time_quantile, file = "survival_time_quantile.csv")
