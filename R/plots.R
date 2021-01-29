ibsc = read.csv("./result/ibsc.csv")
rownames(ibsc) = ibsc$X
ibsc=ibsc[,-1]
colnames(ibsc) = toupper(colnames(ibsc))

####----plot table of ibsc---------------
data = ibsc

library(reshape2)
library(ggplot2)
library(ggrepel)

refine = melt(as.matrix(data))
refine$Var1 = as.factor(refine$Var1)
refine$Var2 = as.factor(refine$Var2)
refine$value = round(refine$value, digits = 3)

pdf(file = "ibsc.pdf",width = 9, height = 7.8)
ggplot(refine, aes(Var2, Var1)) +
  geom_tile(aes(fill=value), color="white")+
  xlab("Dataset") + ylab("Cancer prognosis method") +
  labs(fill = "IBS") +
  geom_text(aes(label = value), size=3 ) +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position="bottom",
        plot.title = element_text(color="Black", size=14, face="bold", hjust = 0.5),
        axis.title.x = element_text(color="Black", size=12, face="plain"),
        axis.title.y = element_text(color="Black", size=12, face="plain"),
        axis.text.x = element_text(angle=45, vjust=1, hjust=1))+
  scale_fill_gradient(low = "steelblue", high = "white") +
  coord_equal()

dev.off()

rm = rowMeans(data)
rsd = apply(data,1, sd)
stability = rm+rsd
stability = data.frame(stability)
colnames(stability)= "stability"
rownames(stability) = rownames(data)

df = melt(as.matrix(stability))
df$Var1 = as.factor(df$Var1)
df$Var2 = as.factor(df$Var2)
df$value = round(df$value, digits = 3)

pdf(file = "ibsc2.pdf",width = 9, height = 7.65)
ggplot(df, aes(Var2, Var1)) +
  geom_tile(aes(fill=value), color="white")+
  xlab("Stability") + 
  labs(fill = expression(Stability["ibs"])) +
  geom_text(aes(label = value), size=3 ) +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position="bottom",
        plot.title = element_text(color="Black", size=14, face="bold", hjust = 0.5),
        axis.title.x = element_text(color="Black", size=12, face="plain"),
        axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.text.x = element_text(angle=45, vjust=1, hjust=1))+
  scale_fill_gradient(low = "grey70", high = "grey100") +
  coord_equal()
dev.off()

####----plot barplot of ibsc---------------
data = ibsc["DGBCox",]
df=data.frame(row.names = names(data))
df$x = names(data)
df$x = factor(df$x, level=names(data))
df$y = round(t(data), digits = 3)

pdf(file = "ibsc_bar.pdf",width = 7, height = 5)
ggplot(data=df, aes(x=x, y=y)) +
  xlab("Dataset") + 
  ylab("Integrated Brier score")+
  geom_bar(stat="identity",width=0.5)+
  theme(
      axis.title.x = element_text(size = 12, face = "bold"),
      axis.text.x = element_text(size = 12, angle =  45, vjust = 0.5, hjust=0.5),
      axis.title.y = element_text(size = 12, face = "bold"),
      axis.text.y = element_text(size = 12, face = "bold"),
      #panel.background = element_blank(),
      #panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(colour = "black"),
      panel.border = element_rect(colour = "black", fill=NA, size=1))+
  scale_y_continuous(expand = c(0,0),
                     limits = c(0,0.3),
                     breaks  = seq(0,0.3, by = 0.1)) +
  geom_text(aes(label=y), position=position_dodge(width=0.5), vjust=-0.25)+
  geom_hline(yintercept= 0.25,color = "red",size=0.8, linetype="dashed")
dev.off()


cindex = read.csv("./result/ci.csv",stringsAsFactors = F, header =  T)
rownames(cindex) = cindex$ï..
cindex=cindex[,-1]
colnames(cindex) = toupper(colnames(cindex))

####----plot table of cindex---------------
data = cindex

library(reshape2)
library(ggplot2)
library(ggrepel)

refine = melt(as.matrix(data))
refine$Var1 = as.factor(refine$Var1)
refine$Var2 = as.factor(refine$Var2)
refine$value = round(refine$value, digits = 3)

pdf(file = "cindex.pdf",width = 9, height = 7.8)
ggplot(refine, aes(Var2, Var1)) +
  geom_tile(aes(fill=value), color="white")+
  xlab("Dataset") + ylab("Cancer prognosis method") +
  labs(fill = "C-index") +
  geom_text(aes(label = value), size=3 ) +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position="bottom",
        plot.title = element_text(color="Black", size=14, face="bold", hjust = 0.5),
        axis.title.x = element_text(color="Black", size=12, face="plain"),
        axis.title.y = element_text(color="Black", size=12, face="plain"),
        axis.text.x = element_text(angle=45, vjust=1, hjust=1))+
  scale_fill_gradient(low = "white", high = "steelblue") +
  coord_equal()

dev.off()

rm = rowMeans(data)
rsd = apply(data,1, sd)
stability = rm-rsd
stability = data.frame(stability)
colnames(stability)= "stability"
rownames(stability) = rownames(data)

df = melt(as.matrix(stability))
df$Var1 = as.factor(df$Var1)
df$Var2 = as.factor(df$Var2)
df$value = round(df$value, digits = 3)

pdf(file = "cindex2.pdf",width = 9, height = 7.65)
ggplot(df, aes(Var2, Var1)) +
  geom_tile(aes(fill=value), color="white")+
  xlab("Stability") + 
  labs(fill = expression(Stability["ci"])) +
  geom_text(aes(label = value), size=3 ) +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position="bottom",
        plot.title = element_text(color="Black", size=14, face="bold", hjust = 0.5),
        axis.title.x = element_text(color="Black", size=12, face="plain"),
        axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.text.x = element_text(angle=45, vjust=1, hjust=1))+
  scale_fill_gradient(low = "grey100", high = "grey70") +
  coord_equal()
dev.off()

####----plot forest plot of cindex---------------
library(survcomp)
dn = c("transbig", "unt", "upp", "mainz", "nki","GSE6532", "GEO", "TCGA753", 
       "TCGA500", "UK", "HEL", "GSE19783")
res = matrix(data=NA, nrow = length(dn),ncol = 4)
for (i in 1:length(dn)){
  data = read.csv(file = paste0("result/",dn[i],".csv"),header = F)
  tt = survcomp::concordance.index(x = data[,3], surv.time = data[,1], surv.event = data[,2], method = "noether", na.rm = TRUE)
  res[i,] = c("cindex" = tt$c.index, "cindex.se" = tt$se, "lower" = tt$lower, "upper" = tt$upper)
}
rownames(res)=dn
colnames(res)=c("cindex","cindex.se","lower","upper")

tt <- as.data.frame(res)
labeltext <- toupper(rownames(tt))

r.mean <- c(tt$cindex)
r.lower <- c(tt$lower)
r.upper <- c(tt$upper)

# Overall C-index
ceData <- combine.est(x=res[,"cindex"], x.se=res[,"cindex.se"], hetero=TRUE, na.rm = T)
cLower <- ceData$estimate + qnorm(0.025, lower.tail=TRUE) * ceData$se
cUpper <- ceData$estimate + qnorm(0.025, lower.tail=FALSE) * ceData$se

cindexO <- rbind("cindex"=ceData$estimate, "cindex.se"=ceData$se, "lower"=cLower, "upper"=cUpper)
library(rmeta)
pdf(file = "forestplot.pdf",width = 7, height = 5)
metaplot.surv(mn=r.mean, lower=r.lower, upper=r.upper, labels=labeltext, xlim=c(0.4,0.8),
              summn = ceData$estimate, sumse = ceData$se, sumlower = cLower, sumupper = cUpper, sumnn = 1, summlabel = "Overall",
              boxsize=0.5, zero=0.5, cex = 0.8, ylab = "Dataset", 
              col=meta.colors(box="royalblue",line="darkblue",zero="firebrick"),
              xlab="Concordance index")
dev.off()

pdf(file = "forestplot1.pdf",width = 7, height = 5)
metaplot.surv(mn=r.mean, lower=r.lower, upper=r.upper, labels=sprintf("%.2e", printPV), xlim=c(0.4,0.8),
              summn = ceData$estimate, sumse = ceData$se, sumlower = cLower, sumupper = cUpper, sumnn = 1, summlabel = "Overall",
              boxsize=0.5, zero=0.5, cex = 0.8, ylab = "P-value", 
              col=meta.colors(box="royalblue",line="darkblue",zero="firebrick"),
              xlab="Concordance index")
dev.off()

###compute PValues H0: cindex > 0.5
pv <- apply(res, 1, function(x) { return(pnorm((x[1] - 0.5) / x[2], lower.tail=x[1] < 0.5)) })
printPV <- matrix(pv,ncol=12)
rownames(printPV) <- "P-value"
colnames(printPV) <- names(pv)
printPV<-t(printPV)
## ----printPvalue,results="asis"------------------------------------------
write.csv(printPV,file = "Pvalue.csv",row.names = TRUE)

####----plot forest plot of hazard ratio---------------
res = matrix(data=NA, nrow = length(dn),ncol = 5)
for (i in 1:length(dn)){
  data = read.csv(file = paste0("result/",dn[i],".csv"),header = F)
  group = binarize(data[,3]*(10^11))
  tt <- survcomp::hazard.ratio(x=group, surv.time=data[,1], surv.event=data[,2], na.rm=TRUE);
  tt = survcomp::hazard.ratio(x = group, surv.time = data[,1], surv.event = data[,2], na.rm = TRUE)
  res[i,] = c("hr" = tt$hazard.ratio, "hr.se" = tt$se, "lower" = tt$lower, "upper" = tt$upper, "P_value" = tt$p.value)
}

rownames(res)=dn
colnames(res)=c("hr","hr.se","lower","upper", "P_value")

tt <- as.data.frame(res)
labeltext <- toupper(rownames(tt))

r.mean <- c(tt$cindex)
r.lower <- c(tt$lower)
r.upper <- c(tt$upper)

# Overall C-index
ceData <- combine.est(x=res[,"hr"], x.se=res[,"hr.se"], hetero=TRUE, na.rm = T)
cLower <- ceData$estimate + qnorm(0.025, lower.tail=TRUE) * ceData$se
cUpper <- ceData$estimate + qnorm(0.025, lower.tail=FALSE) * ceData$se

cindexO <- rbind("hr"=ceData$estimate, "hr.se"=ceData$se, "lower"=cLower, "upper"=cUpper)

library(rmeta)
pdf(file = "forestplot_hr.pdf",width = 7, height = 5)
metaplot.surv(mn=r.mean, lower=r.lower, upper=r.upper, labels=labeltext, xlim=c(0.5,max(r.upper)),
              summn = ceData$estimate, sumse = ceData$se, sumlower = cLower, sumupper = cUpper, sumnn = 1, summlabel = "Overall",
              boxsize=0.5, zero=1.0, cex = 0.8, ylab = "Dataset", 
              col=meta.colors(box="royalblue",line="darkblue",zero="firebrick"),
              xlab="Hazard ratio")
dev.off()

pdf(file = "forestplot_hr1.pdf",width = 7, height = 5)
metaplot.surv(mn=r.mean, lower=r.lower, upper=r.upper, labels=sprintf("%.2e", tt$P_value), xlim=c(0.5,max(r.upper)),
              summn = ceData$estimate, sumse = ceData$se, sumlower = cLower, sumupper = cUpper, sumnn = 1, summlabel = "Overall",
              boxsize=0.5, zero=1.0, cex = 0.8, ylab = "P-value", 
              col=meta.colors(box="royalblue",line="darkblue",zero="firebrick"),
              xlab="Hazard ratio")
dev.off()

###compute PValues H0: cindex > 0.5
printPV <- matrix(tt$P_value,ncol=12)
rownames(printPV) <- "P-value"
colnames(printPV) <- names(pv)
printPV<-t(printPV)
## ----printPvalue,results="asis"------------------------------------------
write.csv(printPV,file = "hr_Pvalue.csv",row.names = TRUE)


####----bar plot of P_value of Log-rank test---------------
pvMatrix = read.csv("./result/LogrankPV.csv",stringsAsFactors = F, header =  T)
rownames(pvMatrix) = pvMatrix$X
pvMatrix=pvMatrix[,-1]
colnames(pvMatrix) = toupper(colnames(pvMatrix))

pv = pvMatrix["DGBCox",]
data = t(-log10(pv))
df=data.frame(row.names = colnames(pvMatrix))
df$x = colnames(pvMatrix)
df$x = factor(df$x, level=colnames(pvMatrix))
df$y = round(data, digits = 3)

pdf(file = "lrpv_bar.pdf",width = 7, height = 5)
ggplot(data=df, aes(x=x, y=y)) +
  xlab("Dataset") + 
  ylab(expression(bold('-log'[10]*'(p-value)')))+
  geom_hline(yintercept= -log10(0.05),color = "red",size=0.8, linetype="dashed")+
  geom_bar(stat="identity",width=0.5)+
  theme(
    axis.title.x = element_text(size = 12, face = "bold"),
    axis.text.x = element_text(size = 12, angle =  45, vjust = 0.5, hjust=0.5),
    axis.title.y = element_text(size = 12, face = "bold"),
    axis.text.y = element_text(size = 12, face = "bold"),
    #panel.background = element_blank(),
    #panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line = element_line(colour = "black"),
    panel.border = element_rect(colour = "black", fill=NA, size=1))+
  scale_y_continuous(expand = c(0,0),
                     limits = c(0,8),
                     breaks  = seq(0,8, by = 2))+
  geom_text(aes(label=round(y, digits = 2)), position=position_dodge(width=0.5), vjust=-0.25)

dev.off()


####----plot table of hazard ratio---------------
# methods <- c("AURKA", "ESR1", "ERBB2", "GGI", "GENIUS", "Endopredict", "OncotypeDx",
#              "TAMR13", "PIK3CAGS", "GENE70", "rorS", "DGBCR")
# data = a[methods,]
# colnames(data) = toupper(colnames(data))

data = read.csv("./result/hr.csv",stringsAsFactors = F, header =  T)
rownames(data) = data$X
data=data[,-1]
colnames(data) = toupper(colnames(data))

library(reshape2)
library(ggplot2)
library(ggrepel)

refine = melt(as.matrix(data))
refine$Var1 = as.factor(refine$Var1)
refine$Var2 = as.factor(refine$Var2)
refine$value = round(refine$value, digits = 3)

pdf(file = "hr.pdf",width = 9, height = 7.8)
ggplot(refine, aes(Var2, Var1)) +
  geom_tile(aes(fill=value), color="white")+
  xlab("Dataset") + ylab("Cancer prognosis method") +
  labs(fill = "HR") +
  geom_text(aes(label = value), size=3 ) +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position="bottom",
        plot.title = element_text(color="Black", size=14, face="bold", hjust = 0.5),
        axis.title.x = element_text(color="Black", size=12, face="plain"),
        axis.title.y = element_text(color="Black", size=12, face="plain"),
        axis.text.x = element_text(angle=45, vjust=1, hjust=1))+
  scale_fill_gradient(low = "white", high = "steelblue") +
  coord_equal()

dev.off()

rm = rowMeans(data)
rsd = apply(data,1, sd)
stability = rm-rsd
stability = data.frame(stability)
colnames(stability)= "stability"
rownames(stability) = rownames(data)

df = melt(as.matrix(stability))
df$Var1 = as.factor(df$Var1)
df$Var2 = as.factor(df$Var2)
df$value = round(df$value, digits = 3)

pdf(file = "hr2.pdf",width = 9, height = 7.65)
ggplot(df, aes(Var2, Var1)) +
  geom_tile(aes(fill=value), color="white")+
  xlab("Stability") + 
  labs(fill = expression(Stability["hr"])) +
  geom_text(aes(label = value), size=3 ) +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position="bottom",
        plot.title = element_text(color="Black", size=14, face="bold", hjust = 0.5),
        axis.title.x = element_text(color="Black", size=12, face="plain"),
        axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.text.x = element_text(angle=45, vjust=1, hjust=1))+
  scale_fill_gradient(low = "grey100", high = "grey70") +
  coord_equal()
dev.off()

####----plot figure of Log-rank P-value---------------
data = t(pvMatrix)

library(ggplot2)
library(reshape2)

refine = melt(as.matrix(data))
refine$Var1 = as.factor(refine$Var1)
refine$Var2 = as.factor(refine$Var2)
refine = na.omit(refine)
refine$Rate = as.factor(ifelse(refine$value<=0.05, "<=0.05", ">0.05"))

# refine$value = ifelse(refine$value > 0.05, 0.051, refine$value)
# refine$value = ifelse(refine$value <= 0.05 & refine$value > 0.02, 0.05, refine$value)
# refine$value = ifelse(refine$value <= 0.02 & refine$value > 0.01, 0.02, refine$value)
# refine$value = ifelse(refine$value <= 0.01, 0.01, refine$value)
# head(refine)

pdf(file = "lrpv.pdf",width = 4, height = 4.5)
ggplot(refine, aes(Var1, Var2)) +
  xlab("Dataset") + ylab("Cancer prognosis method")+
  geom_tile(aes(fill=Rate), colour = "white")+
  labs(fill = "P-value") +
  scale_fill_manual(values=c("steelblue", "grey")) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = -0.1),
    legend.position = c("bottom"), # position the legend in the upper left 
    legend.direction = "horizontal",
    legend.title=element_text(size=10),
    legend.text = element_text(size = 9),
    axis.title.x = element_text(size = 12, face = "bold"), 
    axis.title.y= element_text(size = 12, face = "bold"), 
    axis.text.y = element_text(vjust=1, hjust=1), 
    axis.text.x = element_text(angle=90, vjust=1, hjust=1))
dev.off()

# GO and KEGG enrichment analysis

library(DOSE)
library(org.Hs.eg.db)
library(topGO)
library(clusterProfiler)
library(pathview)

top50 = read.csv(file = "top50genes.csv", header = T, stringsAsFactors = F)
rownames(top50) = top50$Gene


test1 = bitr(top50, fromType="SYMBOL", toType=c("ENSEMBL", "ENTREZID"), OrgDb="org.Hs.eg.db")
ego_ALL <- enrichGO(gene = test1$ENTREZID, 
                    OrgDb = org.Hs.eg.db, 
                    ont = "BP",
                    pAdjustMethod = "BH",
                    pvalueCutoff = 0.05, 
                    readable = TRUE) 

pdf("TOp50GO.pdf",width=10,height=6,onefile = FALSE)
barplot(ego_ALL, showCategory=20,title="Enrichment GO")
dev.off()

kk <- enrichKEGG(gene = test1$ENTREZID,
                 organism = 'hsa', 
                 pvalueCutoff = 1)

pdf("Top50KEGG.pdf",width=10,height=6,onefile = FALSE)
barplot(kk, showCategory=20,title="Enrichment KEGG")
dev.off()
dotplot(kk,title="Enrichment KEGG")