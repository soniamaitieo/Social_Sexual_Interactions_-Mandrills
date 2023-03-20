source(paste0(getwd() , "/General_R_tools/myPCA.R"))
source(paste0(getwd() , "/General_R_tools/calc_dc.R"))
source(paste0(getwd() , "/General_R_tools/plotCorrAF.R"))


plot_PCA_ggplot_ID_MFD<- function(CFD,colScale,contrib1,contrib2){
  ggplot(CFD) + 
    geom_point(aes(pc1, pc2, color = factor(Id_folder), shape = factor(sexe)),size=1) +
    theme_pander() + theme(legend.position = "none") + xlab(paste0("PC1 (",round(contrib1),"%)" )) + ylab(paste0("PC2 (",round(contrib2),"%)" )) + colScale
}

plot_PCA_ggplot_sex_MFD<- function(CFD,colScale,contrib1,contrib2){
  ggplot(CFD) + 
  geom_point(aes(pc1, pc2, color = factor(sexe)),size=1) +
  theme_pander() + theme(legend.position = "none") + xlab(paste0("PC1 (",round(contrib1),"%)" )) + ylab(paste0("PC2 (",round(contrib2),"%)" )) + scale_colour_manual(name = "grp",values = c("#b66dff", "#009999"))
}

###############

MFD <- read.csv("/home/sonia/ProjetMandrillus_Photos/Documents/Metadata/MFD_20220303.csv", row.names=1, stringsAsFactors=FALSE)

MFD <- MFD[,-1]

#------------------
#CG
df_label <- read.csv("/home/sonia/Perception_Typicality_CNN/MANDRILL/results/CLASSIFICATION_GENDER/2022-03-23-11-53/MFD20220303_labelTRAIN.tsv", header=FALSE, sep=";")
names(df_label) <- c("Id_folder" , "Photo_Name" , "Categ_Pred_Ind")
df_emb <- read.delim("/home/sonia/Perception_Typicality_CNN/MANDRILL/results/CLASSIFICATION_GENDER/2022-03-23-11-53/MFD20220303_embTRAIN.tsv", header=FALSE)
df_emb = scale(df_emb)
df_label_m <- left_join(df_label, MFD)

library(RColorBrewer)
col_vector=grDevices::colors()[grep('gr(a|e)y', grDevices::colors(), invert = T)]
names(col_vector) <- levels(MFD$Id_folder)
colScale <- scale_colour_manual(name = "grp",values = col_vector)

dt.acp <- getPCS(df_emb)
df_label_m$pc1=dt.acp$ind$coord[,1]
df_label_m$pc2=dt.acp$ind$coord[,2]
contrib1=dt.acp$eig[1,2]
contrib2=dt.acp$eig[2,2]
CS=plot_PCA_ggplot_ID_MFD(df_label_m,colScale,contrib1,contrib2)
CSsex=plot_PCA_ggplot_sex_MFD(df_label_m,colScale,contrib1,contrib2)
#------------------
#CI
df_label <- read.csv("/home/sonia/Perception_Typicality_CNN/MANDRILL/results/CLASSIFICATION_INDIVIDUAL/2022-04-01-16-28/MFD20220303_labelTRAIN.tsv", header=FALSE, sep=";")
names(df_label) <- c("Id_folder" , "Photo_Name" , "Categ_Pred_Ind")
df_emb <- read.delim("/home/sonia/Perception_Typicality_CNN/MANDRILL/results/CLASSIFICATION_INDIVIDUAL/2022-04-01-16-28/MFD20220303_embTRAIN.tsv", header=FALSE)
df_emb = scale(df_emb)
df_label_m <- left_join(df_label, MFD)

library(RColorBrewer)
col_vector=grDevices::colors()[grep('gr(a|e)y', grDevices::colors(), invert = T)]
names(col_vector) <- levels(MFD$Id_folder)
colScale <- scale_colour_manual(name = "grp",values = col_vector)

dt.acp <- getPCS(df_emb)
df_label_m$pc1=dt.acp$ind$coord[,1]
df_label_m$pc2=dt.acp$ind$coord[,2]
contrib1=dt.acp$eig[1,2]
contrib2=dt.acp$eig[2,2]
CI=plot_PCA_ggplot_ID_MFD(df_label_m,colScale,contrib1,contrib2)
CIsex=plot_PCA_ggplot_sex_MFD(df_label_m,colScale,contrib1,contrib2)


#------------------
#VI
df_label <- read.csv("/home/sonia/Perception_Typicality_CNN/MANDRILL/results/VERIFICATION_INDIVIDUAL/2022-03-31-15-46/MFD20220303_labelTRAIN.tsv", header=FALSE, sep=";")
names(df_label) <- c("Id_folder" , "Photo_Name" , "Categ_Pred_Ind")
df_emb <- read.delim("/home/sonia/Perception_Typicality_CNN/MANDRILL/results/VERIFICATION_INDIVIDUAL/2022-03-31-15-46/MFD20220303_embTRAIN.tsv", header=FALSE)
df_emb = scale(df_emb)
df_label_m <- left_join(df_label, MFD)

library(RColorBrewer)
col_vector=grDevices::colors()[grep('gr(a|e)y', grDevices::colors(), invert = T)]
names(col_vector) <- levels(MFD$Id_folder)
colScale <- scale_colour_manual(name = "grp",values = col_vector)

dt.acp <- getPCS(df_emb)
df_label_m$pc1=dt.acp$ind$coord[,1]
df_label_m$pc2=dt.acp$ind$coord[,2]
contrib1=dt.acp$eig[1,2]
contrib2=dt.acp$eig[2,2]
VI=plot_PCA_ggplot_ID_MFD(df_label_m,colScale,contrib1,contrib2)
VIsex=plot_PCA_ggplot_sex_MFD(df_label_m,colScale,contrib1,contrib2)



#------------------
#VICS
#Attention le dossier a été placé dans “CLASSIFICATION_INDIVIDUAL”:~/Perception_Typicality_CNN/MANDRILL/results/CLASSIFICATION_INDIVIDUAL/2022-04-15-19-35
df_label <- read.csv("/home/sonia/Perception_Typicality_CNN/MANDRILL/results/CLASSIFICATION_INDIVIDUAL/2022-04-15-19-35/MFD20220303_labelTRAIN.tsv", header=FALSE, sep=";")
names(df_label) <- c("Id_folder" , "Photo_Name" , "Categ_Pred_Ind","Categ_Pred_Sex", "Categ_True_Ind","Categ_True_Sex" , "Proba_F" , "Proba_M" )
df_emb <- read.delim("/home/sonia/Perception_Typicality_CNN/MANDRILL/results/CLASSIFICATION_INDIVIDUAL/2022-04-15-19-35/MFD20220303_embTRAIN.tsv", header=FALSE)
df_emb = scale(df_emb)
df_label_m <- left_join(df_label, MFD)

library(RColorBrewer)
col_vector=grDevices::colors()[grep('gr(a|e)y', grDevices::colors(), invert = T)]
names(col_vector) <- levels(MFD$Id_folder)
colScale <- scale_colour_manual(name = "grp",values = col_vector)

dt.acp <- getPCS(df_emb)
df_label_m$pc1=dt.acp$ind$coord[,1]
df_label_m$pc2=dt.acp$ind$coord[,2]
contrib1=dt.acp$eig[1,2]
contrib2=dt.acp$eig[2,2]
VICS=plot_PCA_ggplot_ID_MFD(df_label_m,colScale,contrib1,contrib2)
VICSsex=plot_PCA_ggplot_sex_MFD(df_label_m,colScale,contrib1,contrib2)

#----------------------------------------------------#

allfigM = ggarrange(CS,CI,VI,VICS   , ncol=2 , nrow = 2, 
                    labels = c("A", "B", "C","D"))

allfigMsex = ggarrange(CSsex,CIsex,VIsex,VICSsex   , ncol=2 , nrow = 2, 
                    labels = c("A", "B", "C","D"))

allfigV2 = ggarrange(CS,VI,VICS,CSsex,VIsex,VICSsex, ncol=3 , nrow = 2,
                     labels = c("A", "B", "C","D","E","F"))

pdf("allfigV2.pdf",         # File name
    width = 8, height = 7, # Width and height in inches
    bg = "white",          # Background color
    colormodel = "cmyk",    # Color model (cmyk is required for most publications)
    paper = "A4",# Paper size
    res = 300)        

ggsave("/home/sonia/Perception_Typicality_CNN/Paper1/Figures/allfigV2.pdf",allfigV2, height = 5, width = 7, dpi = 300, Family = "Latin")