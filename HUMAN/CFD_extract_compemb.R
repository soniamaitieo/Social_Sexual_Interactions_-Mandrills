setwd("~/Social_Sexual_Interactions_ Mandrills")

source(paste0(getwd() , "/General_R_tools/myPCA.R"))
source(paste0(getwd() , "/General_R_tools/calc_dc.R"))
source(paste0(getwd() , "/General_R_tools/plotCorrAF.R"))
source(paste0(getwd() ,"/General_R_tools/extract_info_ds_HUMAN.R"))

#-------------------------------------------------------------------------------#
PATH_to_resdir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/VERIFICATION_INDIVIDUAL/2022-04-25-16-50/"
CFD_other =  read.csv(paste0(PATH_to_resdir , "CFD_typicality_measures.csv"), row.names=1)[,1:9]
#CF non neutral
PATH_to_resdir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/CLASSIFICATION_GENDER/2021-04-15-11-46/"
res_emb <- scale(read.delim(paste0(PATH_to_resdir , "CFD_US_emb.tsv"), header=FALSE))
CFDori <- read.csv(paste0(PATH_to_resdir , "CFD_US_label.tsv"), header=FALSE , sep = "\t" )
names(CFDori) <- c("Model" ,"Photoname", "Gender_Pred")
CFD = plyr::join(CFDori,CFD_other) 
dt.acp <- getPCS(res_emb)
plot_PCA(dt.acp , eboulis = FALSE, plot_var = FALSE, corrpl = FALSE)
CS_A = plot_PCA_indiv_colorindsex_CFDmulti(dt.acp, CFD$Model, CFD$Gender_Pred,CFD)
CS_B =plot_PCA_indiv_colorindsex_CFDmulti50(dt.acp, indivcol = CFD$Model, gendersexcol = CFD$Gender_Pred,CFD)

#-------------------------------------------------------------------------------#
PATH_to_resdir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/VERIFICATION_INDIVIDUAL/2022-04-25-16-50/"
CFD_other =  read.csv(paste0(PATH_to_resdir , "CFD_typicality_measures.csv"), row.names=1)[,1:9]

#CF non neutral
PATH_to_resdir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/CLASSIFICATION_INDIVIDUAL/2021-04-19-11-10/"
res_emb <- scale(read.delim(paste0(PATH_to_resdir , "CFD_US_emb.tsv"), header=FALSE))
CFDori <- read.csv(paste0(PATH_to_resdir , "CFD_US_label.tsv"), header=FALSE , sep = "\t" )
names(CFDori) <- c("Model" ,"Photoname", "Model_Pred")
CFD = plyr::join(CFDori,CFD_other) 
dt.acp <- getPCS(res_emb)
plot_PCA(dt.acp , eboulis = FALSE, plot_var = FALSE, corrpl = FALSE)
CI_A = plot_PCA_indiv_colorindsex_CFDmulti(dt.acp, CFD$Model, CFD$GenderSelf,CFD)
CI_B =plot_PCA_indiv_colorindsex_CFDmulti50(dt.acp, indivcol = CFD$Model, gendersexcol = CFD$GenderSelf,CFD)

#-------------------------------------------------------------------------------#
PATH_to_resdir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/VERIFICATION_INDIVIDUAL/2022-04-25-16-50/"
CFD_other =  read.csv(paste0(PATH_to_resdir , "CFD_typicality_measures.csv"), row.names=1)[,1:9]

#CF non neutral
PATH_to_resdir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/VERIFICATION_INDIVIDUAL/2021-03-12-8-47/"
res_emb <- scale(read.delim(paste0(PATH_to_resdir , "CFD_US_emb.tsv"), header=FALSE))
CFDori <- read.csv(paste0(PATH_to_resdir , "CFD_US_label.tsv"), header=FALSE , sep = "\t" )
names(CFDori) <- c("Model" ,"Photoname", "Model_Pred")
CFD = plyr::join(CFDori,CFD_other) 
dt.acp <- getPCS(res_emb)
plot_PCA(dt.acp , eboulis = FALSE, plot_var = FALSE, corrpl = FALSE)
VI_A = plot_PCA_indiv_colorindsex_CFDmulti(dt.acp, CFD$Model, CFD$GenderSelf,CFD)
VI_B =plot_PCA_indiv_colorindsex_CFDmulti50(dt.acp, indivcol = CFD$Model, gendersexcol = CFD$GenderSelf,CFD)

#-------------------------------------------------------------------------------#
PATH_to_resdir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/VERIFICATION_INDIVIDUAL/2022-04-25-16-50/"
CFD_other =  read.csv(paste0(PATH_to_resdir , "CFD_typicality_measures.csv"), row.names=1)[,1:9]

#CF non neutral
PATH_to_resdir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/VERIFICATION_INDIVIDUAL/2022-04-25-16-50/"
res_emb <- scale(read.delim(paste0(PATH_to_resdir , "CFD_US_emb.tsv"), header=FALSE))
CFDori <- read.csv(paste0(PATH_to_resdir , "CFD_US_label.tsv"), header=FALSE , sep = "\t" )
names(CFDori) <- c("Model" ,"Photoname", "Model_Pred")
CFD = plyr::join(CFDori,CFD_other) 
dt.acp <- getPCS(res_emb)
plot_PCA(dt.acp , eboulis = FALSE, plot_var = FALSE, corrpl = FALSE)
CSVI_A = plot_PCA_indiv_colorindsex_CFDmulti(dt.acp, CFD$Model, CFD$GenderSelf,CFD)
CSVI_B =plot_PCA_indiv_colorindsex_CFDmulti50(dt.acp, indivcol = CFD$Model, gendersexcol = CFD$GenderSelf,CFD)

CFD$pc1=dt.acp$ind$coord[,1]
CFD$pc2=dt.acp$ind$coord[,2]
contrib1=dt.acp$eig[1,2]
contrib2=dt.acp$eig[2,2]

#

ggplot(CFD) + 
  geom_point(aes(pc1, pc2, color = factor(GenderSelf)),size=2) +
  theme_pander() + theme() + xlab(paste0("PC1 (",round(contrib1),"%)" )) + ylab(paste0("PC2 (",round(contrib2),"%)" )) + colScale



