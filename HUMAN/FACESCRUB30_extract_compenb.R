source(paste0(getwd() , "/General_R_tools/myPCA.R"))
source(paste0(getwd() , "/General_R_tools/calc_dc.R"))
source(paste0(getwd() , "/General_R_tools/plotCorrAF.R"))
source(paste0(getwd() ,"/General_R_tools/extract_info_ds_HUMAN.R"))


CFD_other =  read.csv("/media/sonia/RENOULT secure/facescrub/download/GenderSelf_miniface.csv")

library(RColorBrewer)
n <- 30
myColors <- colorRampPalette(brewer.pal(8, "Set2"))(n)
#myColors <-grDevices::colors()[grep('gr(a|e)y', grDevices::colors(), invert = T)]


qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))
names(col_vector) <- levels(CFD_other$Id)
colScale <- scale_colour_manual(name = "grp",values = col_vector )

#-------------------------------------------------------------------------------#
#CF non neutral
PATH_to_resdir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/CLASSIFICATION_GENDER/2021-04-15-11-46/"
res_emb <- scale(read.delim(paste0(PATH_to_resdir , "Facescrub30_emb.tsv"), header=FALSE))
CFDori <- read.csv(paste0(PATH_to_resdir , "Facescrub30_label.tsv"), header=FALSE , sep = "\t" )
names(CFDori) <- c("Id" ,"Photoname", "Gender_Pred")
CFD = plyr::join(CFDori,CFD_other) 
dt.acp <- getPCS(res_emb)
plot_PCA(dt.acp , eboulis = FALSE, plot_var = FALSE, corrpl = FALSE)
# Create a ggplot with 18 colors 
# Use scale_fill_manual
CFD$pc1=dt.acp$ind$coord[,1]
CFD$pc2=dt.acp$ind$coord[,2]
contrib1=dt.acp$eig[1,2]
contrib2=dt.acp$eig[2,2]
CG = plot_PCA_ggplot_ID(CFD,colScale, contrib1,contrib2 )
----------------------------------------------------#
#CF non neutral
PATH_to_resdir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/CLASSIFICATION_INDIVIDUAL/2021-04-19-11-10/"
res_emb <- scale(read.delim(paste0(PATH_to_resdir , "Facescrub30_emb.tsv"), header=FALSE))
CFDori <- read.csv(paste0(PATH_to_resdir , "Facescrub30_label.tsv"), header=FALSE , sep = "\t" )
names(CFDori) <- c("Id" ,"Photoname", "Id_Pred")
CFD = plyr::join(CFDori,CFD_other) 
dt.acp <- getPCS(res_emb)
plot_PCA(dt.acp , eboulis = FALSE, plot_var = FALSE, corrpl = FALSE)
# Create a ggplot with 18 colors 
# Use scale_fill_manual
CFD$pc1=dt.acp$ind$coord[,1]
CFD$pc2=dt.acp$ind$coord[,2]
contrib1=dt.acp$eig[1,2]
contrib2=dt.acp$eig[2,2]
CI = plot_PCA_ggplot_ID(CFD,colScale, contrib1,contrib2 )

#----------------------------------------------------#
  #CF non neutral
  PATH_to_resdir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/VERIFICATION_INDIVIDUAL/2021-03-12-8-47/"
res_emb <- scale(read.delim(paste0(PATH_to_resdir , "Facescrub30_emb.tsv"), header=FALSE))
CFDori <- read.csv(paste0(PATH_to_resdir , "Facescrub30_label.tsv"), header=FALSE , sep = "\t" )
names(CFDori) <- c("Id" ,"Photoname", "Id_Pred")
CFD = plyr::join(CFDori,CFD_other) 
dt.acp <- getPCS(res_emb)
plot_PCA(dt.acp , eboulis = FALSE, plot_var = FALSE, corrpl = FALSE)
# Create a ggplot with 18 colors 
# Use scale_fill_manual
CFD$pc1=dt.acp$ind$coord[,1]
CFD$pc2=dt.acp$ind$coord[,2]
contrib1=dt.acp$eig[1,2]
contrib2=dt.acp$eig[2,2]
VI = plot_PCA_ggplot_ID(CFD,colScale, contrib1,contrib2 )
#----------------------------------------------------#
  
#CF non neutral
PATH_to_resdir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/VERIFICATION_INDIVIDUAL/2022-04-25-16-50/"
res_emb <- scale(read.delim(paste0(PATH_to_resdir , "Facescrub30_emb.tsv"), header=FALSE))
CFDori <- read.csv(paste0(PATH_to_resdir , "Facescrub30_label.tsv"), header=FALSE , sep = "\t" )
names(CFDori) <- c("Id" ,"Photoname", "Sex_Pred","Id_Pred")
CFD = plyr::join(CFDori,CFD_other) 
dt.acp <- getPCS(res_emb)
plot_PCA(dt.acp , eboulis = FALSE, plot_var = FALSE, corrpl = FALSE)
# Create a ggplot with 18 colors 
# Use scale_fill_manual
CFD$pc1=dt.acp$ind$coord[,1]
CFD$pc2=dt.acp$ind$coord[,2]
contrib1=dt.acp$eig[1,2]
contrib2=dt.acp$eig[2,2]
VICG = plot_PCA_ggplot_ID(CFD,colScale, contrib1,contrib2)
#----------------------------------------------------#

allfigH = ggarrange(CG,CI,VI,VICG   , ncol=2 , nrow = 2, 
                    labels = c("A", "B", "C","D"))

allfigH_v2 = ggarrange(CG,VI,VICG   , ncol=3 , nrow = 1, 
                    labels = c("A", "B", "C","D"))
ggsave("/home/sonia/Perception_Typicality_CNN/Paper1/Figures/allfigV2facescrub.tiff",allfigH_v2, height = 3.5, width = 9, dpi = 300)