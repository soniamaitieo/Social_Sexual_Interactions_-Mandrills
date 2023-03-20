PATH_to_resdir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/VERIFICATION_INDIVIDUAL/2022-04-25-16-50/"
source(paste0(getwd() , "/General_R_tools/myPCA.R"))
source(paste0(getwd() , "/General_R_tools/calc_dc.R"))
source(paste0(getwd() , "/General_R_tools/plotCorrAF.R"))
source(paste0(getwd() ,"/General_R_tools/extract_info_ds_HUMAN.R"))


#OFFICIAL VI-CS
res_emb <- scale(read.delim(paste0(PATH_to_resdir , "CFD_emb.tsv"), header=FALSE))
CFD <- read.csv(paste0(PATH_to_resdir , "CFD_typicality_measures.csv"), row.names=1)


################################################################################

CFD$GenderSelf[which(CFD$GenderSelf == "F ")] = as.factor("F")
CFD$Ethn = substr(CFD$Model,1,1)
res_emb = res_emb[-which(CFD$Model == "IM-719-221"),]
CFD = CFD[-which(CFD$Model == "IM-719-221"),]
res_emb = res_emb[-which(is.na(CFD$Feminine)),]
CFD = CFD[-which(is.na(CFD$Feminine)),]

dt.acp <- getPCS(res_emb)

plot_PCA(dt.acp , eboulis = FALSE, plot_var = FALSE, corrpl = FALSE)


acp_ind_gender = plot_PCA_indiv_colorsex(dt.acp, CFD$GenderSelf)
#acp_ind_modelgender_sub = plot_PCA_indiv_colorindsex_top50(dt.acp, CFD$Model, CFD$GenderSelf)


#PC1
CFD$PC1 = getPC1(dt.acp)

#centroid
cf = calc_centroid_PC1(CFD, "GenderSelf","F" ,"PC1")
cm = calc_centroid_PC1(CFD, "GenderSelf","M" ,"PC1")

#calcul de la dist au centroide
CFD$dcmPC1 = dist_eachrow_vec(as.matrix(CFD$PC1),c(cm))
CFD$dcfPC1 = dist_eachrow_vec(as.matrix(CFD$PC1),c(cf))

allvar <- c('dist_centre_F','dist_centre_H' , 'dcmPC1', 'dcfPC1')
#AFvar <- c("Mean_YM_attr", "Mean_YF_attr","Mean_MM_attr","Mean_MF_attr","Mean_OM_attr","Mean_OF_attr", "Mean_attr")
AFvar  = c("Feminine", "Attractive")

Lplot = plotcorr_Attr_dcm_inL(CFD,"GenderSelf" ,allvar , AFvar)

#ggarrange(Lplot$dist_centre_H.Mean_attr, Lplot$dist_centre_F.Mean_attr, Lplot$dcmPC1.Mean_attr , Lplot$dcfPC1.Mean_attr , nrow = 2, ncol=2 )

#get R2
R2_Attract_f = summary(lm(Attractive ~ dcmPC1, data = CFD[CFD$GenderSelf == "F",] ))$r.squared
R2_Fem_f =  summary(lm(Feminine ~ dcmPC1, data = CFD[CFD$GenderSelf == "F",] ))$r.squared

CFD_ds = extract_info_ds(CFD, dt.acp)


###############


distcentroids_ethnic <- function(ethnie,listvide){
  CFD_ethnic =  CFD[CFD$Ethn == ethnie, ]
  res_emb_ethnic =  res_emb[CFD$Ethn == ethnie, ]
  dt.acp <- getPCS(res_emb_ethnic)
  #plot_PCA(dt.acp_ethnic , eboulis = FALSE, plot_var = FALSE, corrpl = FALSE)
  acp_ind_gender_ethnic = plot_PCA_indiv_colorsex(dt.acp, CFD_ethnic$GenderSelf)
  #centroid
  cf = calc_centroid_PC1(CFD_ethnic, "GenderSelf","F" ,"PC1")
  cm = calc_centroid_PC1(CFD_ethnic, "GenderSelf","M" ,"PC1")
  #calcul de la dist au centroide
  CFD_ethnic$dcmPC1 = dist_eachrow_vec(as.matrix(CFD_ethnic$PC1),c(cm))
  CFD_ethnic$dcfPC1 = dist_eachrow_vec(as.matrix(CFD_ethnic$PC1),c(cf))
  Lplot = plotcorr_Attr_dcm_inL(CFD_ethnic,"GenderSelf" ,allvar , AFvar)
  listvide = summary(lm(Attractive ~ dcmPC1, data = CFD_ethnic[CFD_ethnic$GenderSelf == "F",] ))$r.squared
  dimor = abs(cf-cm)
  machou = machouille(CFD_ethnic,"GenderSelf" ,allvar , AFvar)
  LplotR2dimor = list(Lplot = Lplot,R2 = listvide, dimorphisme = dimor, machouille=machou)
  return(LplotR2dimor)}

distcentroids_ethnic_FEM <- function(ethnie,listvide){
  CFD_ethnic =  CFD[CFD$Ethn == ethnie, ]
  res_emb_ethnic =  res_emb[CFD$Ethn == ethnie, ]
  dt.acp <- getPCS(res_emb_ethnic)
  #plot_PCA(dt.acp_ethnic , eboulis = FALSE, plot_var = FALSE, corrpl = FALSE)
  acp_ind_gender_ethnic = plot_PCA_indiv_colorsex(dt.acp, CFD_ethnic$GenderSelf)
  #centroid
  cf = calc_centroid_PC1(CFD_ethnic, "GenderSelf","F" ,"PC1")
  cm = calc_centroid_PC1(CFD_ethnic, "GenderSelf","M" ,"PC1")
  #calcul de la dist au centroide
  CFD_ethnic$dcmPC1 = dist_eachrow_vec(as.matrix(CFD_ethnic$PC1),c(cm))
  CFD_ethnic$dcfPC1 = dist_eachrow_vec(as.matrix(CFD_ethnic$PC1),c(cf))
  Lplot = plotcorr_Attr_dcm_inL_femethn(CFD_ethnic,"GenderSelf" ,allvar , AFvar)
  listvide = summary(lm(Attractive ~ dcmPC1, data = CFD_ethnic[CFD_ethnic$GenderSelf == "F",] ))$r.squared
  dimor = abs(cf-cm)
  machou = machouille(CFD_ethnic,"GenderSelf" ,allvar , AFvar)
  LplotR2dimor = list(Lplot = Lplot,R2 = listvide, dimorphisme = dimor, machouille=machou)
  return(LplotR2dimor)}


all_ethn = unique(CFD$Ethn)

LLol = lapply(all_ethn,distcentroids_ethnic)
names(LLol) = all_ethn

tableau=cbind(unlist(lapply(X = LLol, FUN = `[[`,'R2')),unlist(lapply(X = LLol, FUN = `[[`,'dimorphisme')))

############## FIGURE PAPERS

CFDsansMRI = CFD[-which(CFD$Ethn == "M"),]
CFDsansMRI = CFDsansMRI [-which(CFDsansMRI$Ethn == "I"),]

  
  
LplotBIS = plotcorr_Attr_dcm_inL_femethn(CFDsansMRI,"GenderSelf" ,allvar , AFvar)

LLolBIS = lapply(all_ethn,distcentroids_ethnic_FEM)
names(LLolBIS) = all_ethn

library(grid)

A = LplotBIS$dcmPC1.Feminine +  ylab("Perceived Femininity") + xlab("Femaleness (dcm)")+    ylim(2,6) + annotation_custom(grobTree(textGrob("All", x=0.83,  y=0.93, hjust=0,
                                                                                                                                    gp=gpar(col="#9c2bff", fontsize=15,family="Times",fill = "blue")))) +theme(text=element_text( family="Times"))

B = LplotBIS$dcmPC1.Attractive +   ylab("Perceived Attractivness") + xlab("Femaleness (dcm)") +  ylim(1,6) + annotation_custom(grobTree(textGrob("All", x=0.83,  y=0.93, hjust=0,
                                                                                                                                           gp=gpar(col="#9c2bff", fontsize=15,family="Times")))) +theme(text=element_text( family="Times"))

C = LLolBIS$B$Lplot$dcmPC1.Feminine +   ylab("Perceived Femininity") + xlab("Femaleness (dcm)") +  ylim(2,6) + annotation_custom(grobTree(textGrob("Black", x=0.83,  y=0.93, hjust=0,
                                                                                                                                             gp=gpar(col="#9c2bff", fontsize=15,family="Times"))))  +theme(text=element_text( family="Times"))

D = LLolBIS$A$Lplot$dcmPC1.Feminine +   ylab("Perceived Femininity") + xlab("Femaleness (dcm)") +  ylim(2,6) + annotation_custom(grobTree(textGrob("Asian", x=0.83,  y=0.93, hjust=0,
                                                                                                                                                                    gp=gpar(col="#9c2bff", fontsize=15,family="Times"))))  +theme(text=element_text( family="Times"))
E =  LLolBIS$L$Lplot$dcmPC1.Feminine +   ylab("Perceived Femininity") +xlab("Femaleness (dcm)") +  ylim(2,6) + annotation_custom(grobTree(textGrob("Latino", x=0.83,  y=0.93, hjust=0,
                                                                                                                                                                     gp=gpar(col="#9c2bff",  fontsize=15,family="Times"))))  +theme(text=element_text( family="Times"))
G = LLolBIS$W$Lplot$dcmPC1.Feminine +   ylab("Perceived Femininity") +xlab("Femaleness (dcm)") +  ylim(2,6) + annotation_custom(grobTree(textGrob("White", x=0.83,  y=0.93, hjust=0,
                                                                                                                                                                    gp=gpar(col="#9c2bff", fontsize=15,family="Times")))) +theme(text=element_text( family="Times"))


allfigH = ggarrange(A,B,C,D,E,G   , ncol=3 , nrow = 2, 
          labels = c("A", "B", "C","D","E","F"), common.legend = TRUE)

annotate_figure(allfigH, 
               bottom = textGrob("Predicted Femininity (dcm)",gp = gpar(cex = 1.3, fontsize=18,family="Times")))

