PATH_to_resdir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/VERIFICATION_INDIVIDUAL/2022-04-25-16-50/"
source(paste0(getwd() , "/General_R_tools/myPCA.R"))
source(paste0(getwd() , "/General_R_tools/calc_dc.R"))
source(paste0(getwd() , "/General_R_tools/plotCorrAF.R"))
source(paste0(getwd() , "/General_R_tools/extract_info_ds_HUMAN.R"))

allvar <- c('dist_centre_F','dist_centre_H' , 'dcmPC1', 'dcfPC1')
#AFvar <- c("Mean_YM_attr", "Mean_YF_attr","Mean_MM_attr","Mean_MF_attr","Mean_OM_attr","Mean_OF_attr", "Mean_attr")
AFvar  = c( "Attractive")

res_emb <- scale(read.delim(paste0(PATH_to_resdir , "SCUT_emb.tsv"), header=FALSE))
SCUT <- read.csv(paste0(PATH_to_resdir , "SCUT_typicality_measures.csv"), row.names=1)
SCUT$Ethn = substr(SCUT$Model,1,1)

res_emb_A = res_emb[which(SCUT$Ethn == "A"),]
SCUT_A = SCUT %>% filter(Ethn == "A")
res_emb_C = res_emb[which(SCUT$Ethn == "C"),]
SCUT_C = SCUT %>% filter(Ethn == "C")

#A
dt.acp_A <- getPCS(res_emb_A)
plot_PCA(dt.acp_A , eboulis = FALSE, plot_var = FALSE, corrpl = FALSE)
acp_ind_gender_A = plot_PCA_indiv_colorsex(dt.acp_A, SCUT_A$GenderSelf)
#PC1
SCUT_A$PC1 = getPC1(dt.acp_A)
#centroid
cf = calc_centroid_PC1(SCUT_A, "GenderSelf","F" ,"PC1")
cm = calc_centroid_PC1(SCUT_A, "GenderSelf","M" ,"PC1")
#calcul de la dist au centroide
SCUT_A$dcmPC1 = dist_eachrow_vec(as.matrix(SCUT_A$PC1),c(cm))
SCUT_A$dcfPC1 = dist_eachrow_vec(as.matrix(SCUT_A$PC1),c(cf))

Lplot_A = plotcorr_Attr_dcm_inL(SCUT_A,"GenderSelf" ,allvar , AFvar)

################################################################################
#C
dt.acp_C <- getPCS(res_emb_C)
plot_PCA(dt.acp_C , eboulis = FALSE, plot_var = FALSE, corrpl = FALSE)
acp_ind_gender_C = plot_PCA_indiv_colorsex(dt.acp_C, SCUT_C$GenderSelf)
#PC1
SCUT_C$PC1 = getPC1(dt.acp_C)
#centroid
cf = calc_centroid_PC1(SCUT_C, "GenderSelf","F" ,"PC1")
cm = calc_centroid_PC1(SCUT_C, "GenderSelf","M" ,"PC1")
#calcul de la dist au centroide
SCUT_C$dcmPC1 = dist_eachrow_vec(as.matrix(SCUT_C$PC1),c(cm))
SCUT_C$dcfPC1 = dist_eachrow_vec(as.matrix(SCUT_C$PC1),c(cf))

Lplot_C = plotcorr_Attr_dcm_inL(SCUT_C,"GenderSelf" ,allvar , AFvar)

# R2
R2_Attract_f = summary(lm(Attractive ~ dcmPC1, data = SCUT_A[SCUT_A$GenderSelf == "F",] ))$r.squared
SCUTa_ds = extract_info_ds(SCUT_A, dt.acp_A)

R2_Attract_f = summary(lm(Attractive ~ dcmPC1, data = SCUT_C[SCUT_C$GenderSelf == "F",] ))$r.squared
SCUTc_ds = extract_info_ds(SCUT_C, dt.acp_C)



#-------------------

scutALL=do.call(rbind,machouille(SCUT,"GenderSelf" ,allvar , AFvar))
scutA=do.call(rbind,machouille(SCUT_A,"GenderSelf" ,allvar , AFvar))
scutW=do.call(rbind,machouille(SCUT_C,"GenderSelf" ,allvar , AFvar))
r4all = cbind(All=scutALL,Asian=scutA,White=scutW)
r4all = signif(r4all,2)

#write.csv(r4all,"/home/sonia/Perception_Typicality_CNN/Report_memoiR/MatMetHUMAN/out/SCUT_rp_all_a_c.csv")
