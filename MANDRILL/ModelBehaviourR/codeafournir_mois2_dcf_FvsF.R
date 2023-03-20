
library(ggplot2)
library(spaMM)
library(ggthemes)
library(sjPlot)

df_withBEH_FF =readRDS("/home/sonia/Social_Sexual_Interactions_ Mandrills/MANDRILL/ModelBehaviourR/Rdata_behavior_femininity/list_df_withBEH_month_FvsF.RData")

dfall= df_withBEH_FF$dfQ2Q3YSR
dfall=dfall[-which(is.na(dfall$CycleCateg )),]
dfall$CycleCateg[dfall$CycleCateg == "CycleBB"] <- "Cycle"


dfall$age=scale(dfall$age)
dfall$Mean_dcM=scale(dfall$Mean_dcM)

dfall$AGall_Nb <- dfall$AGmild_Nb + dfall$AGsev_Nb
dfall$AGall_Nb <- dfall$AGmild_Nb + dfall$AGsev_Nb

dfall$AGall_Nb_SameSex <- dfall$AGmild_Nb_SameSex + dfall$AGsev_Nb_SameSex


dfrepro = dfall %>% filter(saison_repro == "Repro")
dfnorepro = dfall %>% filter(saison_repro == "Naiss")

applySPAMM <- function(formula,df){
  spfitMO <- spaMM::fitme(formula = formula(formula), data = df, family = negbin, method=c("ML","obs"))
  forrates <- df
  forrates$Tps_Foc_IndAllmal <- 1L
  forrates$Nb_Scan_IndAllmal <- 1L
  CycleF <- CycleT <- forrates
  CycleF$CycleCateg <- "no"
  CycleT$CycleBin <- "yes"
  pdep_F <- pdep_effects(spfitMO,"Mean_dcM", newdata=as.data.frame(CycleF), length.out = 500)
  pdep_T <- pdep_effects(spfitMO,"Mean_dcM", newdata=as.data.frame(CycleT), length.out = 500)
  pdep_F$CycleBin <- "no"
  pdep_T$CycleBin <- "yes"
  pdep <- rbind(pdep_F,pdep_T)
  return(list(mymodel=spfitMO, pdep=pdep))
}

applySPAMM_onevar <- function(formula,df){
  spfitMO <- spaMM::fitme(formula = formula(formula), data = df, family = negbin, method=c("ML","obs"))
  forrates <- df
  forrates$Tps_Foc_IndAllmal <- 1L
  forrates$Nb_Scan_IndAllmal <- 1L
  pdep <- pdep_effects(spfitMO,"Mean_dcM", newdata=as.data.frame(forrates), length.out = 500)
  return(list(mymodel=spfitMO, pdep=pdep))
}

applySPAMM_2var <- function(formula,df){
  spfitMO <- spaMM::fitme(formula = formula(formula), data = df, family = negbin, method=c("ML","obs"))
  forrates <- df
  forrates$Tps_Foc_IndAllmal <- 1L
  forrates$Nb_Scan_IndAllmal <- 1L
  Cycle <- CycleNO <- forrates
  CycleNO$CycleCateg <- "CycleNO"
  Cycle$CycleCateg <- "Cycle"
  pdep_NO <- pdep_effects(spfitMO,"Mean_dcM", newdata=as.data.frame(CycleNO), length.out = 500)
  pdep <- pdep_effects(spfitMO,"Mean_dcM", newdata=as.data.frame(Cycle), length.out = 500)
  pdep_NO$CycleCateg <- "CycleNO"
  pdep$CycleCateg <- "Cycle"
  pdepall <- rbind(pdep_NO,pdep)
  return(list(mymodel=spfitMO, pdep=pdepall))
}


#setwd("/home/sonia/Perception_Typicality_CNN/Paper1/Figures/forestplot_month_2_dcf_Fem/")

################################################################################
#TO#
################################################################################
TOformula = "TO_Nb_SameSex ~ (1 | id) + (1 | year) + Mean_dcM + age +  
    rang + CycleCateg +  saison_repro +offset(log(Tps_Foc_IndAllfem))"
spfitTO = applySPAMM_2var(TOformula,dfall)
summary(spfitTO$mymodel,details=c(p_value="Wald"))
plot_model(spfitTO$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "") + 
  theme(face = "bold") + theme_light(base_size=16)



#TO2
TOformula = "TO_Nb_SameSex ~ (1 | id) + (1 | year) + Mean_dcM + age +  
    rang + CycleCateg + offset(log(Tps_Foc_IndAllfem))"
spfitTO2 = applySPAMM_onevar(TOformula,dfrepro)
summary(spfitTO2$mymodel,details=c(p_value="Wald"))
plot_model(spfitTO2$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "") + 
  theme(face = "bold") + theme_light(base_size=16)


#TO3
AGformula = "TO_Nb_SameSex ~ (1 | id) + (1 | year) + Mean_dcM + age +  
    rang + CycleCateg +  offset(logTps_Foc_IndAllfem))"
spfitTO3 = applySPAMM_onevar(TOformula,dfnorepro)
summary(spfitTO3$mymodel,details=c(p_value="Wald"))
plot_model(spfitTO3$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "") + 
  theme(face = "bold") + theme_light(base_size=16)

################################################################################
#AG#
################################################################################
AGformula = "AGall_Nb_SameSex ~ (1 | id) + (1 | year) + Mean_dcM + age +  
    rang + CycleCateg +  saison_repro+  +offset(log(Tps_Foc_IndAllfem))"
spfitAG = applySPAMM_onevar(AGformula,dfall)
summary(spfitAG$mymodel,details=c(p_value="Wald"))
plot_model(spfitAG$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "") + 
  theme(face = "bold") + theme_light(base_size=16)

plot_model(spfitAG$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "",axis.labels = rev(c(
  "Femininity",  "Age" , "Rank [Low]" ,"Rank [Medium]" , "Cycle [No]" , "Season [Breeding]"))) + 
  theme(face = "bold") + theme_clean(base_size=20, base_family = "Times")
#--> ici linteraction pas signif mais dcm et cyclecateg separe ouiii
save_plot(file="AG_all.svg",fig= last_plot(),width = 11,height = 10)
ggplot(spfitAG$pdep,aes(y = pointp , x = focal_var )) + geom_point() +
  geom_ribbon(aes(ymin = low, ymax = up), alpha = 0.3) +
  xlab(expression("Femininity")) + ylab("Rates of Agression") + theme_clean(base_size=20, base_family = "Times") +
  theme(legend.key.size = unit(0.2, 'cm'),legend.position = c(.7, .85), legend.title = element_text(size=20))
save_plot(file="AG_all_pred.svg",fig= last_plot() ,width = 9.5,height = 10)


#AG2
AGformula = "AGall_Nb_SameSex ~ (1 | id) + (1 | year) + Mean_dcM + age +  
    rang + CycleCateg + offset(log(Tps_Foc_IndAllfem))"
spfitAG2 = applySPAMM_onevar(AGformula,dfrepro)
summary(spfitAG2$mymodel,details=c(p_value="Wald"))
plot_model(spfitAG2$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "") + 
  theme(face = "bold") + theme_light(base_size=16)


plot_model(spfitAG2$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "",axis.labels = rev(c(
  "Femininity",  "Age" , "Rank [Low]" ,"Rank [Medium]" , "Cycle [No]"))) + 
  theme(face = "bold") + theme_clean(base_size=20, base_family = "Times")
#--> ici linteraction pas signif mais dcm et cyclecateg separe ouiii
save_plot(file="AG_SR.svg",fig= last_plot(),width = 11,height = 10)
ggplot(spfitAG2$pdep,aes(y = pointp , x = focal_var )) + geom_point() +
  geom_ribbon(aes(ymin = low, ymax = up), alpha = 0.3) +
  xlab(expression("Femininity")) + ylab("Rates of Agression") + theme_clean(base_size=16) +
  theme(legend.key.size = unit(0.2, 'cm'),legend.position = c(.7, .85), legend.title = element_text(size=20))
save_plot(file="AG_SR_pred.svg",fig= last_plot() ,width = 9.5,height = 10)

#AG3
AGformula = "AGall_Nb_SameSex ~ (1 | id) + (1 | year) + Mean_dcM + age +  
    rang + CycleCateg +  offset(log(Tps_Foc_IndAllfem))"
spfitAG3 = applySPAMM_onevar(AGformula,dfnorepro)
summary(spfitAG3$mymodel,details=c(p_value="Wald"))
plot_model(spfitAG3$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "") + 
  theme(face = "bold") + theme_light(base_size=16)
ggplot(spfitAG3$pdep,aes(y = pointp , x = focal_var)) + geom_point() +
  geom_ribbon(aes(ymin = low, ymax = up), alpha = 0.3) +
  xlab(expression("Femininity (dcm)")) + ylab("Rates of Agression") + theme_classic(base_size=16) 
#agressent les moins femininies en Saison de Naissance


plot_model(spfitAG3$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "",axis.labels = rev(c(
  "Femininity",  "Age" , "Rank [Low]" ,"Rank [Medium]" , "Cycle [No]"))) + 
  theme(face = "bold") + theme_clean(base_size=20, base_family = "Times")
#--> ici linteraction pas signif mais dcm et cyclecateg separe ouiii
save_plot(file="AG_noSR.svg",fig= last_plot(),width = 11,height = 10)

ggplot(spfitAG3$pdep,aes(y = pointp , x = focal_var )) + geom_point() +
  geom_ribbon(aes(ymin = low, ymax = up), alpha = 0.3) +
  xlab(expression("Femininity")) + ylab("Rates of Agression") + theme_clean(base_size=16) +
  theme(legend.key.size = unit(0.2, 'cm'),legend.position = c(.7, .85), legend.title = element_text(size=20))
save_plot(file="AG_noSR_pred.svg",fig= last_plot() ,width = 9.5,height = 10)

# scan

# SCAN1
SCANformula = "SCAN_Nb_SexSame ~ (1 | id) + (1 | year) + Mean_dcM + age +  
    rang + CycleCateg  + saison_repro +offset(log(Nb_Scan_IndAllfem))"
spfitSCAN = applySPAMM_onevar(SCANformula,dfall)
summary(spfitSCAN$mymodel,details=c(p_value="Wald"))
plot_model(spfitSCAN$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "") + 
  theme(face = "bold") + theme_light(base_size=16)
#--> 
ggplot(spfitSCAN$pdep,aes(y = pointp , x = focal_var)) + geom_point() +
  geom_ribbon(aes(ymin = low, ymax = up), alpha = 0.3) +
  xlab(expression("Femininity (dcm)")) + ylab("Rates of Spatial Proximity") + theme_classic(base_size=20, base_family = "Times") 

plot_model(spfitSCAN$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "",axis.labels = rev(c(
  "Femininity",  "Age" , "Rank [Low]" ,"Rank [Medium]" , "Cycle [No]", "Season [Breeding]"))) + 
  theme(face = "bold") + theme_clean(base_size=20, base_family = "Times")
#--> ici linteraction pas signif mais dcm et cyclecateg separe ouiii
save_plot(file="SCAN_all.svg",fig= last_plot(),width = 11,height = 10)
ggplot(spfitSCAN$pdep,aes(y = pointp , x = focal_var )) + geom_point() +
  geom_ribbon(aes(ymin = low, ymax = up), alpha = 0.3) +
  xlab(expression("Femininity")) + ylab("Rates of Association") + theme_clean(base_size=20, base_family = "Times") +
  theme(legend.key.size = unit(0.2, 'cm'),legend.position = c(.7, .85), legend.title = element_text(size=20))
save_plot(file="SCAN_all_pred.svg",fig= last_plot() ,width = 9.5,height = 10)


# SCAN2
SCANformula = "SCAN_Nb_SexSame ~ (1 | id) + (1 | year) + Mean_dcM + age +  
    rang + CycleCateg +offset(log(Nb_Scan_IndAllfem))"
spfitSCAN2 = applySPAMM_onevar(SCANformula,dfrepro)
summary(spfitSCAN2$mymodel,details=c(p_value="Wald"))
plot_model(spfitSCAN2$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "") + 
  theme(face = "bold") + theme_light(base_size=16)
ggplot(spfitSCAN2$pdep,aes(y = pointp , x = focal_var)) + geom_point() +
  geom_ribbon(aes(ymin = low, ymax = up), alpha = 0.3) +
  xlab(expression("Femininity (dcm)")) + ylab("Rates of Spatial Proximity") + theme_classic(base_size=16) 


plot_model(spfitSCAN2$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "",axis.labels = rev(c(
  "Femininity",  "Age" , "Rank [Low]" ,"Rank [Medium]" , "Cycle [No]"))) + 
  theme(face = "bold") + theme_clean(base_size=20, base_family = "Times")
#--> ici linteraction pas signif mais dcm et cyclecateg separe ouiii
save_plot(file="SCAN_SR.svg",fig= last_plot(),width = 11,height = 10)
ggplot(spfitSCAN2$pdep,aes(y = pointp , x = focal_var )) + geom_point() +
  geom_ribbon(aes(ymin = low, ymax = up), alpha = 0.3) +
  xlab(expression("Femininity")) + ylab("Rates of Association") + theme_clean(base_size=16) +
  theme(legend.key.size = unit(0.2, 'cm'),legend.position = c(.7, .85), legend.title = element_text(size=20))
save_plot(file="SCAN_SR_pred.svg",fig= last_plot() ,width = 9.5,height = 10)

# SCAN3
SCANformula = "SCAN_Nb_SexSame ~ (1 | id) + (1 | year) + Mean_dcM + age +  
    rang + CycleCateg +offset(log(Nb_Scan_IndAllfem))"
spfitSCAN3 = applySPAMM_onevar(SCANformula,dfnorepro)
summary(spfitSCAN3$mymodel,details=c(p_value="Wald"))
plot_model(spfitSCAN3$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "") + 
  theme(face = "bold") + theme_light(base_size=16)
ggplot(spfitSCAN3$pdep,aes(y = pointp , x = focal_var)) + geom_point() +
  geom_ribbon(aes(ymin = low, ymax = up), alpha = 0.3) +
  xlab(expression("Femininity (dcm)")) + ylab("Rates of Spatial Proximity") + theme_classic(base_size=16) 


plot_model(spfitSCAN3$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "",axis.labels = rev(c(
  "Femininity",  "Age" , "Rank [Low]" ,"Rank [Medium]" , "Cycle [No]"))) + 
  theme(face = "bold") + theme_clean(base_size=20, base_family = "Times")
#--> ici linteraction pas signif mais dcm et cyclecateg separe ouiii
save_plot(file="SCAN_noSR.svg",fig= last_plot(),width = 11,height = 10)
ggplot(spfitSCAN3$pdep,aes(y = pointp , x = focal_var )) + geom_point() +
  geom_ribbon(aes(ymin = low, ymax = up), alpha = 0.3) +
  xlab(expression("Femininity")) + ylab("Rates of Association") + theme_clean(base_size=16) +
  theme(legend.key.size = unit(0.2, 'cm'),legend.position = c(.7, .85), legend.title = element_text(size=20))
save_plot(file="SCAN_noSR_pred.svg",fig= last_plot() ,width = 9.5,height = 10)


################################################################################
#SCAN# AVEC AG
################################################################################
dfall$scanrate = dfall$SCAN_Nb/dfall$Nb_Scan_IndAllmal

AGbySCANformula = "AGall_Nb_SameSex ~ (1 | id) + (1 | year) + Mean_dcM + age +  
    rang + CycleCateg + saison_repro +scanrate+ offset(log(Tps_Foc_IndAllmal))"
spfitSCANAG = applySPAMM_onevar(AGbySCANformula,dfall)
summary(spfitSCANAG$mymodel,details=c(p_value="Wald"))
plot_model(spfitSCANAG$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "") + 
  theme(face = "bold") + theme_light(base_size=16)

plot_model(spfitSCANAG$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "",axis.labels = rev(c(
  "Femininity",  "Age" , "Rank [Low]" ,"Rank [Medium]" , "Cycle [No]","Season [Breeding]","Scan Rates"))) + 
  theme(face = "bold") + theme_clean(base_size=20, base_family = "Times")
#--> ici linteraction pas signif mais dcm et cyclecateg separe ouiii
save_plot(file="AGSCAN_all.svg",fig= last_plot(),width = 11,height = 10)
ggplot(spfitSCANAG$pdep,aes(y = pointp , x = focal_var )) + geom_point() +
  geom_ribbon(aes(ymin = low, ymax = up), alpha = 0.3) +
  xlab(expression("Femininity")) + ylab("Rates of Association") + theme_clean(base_size=16) +
  theme(legend.key.size = unit(0.2, 'cm'),legend.position = c(.7, .85), legend.title = element_text(size=20))
save_plot(file="AGSCAN_all_pred.svg",fig= last_plot() ,width = 9.5,height = 10)

#AG2
dfrepro$scanrate = dfrepro$SCAN_Nb/dfrepro$Nb_Scan_IndAllmal

AGformula = "AGall_Nb_SameSex ~ (1 | id) + (1 | year) + Mean_dcM + age +  
    rang + CycleCateg  +SCAN_Nb+ offset(log(Tps_Foc_IndAllfem))"
spfitSCANAG2 = applySPAMM_onevar(AGformula,dfrepro)
summary(spfitSCANAG2$mymodel,details=c(p_value="Wald"))
plot_model(spfitSCANAG2$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "") + 
  theme(face = "bold") + theme_light(base_size=16)


plot_model(spfitSCANAG2$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "",axis.labels = rev(c(
  "Femininity",  "Age" , "Rank [Low]" ,"Rank [Medium]" , "Cycle [No]","Scan Rates"))) + 
  theme(face = "bold") + theme_clean(base_size=20, base_family = "Times")
#--> ici linteraction pas signif mais dcm et cyclecateg separe ouiii
save_plot(file="AGSCAN_SR.svg",fig= last_plot(),width = 11,height = 10)
ggplot(spfitSCANAG2$pdep,aes(y = pointp , x = focal_var )) + geom_point() +
  geom_ribbon(aes(ymin = low, ymax = up), alpha = 0.3) +
  xlab(expression("Femininity")) + ylab("Rates of Association") + theme_clean(base_size=16) +
  theme(legend.key.size = unit(0.2, 'cm'),legend.position = c(.7, .85), legend.title = element_text(size=20))
save_plot(file="AGSCAN_SR_pred.svg",fig= last_plot() ,width = 9.5,height = 10)


#AG3

dfnorepro$scanrate = dfnorepro$SCAN_Nb/dfnorepro$Nb_Scan_IndAllmal

AGformula = "AGall_Nb_SameSex ~ (1 | id) + (1 | year) + Mean_dcM + age +  
    rang + CycleCateg +  SCAN_Nb+ offset(log(Tps_Foc_IndAllfem))"
spfitSCANAG3 = applySPAMM_onevar(AGformula,dfnorepro)
summary(spfitSCANAG3$mymodel,details=c(p_value="Wald"))
plot_model(spfitSCANAG3$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "") + 
  theme(face = "bold") + theme_light(base_size=16)

plot_model(spfitSCANAG3$mymodel, show.values = TRUE, vline.color = "lightgrey", value.offset = .3, axis.title = "Estimates", title = "",axis.labels = rev(c(
  "Femininity",  "Age" , "Rank [Low]" ,"Rank [Medium]" , "Cycle [No]","Scan Rates"))) + 
  theme(face = "bold") + theme_clean(base_size=20, base_family = "Times")
#--> ici linteraction pas signif mais dcm et cyclecateg separe ouiii
save_plot(file="AGSCAN_noSR.svg",fig= last_plot(),width = 11,height = 10)
ggplot(spfitSCANAG3$pdep,aes(y = pointp , x = focal_var )) + geom_point() +
  geom_ribbon(aes(ymin = low, ymax = up), alpha = 0.3) +
  xlab(expression("Femininity")) + ylab("Rates of Association") + theme_clean(base_size=16) +
  theme(legend.key.size = unit(0.2, 'cm'),legend.position = c(.7, .85), legend.title = element_text(size=20))
save_plot(file="AGSCAN_noSR_pred.svg",fig= last_plot() ,width = 9.5,height = 10)




mylist <- mget(ls(pattern = "spfit")) 


for (n in names(mylist)) {
  print(paste0(n,".doc"))
  print(tab_model(mylist[[n]][1], file = paste0(n,".doc")))
}


library(performance)

for (n in names(mylist)) {
  print(paste0(n,"_checkcoli.png"))
  plot(check_collinearity(mylist[[n]]$mymodel))
  save_plot(paste0(n,"_checkcoli.png"),fig= last_plot(),width = 11,height = 10)
}
