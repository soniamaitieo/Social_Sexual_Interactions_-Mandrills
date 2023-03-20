MFD <- read.csv("/home/sonia/ProjetMandrillus_Photos/Documents/Metadata/MFD_20220303.csv", row.names=1, stringsAsFactors=FALSE)

library(tidyverse)

nbphoto=MFD %>%  group_by(Id_folder) %>%  summarise(Nb = n())

#papier1

df_label_m = read.csv("/home/sonia/Perception_Typicality_CNN/MANDRILL/Analysis5_202204/Datasets/df_typicality_20220425.csv")
nbphoto = df_label_m %>%  group_by(Id_folder, sexe) %>%  summarise(Nb = n())
mean(nbphoto$Nb)
sd(nbphoto$Nb)
df_label_m %>%  group_by( sexe) %>%  summarise(Nb = n())
