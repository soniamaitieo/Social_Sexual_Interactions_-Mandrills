library(ggplot2)
library(dplyr)
library(ggthemes)

plotcorr_Attr_dcm_inL <- function(df, gendersexcol, allvar, AFvar) {
  Lplot = list()
  for (af in AFvar) {
    for (a in allvar){
      p1 <- ggplot(data = df, aes_string(x =af, y =a, color=gendersexcol)) +
        geom_smooth(method="lm",se=FALSE) +
        geom_point(alpha = 0.9,size = 1.5) + stat_cor(aes_string(color = gendersexcol ),size = 6)  + 
        scale_color_manual(values=c("#b66dff", "#009999")) +
        theme_classic(base_size=14)
      namep = paste(a,af, sep = ".")
      Lplot[[namep]] <- p1
    }
  }
  return(Lplot)
}

plotcorr_Attr_dcm_inL_fem <- function(df, gendersexcol, allvar, AFvar) {
  Lplot = list()
  df=df[df[gendersexcol] == "F",]
  for (af in AFvar) {
    for (a in allvar){
      p1 <- ggplot(data = df, aes_string(y =af, x =a, color=gendersexcol)) +
        geom_smooth(method="lm",se=FALSE) +
        geom_point(alpha = 0.9,size = 1.5) + stat_cor(aes_string(color = gendersexcol ),size = 6)  + 
        scale_color_manual(values=c("#b66dff", "#009999")) +
        theme_classic(base_size=24)
      namep = paste(a,af, sep = ".")
      Lplot[[namep]] <- p1
    }
  }
  return(Lplot)
}

plotcorr_Attr_dcm_inL_femethn <- function(df, gendersexcol, allvar, AFvar) {
  Lplot = list()
  df=df[df[gendersexcol] == "F",]
  for (af in AFvar) {
    for (a in allvar){
      p1 <- ggplot(data = df, aes_string(y =af, x =a)) +
        geom_smooth(method="lm",se=FALSE,color = "#9c2bff") +
        #geom_point(alpha = 0.9,size = 1.5) + stat_cor(label.x.npc = 0.8, label.y.npc = 0.15,color="#9c2bff", label.sep='\n',size = 4, family="Times" ) + 
        
        geom_point(alpha = 1,size = 1.5) + stat_cor(label.x.npc = 0.40, label.y.npc = 0.10,color="#9c2bff",size = 5, family="Times" ) + 
        xlim(0, 18) + ylim(0,6) + 
        #scale_color_manual(values=c("#b66dff")) + 
        theme_clean(base_size=18, base_family = "Times") 
         #theme_tufte(base_size = 15, ticks = T) 
      namep = paste(a,af, sep = ".")
      Lplot[[namep]] <- p1
    }
  }
  return(Lplot)
}
