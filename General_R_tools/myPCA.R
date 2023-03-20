library(FactoMineR)
library(factoextra)
library(corrplot)
library(ggpubr)


getPCS <- function(res_emb){
  dt.acp=PCA(res_emb ,axes = c(1,2),ncp=30,graph = FALSE)
  #head(dt.acp$eig)
  #variabilité expliqés par les différents composants - distance cum des variabilités
  #barplot(dt.acp$eig[,1],main="Eboulis de valeur propre", xlab="Nb de composants", names.arg=1:nrow(dt.acp$eig))
  #dt.acp.30dim = dt.acp$ind$coord
  #print(paste("Cumulative percentage of variance for 30 PCs",round(dt.acp$eig[30,3]),"%"))
  return(dt.acp)
}


plot_PCA <- function(dt.acp , eboulis = TRUE , plot_var = TRUE, corrpl = TRUE){
  if (eboulis == TRUE){
    print(fviz_eig(dt.acp, addlabels = T, ylim = c(0, 100)))
  }
  if (plot_var == TRUE){
    print(fviz_pca_var(dt.acp, col.var = "black", repel = T))
  }
  if (corrpl == TRUE){
    print(corrplot(get_pca_var(dt.acp)$cos2, is.corr = FALSE))
  }
}


plot_PCA_indiv_colorsex_bis <- function(dt.acp , gendersexcol) {
  p =fviz_pca_ind(dt.acp, 
               habillage=gendersexcol,
               label="none", palette = c("#b66dff", "#009999") +
                 theme_pander()  + xlab(paste0("PC1 (",round(contrib1),"%)" )) + ylab(paste0("PC2 (",round(contrib2),"%)" )) 
  ) + theme_pander()
  return(p)
}

plot_PCA_indiv_colorsex <- function(dt.acp , gendersexcol) {
  p =fviz_pca_ind(dt.acp, 
                  habillage=gendersexcol,
                  label="none", palette = c("#b66dff", "#009999") 
  ) + theme_pander()
  return(p)
}

plot_PCA_indiv_colorethn <- function(dt.acp , Ethn) {
  p =fviz_pca_ind(dt.acp, 
                  habillage=Ethn,
                  label="none"
  ) + theme_pander()
  return(p)
}

plot_PCA_indiv_colorindsex_top50 <- function(dt.acp , indivcol,gendersexcol) {
  #here we only plot 50 individual for visibility
  p1 = fviz_pca_ind(dt.acp, geom = "point", invisible="quali",
               col.ind =indivcol,
               habillage = "none",pointsize = 4,
               label="none",select.ind = list(cos2 = 50))
  p1 = p1 + theme(legend.position = "none")
  p2 = fviz_pca_ind(dt.acp, 
                    habillage=gendersexcol,pointsize = 4,
                    geom = "point", invisible="quali",
                    select.ind = list(cos2 = 50),
                    label="none", palette = c("#b66dff", "#009999") 
  )
  p1p2 <- ggarrange(p1,p2,
                      ncol = 2, nrow = 1)
  return(p1p2)
}

plot_PCA_indiv_colorindsex_CFDmulti <- function(dt.acp , indivcol,gendersexcol,CFD) {
  #here we only plot 50 individual for visibility
  p1 = fviz_pca_ind(dt.acp, geom = "point", invisible="quali",
                    col.ind =indivcol,
                    habillage = "none",pointsize = 4,
                    label="none",select.ind = list(name=which(duplicated(CFD$Model))))
  p1 = p1 + theme(legend.position = "none")
  p2 = fviz_pca_ind(dt.acp, 
                    habillage=gendersexcol,pointsize = 4,
                    geom = "point", invisible="quali",
                    label="none", palette = c("#b66dff", "#009999"),
                    select.ind = list(name=which(duplicated(CFD$Model)))
  )
  p1p2 <- ggarrange(p1,p2,
                    ncol = 2, nrow = 1)
  return(p1p2)
}


plot_PCA_indiv_colorindsex_CFDmulti50 <- function(dt.acp , indivcol,gendersexcol,CFD) {
  #here we only plot 50 individual for visibility
  p1 = fviz_pca_ind(dt.acp, geom = "point", invisible="quali",
                    col.ind =indivcol,
                    habillage = "none",pointsize = 4,
                    label="none",select.ind = list(name=which(duplicated(CFD$Model))[1:250]))
  p1 = p1 + theme(legend.position = "none")
  p2 = fviz_pca_ind(dt.acp, 
                    habillage=gendersexcol,pointsize = 4,
                    geom = "point", invisible="quali",
                    select.ind = list(name=which(duplicated(CFD$Model))[1:250]),
                    label="none", palette = c("#b66dff", "#009999") 
  )
  p1p2 <- ggarrange(p1,p2,
                    ncol = 2, nrow = 1)
  return(p1p2)
}


getPC1 <- function(dt.acp){
  PC1coord = dt.acp$ind$coord[,1]
  return(PC1coord)
}

plot_PCA_ggplot_ID_CFD<- function(CFD,colScale,contrib1,contrib2){
  ggplot(CFD) + 
    geom_point(aes(pc1, pc2, color = factor(Model), shape = factor(GenderSelf)),size=2) +
    theme_pander() + theme(legend.position = "none") + xlab(paste0("PC1 (",round(contrib1),"%)" )) + ylab(paste0("PC2 (",round(contrib2),"%)" )) + colScale
}



plot_PCA_ggplot_ID<- function(CFD,colScale,contrib1,contrib2){
  ggplot(CFD) + 
    geom_point(aes(pc1, pc2, color = factor(Id), shape = factor(Gender)),size=2) +
    theme_pander() + theme(legend.position = "none") + xlab(paste0("PC1 (",round(contrib1),"%)" )) + ylab(paste0("PC2 (",round(contrib2),"%)" )) + colScale
}
