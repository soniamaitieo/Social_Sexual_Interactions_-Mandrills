extract_info_ds <- function(CFD, dt.acp){
  cm = table(CFD$y_pred, CFD$y_true)
  CFD_ds = data.frame( 
    Nb_indiv = length(unique(CFD$Model)),
    Nb_photos = nrow(CFD),
    Nb_M = sum(CFD$GenderSelf == "M"),
    Nb_F = sum(CFD$GenderSelf == "F"),
    accuracy = sum(cm[1], cm[4]) / sum(cm[1:4]),
    precision =cm[4] / sum(cm[4], cm[2]),
    sensitivity = cm[4] / sum(cm[4], cm[3]),
    #fscore = (2 * (sensitivity * precision))/(sensitivity + precision),
    specificity = cm[1] / sum(cm[1], cm[2]),
    PC1_varexplained = dt.acp$eig[1,2],
    R2_Attr_dcm_f = R2_Attract_f
  )
  return(CFD_ds)
}

machouille<- function(df, gendersexcol, allvar, AFvar) {
  Lplot = list()
  df=df[df[gendersexcol] == "F",]
  for (af in AFvar) {
    for (a in allvar){
      r <- cor(df[,af], df[,a])
      p <- cor.test(df[,af], df[,a])$p.value
      namep = paste(a,af, sep = ".")
      Lplot[[namep]]['R'] <- r
      Lplot[[namep]]['p'] <- p
      
    }
  }
  return(Lplot)
}