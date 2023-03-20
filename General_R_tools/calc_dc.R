calc_centroid_PC1 <- function( df , gendersexcol, gendersex, PC1col){
  centroid = mean(df[PC1col][df[gendersexcol] == gendersex,] )
  return(centroid)
}

dist_eachrow_vec <- function(res_emb, vector_c){
  dist_c <- c()
  for (i in 1:nrow(res_emb)){
    va = dist(rbind(res_emb[i,], t(as.matrix(vector_c))))
    dist_c  <- c(dist_c, va)
  }
  return(dist_c)
}


