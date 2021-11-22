library(moments)
library(R.matlab)
library(ineq)
library(entropy)
library(seewave)
setwd("D://Github//Open_IIT_DA")

CV <- function(mean, sd){
  (sd/mean)*100
}

mat_data = readMat("dbhole.mat")
name <- names(mat_data)
dh_features = data.frame()
for(i in 1:333){
  #file.name <- paste0("D://Github//Open_IIT_DA//Data//FileName_d1h",i, ".csv")
  #data <- read.csv(file.name)
  
  dh_i = mat_data[[i]]
  dxhy.name_i =names(mat_data[i])
  force.mean_i = mean(dh_i[,1])
  torque.mean_i = mean(dh_i[,2])
  force.median_i = median(dh_i[,1])
  torque.median_i = median(dh_i[,2])
  torque.sd_i = sd(dh_i[,2])
  force.sd_i = sd(dh_i[,1])
  force.max_i = max(dh_i[,1])
  torque.max_i = max(dh_i[,2])
  force.skewness_i = skewness(dh_i[,1])
  torque.skewness_i = skewness(dh_i[,2])
  force.kurtosis_i = kurtosis(dh_i[,1])
  torque.kurtosis_i = kurtosis(dh_i[,2])
  force.coefficient_of_variation_i = CV(force.mean_i, force.sd_i)
  torque.coefficient_of_variation_i = CV(torque.mean_i, torque.sd_i)
  force.IQR_i = IQR(dh_i[,1])
  torque.IQR_i = IQR(dh_i[,2])
  spearman.correlation_i = cor(dh_i[,1], dh_i[,2], method = "spearman")
  pearson.correlation_i = cor(dh_i[,1], dh_i[,2], method = "pearson")
  sp = spectrum(dh_i[,1], method = c("pgram", "ar"), plot = FALSE)
  spectral_density.df_i <- sp$df
  spectral_density.bandwidth_i <- sp$bandwidth
  dual.gini_i <- ineq(mat_data[[i]], type="Gini")
  force.gini_i = ineq(dh_i[,1], type = "Gini")
  torque.gini_i = ineq(dh_i[,2], type = "Gini")
  force.autocorr.mean_i = mean(acf(dh_i[,1], type = "correlation", plot = FALSE)[[1]])
  torque.autocorr.mean_i = mean(acf(dh_i[,2], type = "correlation", plot = FALSE)[[1]])
  force.autocorr.sd_i = sd(acf(dh_i[,1], type = "correlation", plot = FALSE)[[1]])
  torque.autocorr.sd_i = sd(acf(dh_i[,2], type = "correlation", plot = FALSE)[[1]])
  
  dh_features_i = data.frame(dxhy.name = dxhy.name_i, force.mean = force.mean_i,
                             torque.mean = torque.mean_i,
                             force.median = force.median_i,
                             torque.median = torque.median_i,
                             force.sd = force.sd_i, torque.sd = torque.sd_i,
                             force.max = force.max_i, torque.max = force.max_i,
                             force.skewness = force.skewness_i,
                             torque.skewness = torque.skewness_i,
                             force.kurtosis = force.kurtosis_i,
                             torque.kurtosis = torque.kurtosis_i,
                             force.IQR = force.IQR_i, torque.IQR = torque.IQR_i,
                             force.coefficient_of_variation = force.coefficient_of_variation_i,
                             torque.coefficient_of_variation = torque.coefficient_of_variation_i,
                             spearman.correlation = spearman.correlation_i,
                             pearson.correlation = pearson.correlation_i,
                             spectral_density.df = spectral_density.df_i,
                             spectral_density.bandwidth = spectral_density.bandwidth_i,
                             dual.gini = dual.gini_i, force.gini = force.gini_i, torque.gini = torque.gini_i,
                             force.autocorr.mean = force.autocorr.mean_i, 
                             force.autocorr.sd = force.autocorr.sd_i, 
                             torque.autocorr.mean = torque.autocorr.mean_i, 
                             torque.autocorr.sd = torque.autocorr.sd_i)

  if(nrow(dh_features) == 0)
  {
    dh_features = rbind(dh_features, dh_features_i)
    write.csv(x = dh_features, file = "dh_features.csv", row.names = F)
  }
  else{
    write.table(x = dh_features_i, file = "dh_features.csv", append = T, sep = ",", row.names = F, col.names = F)
  }
  cat(dxhy.name_i, "updated\n")
}
