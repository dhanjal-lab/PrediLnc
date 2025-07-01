

library(DOSE)



data <- read.csv("create_x1_forDisease.csv", header = FALSE)  
disease_list <- as.list(data[[1]])

length(disease_list)
# Extract the last element from the list
target <- tail(disease_list, n = 1)
target
# Remove the last element from the list
disease_list <- disease_list[-length(disease_list)]

ddsim<-doSim(disease_list,target, measure = "Wang")

ddsim[1:100]
dim(ddsim)
ddsim[is.na(ddsim)] <- 0
write.csv(ddsim, file = "ddsim_target.csv", row.names = FALSE)

