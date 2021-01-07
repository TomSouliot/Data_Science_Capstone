## HarvardX: PH125.9x Data Science: Capstone
## Air pressure system failures in Scania trucks Project
## Predict failures and minimize costs based on sensor readings
## Souliotis Thomas
## December 2020


##########################################################
# Load Required Libraries
##########################################################
#-----------------------------------------------------------------

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(funModeling)) install.packages("funModeling", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(DMwR)) install.packages("DMwR", repos = "http://cran.us.r-project.org")
if(!require(doParallel)) install.packages("doParallel", repos = "http://cran.us.r-project.org")
if(!require(caTools)) install.packages("caTools", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(psych)) install.packages("psych", repos = "http://cran.us.r-project.org")
if(!require(magrittr)) install.packages("magrittr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(knitr)
library(gridExtra)
library(kableExtra)
library(funModeling)
library(corrplot)
library(matrixStats)
library(DMwR)
library(doParallel)
library(caTools)
library(randomForest)
library(xgboost)
library(psych)
library(magrittr)
#-----------------------------------------------------------------


##########################################################
# Load and Create Train and Test Datasets
##########################################################
#-----------------------------------------------------
## Load training and test datasets from github repository
train_data<-read.csv("https://raw.githubusercontent.com/TomSouliot/Data_Science_Capstone/main/aps_failure_training_set.csv")
test_data<-read.csv("https://raw.githubusercontent.com/TomSouliot/Data_Science_Capstone/main/aps_failure_test_set.csv")

## Global setting
options(digits = 3)

## Create the Datasets
#-----------------------------------------------------
## Train Dataset
y_train<-as.factor(train_data$class)
x_train<-sapply(train_data, as.numeric)
x_train<-as.data.frame(x_train)%>%select(-class)

## Test Dataset
y_test<-as.factor(test_data$class)
x_test<-sapply(test_data, as.numeric)
x_test<-as.data.frame(x_test)%>%select(-class)
#----------------------------------------------------

## Initial Dataset Characteristics
rtr <- dim(x_train)[1]
ctr <- dim(x_train)[2]
rts <- dim(x_test)[1]
nap <- sum(is.na(x_train))/sum(!is.na(x_train))*100
perce <- data.frame(prop.table(table(y_train)))
#----------------------------------------------------

##########################################################
# Dataset Structure
##########################################################

## Features Names
features<-colnames(x_test)
#------------------------------------------------------
# make structure dataset for first 50 features
#dstr <- data.frame(variable = names(x_train[,1:50]),
#                   class = sapply(x_train[,1:50], typeof),
#                   First_Values = sapply(x_train[,1:50], function(x) paste0(head(x),  collapse = ", ")),
#                   row.names = NULL)

# plot structure as table
#kable(dstr, caption = "Dataset Structure", booktabs = TRUE, linesep = "")
#rm(dstr)

#-----------------------------------------------------

##########################################################
# Data Analysis
##########################################################

## I. Histogram Features
##########################################################
## The 7 histogram bins are ag, ay, az, ba, cn, cs, ee
#------------------------------------------------------
# Histogram Bins
ag_hist= x_train[,c(7:16)]
ay_hist= x_train[,c(33:42)]
az_hist= x_train[,c(43:52)]
ba_hist= x_train[,c(53:62)]
cn_hist= x_train[,c(100:109)]
cs_hist= x_train[,c(114:123)]
ee_hist= x_train[,c(159:168)]
# Sum across bins
ag_cml = rowSums(ag_hist)
ay_cml = rowSums(ay_hist)
az_cml = rowSums(az_hist)
ba_cml = rowSums(ba_hist)
cn_cml = rowSums(cn_hist)
cs_cml = rowSums(cs_hist)
ee_cml = rowSums(ee_hist)

# Sum is the same for all bins
h=as.data.frame(cbind(ag_cml, ay_cml, az_cml, ba_cml, cn_cml, cs_cml, ee_cml))
h1<- rowSds(as.matrix(h))
h2<- rowMeans(as.matrix(h))

## Remove all the histogram bins features and replace by the Operational Time
x_train_m11<- x_train[, -c(7:16, 33:42, 43:52, 53:62, 100:109, 114:123, 159:168 )]
x_train_m11<- cbind(x_train_m11, Ttot = h2)

#--------------------------------------------------
## Perform the same on test set
agt_hist= x_test[,c(7:16)]
ayt_hist= x_test[,c(33:42)]
azt_hist= x_test[,c(43:52)]
bat_hist= x_test[,c(53:62)]
cnt_hist= x_test[,c(100:109)]
cst_hist= x_test[,c(114:123)]
eet_hist= x_test[,c(159:168)]
# Sum across bins
agt_cml = rowSums(agt_hist)
ayt_cml = rowSums(ayt_hist)
azt_cml = rowSums(azt_hist)
bat_cml = rowSums(bat_hist)
cnt_cml = rowSums(cnt_hist)
cst_cml = rowSums(cst_hist)
eet_cml = rowSums(eet_hist)

ht=as.data.frame(cbind(agt_cml, ayt_cml, azt_cml, bat_cml, cnt_cml, cst_cml, eet_cml))
ht1<- rowSds(as.matrix(ht))
ht2<- rowMeans(as.matrix(ht))

x_test_m11<- x_test[, -c(7:16, 33:42, 43:52, 53:62, 100:109, 114:123, 159:168 )]
x_test_m11<- cbind(x_test_m11, Ttot = ht2)
#----------------------------------------------

#----------------------------------------------
# Remove unnecessary data
rm(ag_hist, ay_hist,az_hist, ba_hist, cn_hist, cs_hist, ee_hist, 
   ag_cml, ay_cml, az_cml, ba_cml, cn_cml, cs_cml, ee_cml)

rm(agt_hist, ayt_hist,azt_hist, bat_hist, cnt_hist, cst_hist, eet_hist, 
   agt_cml, ayt_cml, azt_cml, bat_cml, cnt_cml, cst_cml, eet_cml,
   h, ht, ht1, ht2)
#----------------------------------------------

## Failures vs Total operating Time
#-----------------------------------------------
#  Grouped histogram for pos and neg classes vs Total Operating Time
data.frame(Type=y_train,Ttot= x_train_m11[,101]/86400)%>% group_by(Type)%>%
  ggplot(aes(Ttot,fill= Type))+
  scale_x_continuous(trans="sqrt", breaks= c(0,100, 200, 500, 1000, 2000), limits = c(0,2000))+
  geom_histogram(aes(y = ..density..), alpha=0.5, bins=200) +
  xlab("Total Operating Time (counted in DayHours)")+
  ylab("Density")+
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=12,face="bold"))
#----------------------------------------------

##  II. Zero Variability Features
##########################################################

## Check for features with 0 variance
#---------------------------------------------
clsds1<-colSds(as.matrix(x_train_m11), na.rm= TRUE)
clsds2<-colSds(as.matrix(x_train), na.rm= TRUE)

# Columns with 0 variance
sum(clsds1==0)
sum(clsds2==0)

# Remove features with 0 variance |Reduced
x_train_m12<-x_train_m11[!clsds1==0]
x_test_m12<-x_test_m11[!clsds1==0]

# Remove features with 0 variance |Extended
x_train_m22<-x_train[!clsds2==0]
x_test_m22<-x_test[!clsds2==0]
#---------------------------------------------

##  III. Missing Values
##########################################################

## Calculate percent of nas per Feature
#--------------------------------------------------------
# Percentage na per feauture
nacls1<-colSums(is.na(x_train_m12))/ dim(x_train_m12)[1]
nacls2<-colSums(is.na(x_train_m22))/ dim(x_train_m22)[1]

# Percentage na per feature for pos and neg |Reduced
x_tr1_p <- x_train_m12[y_train=="pos",]
x_tr1_n <- x_train_m12[y_train=="neg",]

nacls1_p<-colSums(is.na(x_tr1_p))/ dim(x_tr1_p)[1]
nacls1_n<-colSums(is.na(x_tr1_n))/ dim(x_tr1_n)[1]

x_tr2_p <- x_train_m22[y_train=="pos",]
x_tr2_n <- x_train_m22[y_train=="neg",]

# Percentage na per feature for pos and neg |Extended
nacls2_p<-colSums(is.na(x_tr2_p))/ dim(x_tr2_p)[1]
nacls2_n<-colSums(is.na(x_tr2_n))/ dim(x_tr2_n)[1]

# Data frames for plots
k11<-colnames(x_tr1_p)
k21<-colnames(x_tr2_p)

k12<-data.frame(k11, p_tr = nacls1_p, n_tr = nacls1_n)
k22<-data.frame(k21, p_tr = nacls2_p, n_tr = nacls2_n)
#--------------------------------------------------------

# Plot percentage na per feature for pos and neg |Reduced
ggplot(k12,aes(x=k11))+
  geom_point(aes(y=n_tr), colour="#F8766D",size=5,shape=1)+
  geom_point(aes(y=p_tr),colour="#00BFC4", size=3)+
  xlab("Features")+
  ylab("Percentage of na values per feature")+
  theme(legend.position = "right")+
  scale_fill_discrete(name = "Type",labels = c("neg", "pos"))+
  theme(axis.text.x = element_text(angle=90, hjust=1))+
  theme(axis.text=element_text(size=7),
        axis.title=element_text(size=12,face="bold"))
#------------------------------------------------------------------

# produce the same plot for the extended dataset -> same behavior
ggplot(k22,aes(x=k21))+
  geom_point(aes(y=n_tr), colour="#F8766D",size=5,shape=1)+
  geom_point(aes(y=p_tr),colour="#00BFC4", size=3)+
  xlab("Features")+
  ylab("Percentage of na values per feature")+
  theme(legend.position = "right")+
  scale_fill_discrete(name = "Type",labels = c("neg", "pos"))+
  theme(axis.text.x = element_text(angle=90, hjust=1))+
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=12,face="bold"))
#------------------------------------------------------------------

## Exclude features with high percentage of nas both for pos and neg
#---------------------------------------------------------------------
# Reduced
x_train_m13<-x_train_m12[!(k12$p_tr>0.75 & k12$n_tr>0.75)]
x_test_m13<-x_test_m12[!(k12$p_tr>0.75 & k12$n_tr>0.75)]

# Extended
x_train_m23<-x_train_m22[!(k22$p_tr>0.75 & k22$n_tr>0.75)]
x_test_m23<-x_test_m22[!(k22$p_tr>0.75 & k22$n_tr>0.75)]
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# Function to fill na with column median
fna=function(x){x[is.na(x)]= median(x, na.rm= TRUE)
x}
# Fill nas of Reduced Dataset
x_train_m14=data.frame(apply(x_train_m13,2,fna))
x_test_m14=data.frame(apply(x_test_m13,2,fna))

# Fill nas of Extended Dataset
x_train_m24=data.frame(apply(x_train_m23,2,fna))
x_test_m24=data.frame(apply(x_test_m23,2,fna))

# Confirm that there are no missing values
sum(is.na(x_train_m14))
sum(is.na(x_test_m14))

sum(is.na(x_train_m24))
sum(is.na(x_test_m24))
#---------------------------------------------------------------------

#---------------------------------------------------------------------
rm(x_train_m13, x_test_m13, x_train_m23, x_test_m23  )
#---------------------------------------------------------------------

##  IV. Scaling
##########################################################

#---------------------------------------------------------------------
# Scale Reduced Dataset
x1i <- as.matrix(x_train_m14)
x1c <- sweep(x1i, 2, colMeans(x1i))
x1s <- sweep(x1c, 2, colSds(x1i), FUN = "/")

# Scale Extended Dataset
x2i <- as.matrix(x_train_m24)
x2c <- sweep(x2i, 2, colMeans(x2i))
x2s <- sweep(x2c, 2, colSds(x2i), FUN = "/")

# Scale Extended Test set -> required for simplified model
x2i_ts <- as.matrix(x_test_m24)
x2c_ts <- sweep(x2i_ts, 2, colMeans(x2i))
x2s_ts <- sweep(x2c_ts, 2, colSds(x2i), FUN = "/")
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# Confirm that standard deviation is 1 across all features
plot(round(colSds(x1s)))
plot(round(colSds(x2s)))
#---------------------------------------------------------------------


##  V. Features Distribution
##########################################################

#-----------------------------------------------------
# Plot distribution of first 25 features
x1sd<-data.frame(x1s)
plot_num(x1sd[,1:25], bins=10)
#plot_num(x1sd[,26:50], bins=10)
#plot_num(x1sd[,51:75], bins=10)
#-----------------------------------------------------

#-----------------------------------------------------
# plot summary as table for first 80 features
#kable(t(summary(x1s[,1:70])), caption = "Dataset Summary", booktabs = TRUE, linesep = "") %>%
#  kable_styling(font_size = 7)

#-----------------------------------------------------

##  VI. Features Correlation
##########################################################

#-----------------------------------------------------
# Correlation matrix
train_cor1 <- cor(x1s)
train_cor2 <- cor(x2s)

# Find features with high correlation >0.9
hcor1 <- findCorrelation(train_cor1, cutoff=0.9,names = TRUE)
hcor2 <- findCorrelation(train_cor2, cutoff=0.9,names = TRUE)
#-----------------------------------------------------

#-----------------------------------------------------
# Plot together
par(mfrow = c(1, 2))

# Correlation plot |Reduced
corrplot(train_cor1, diag = FALSE, order = "FPC",
         tl.pos = "td", tl.cex = 0.5, method = "color", type = "upper", title= "Reduced", mar=c(0,0,1,0))

# Correlation plot |Extended
corrplot(train_cor2, diag = FALSE, order = "FPC",
         tl.pos = "td", tl.cex = 0.5, method = "color", type = "upper", title = "Extended", mar=c(0,0,1,0))
#-----------------------------------------------------

##  VII. Principal Component Analysis PCA
##########################################################

#-----------------------------------------------------
# PCA on datasets
pca_train1 <- prcomp(x_train_m14, scale. = TRUE, center = TRUE)
pca_train2 <- prcomp(x_train_m24, scale. = TRUE, center = TRUE)

# Check cumulative proportion of variance explained
summary(pca_train1)
summary(pca_train2)
#-----------------------------------------------------

#-----------------------------------------------------
# PCA1 vs PCA2 |Reduced
p1 <-data.frame(pca_train1$x[,1:2], type = y_train) %>%
  ggplot(aes(PC1, PC2, color = type)) +
  geom_point()+ 
  ggtitle("Reduced")+
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=12,face="bold"))

# PCA3 vs PCA4 |Reduced
p2 <-data.frame(pca_train1$x[,3:4], type = y_train) %>%
  ggplot(aes(PC3, PC4, color = type)) +
  geom_point()+
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=12,face="bold"))

# PCA1 vs PCA2 |Extended
p3 <-data.frame(pca_train2$x[,1:2], type = y_train) %>%
  ggplot(aes(PC1, PC2, color = type)) +
  geom_point() + 
  ggtitle("Extended") +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=12,face="bold"))

# PCA3 vs PCA4 |Extended
p4 <-data.frame(pca_train2$x[,3:4], type = y_train) %>%
  ggplot(aes(PC3, PC4, color = type)) +
  geom_point() +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=12,face="bold"))

grid.arrange(p1, p3, p2, p4, nrow =2, ncol=2)
#-----------------------------------------------------

#-----------------------------------------------------
# Calculate Variance Explained |Reduced
cml_var_explained1 <- cumsum(pca_train1$sdev^2 / sum(pca_train1$sdev^2))
cml_var_explained1 <- data.frame(c(1:length(pca_train1$sdev)),cml_var_explained1)
names(cml_var_explained1)[1] = 'PCs'
names(cml_var_explained1)[2] = 'Percentage'
#-----------------------------------------------------
# Calculate features required for 95% Var |Reduced
features_needed_Red<-min(which(cml_var_explained1>0.95))
#-----------------------------------------------------
# For simplicity not calculated for Extended
# -> obvious from summary of PCA
#-----------------------------------------------------

#-----------------------------------------------------
# Visualize PCAs required
ggplot(cml_var_explained1 , aes(x = PCs, y = Percentage))+ 
  geom_point(colour="blue",size=2, shape=1) + geom_line(colour="blue")+
  geom_abline(intercept = 0.95, color = "red", lty=2, lwd=1, slope = 0)+
  xlab("Principal Components")+
  ylab("Variance Explained")+
  theme_bw() +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=12,face="bold"))
#-----------------------------------------------------

#-----------------------------------------------------
# Select first 50 PCAS for 95% var |Reduced
pca_train_m1<- as.data.frame(pca_train1$x)
pca_train_m1<- pca_train_m1[,c(1:50)]

# Select first 80 PCAS for 95% var |Extended
pca_train_m2<- as.data.frame(pca_train2$x)
pca_train_m2<- pca_train_m2[,c(1:80)]
#-----------------------------------------------------

## PCA transformation on test sets
#-----------------------------------------------------
pca_test1<- predict(pca_train1, x_test_m14)
pca_test2<- predict(pca_train2, x_test_m24)

pca_test_m1<- as.data.frame(pca_test1)
pca_test_m1<- pca_test_m1[,c(1:50)]

pca_test_m2<- as.data.frame(pca_test2)
pca_test_m2<- pca_test_m2[,c(1:80)]
#-----------------------------------------------------

##  VIII. Data Class Balancing
##########################################################

#-----------------------------------------------------
## Distribution of pos and neg
data.frame(type=y_train)%>%
  ggplot()+
  geom_bar(aes(x=type, fill=type), width =0.6)+ 
  xlab("Type")+
  ylab("Count")+
  theme_bw() +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=12,face="bold"))
#-----------------------------------------------------

## Balancing SMOTE 
#---------------------------------------------
# SMOTE balancing train |REduced
set.seed(1989, sample.kind = "Rounding")
pca_train_m1bal <- as.data.frame(cbind(pca_train_m1, Type=y_train))
pca_train_m1bal = SMOTE(Type~., pca_train_m1bal, perc.over = 1900, perc.under = 157.895, k=5)

# SMOTE balancing train |Extended
set.seed(1989, sample.kind = "Rounding")
pca_train_m2bal <- as.data.frame(cbind(pca_train_m2, Type=y_train))
pca_train_m2bal = SMOTE(Type~., pca_train_m2bal, perc.over = 1900, perc.under = 157.895, k=5)
#-----------------------------------------

## Final Training-Test sets
##########################################################
#---------------------------------------------
# Reduced
y_train_f1<-as.factor(pca_train_m1bal$Type)
x_train_f1<-as.data.frame(pca_train_m1bal)%>%select(-Type)
x_test_f1<- pca_test_m1

# Extended
y_train_f2<-as.factor(pca_train_m2bal$Type)
x_train_f2<-as.data.frame(pca_train_m2bal)%>%select(-Type)
x_test_f2<- pca_test_m2
#---------------------------------------------

#---------------------------------------------
rm(x1i, x1c, x2i, x2c, x2i_ts, x2c_ts)
#---------------------------------------------

##########################################################
# Models
##########################################################

##  I. Simplified Engineering Method
##########################################################

#---------------------------------------------
# na per feature |Negative Class
n_train<-x_train[y_train=="neg",]
nacl_n_train <-colSums(is.na(n_train))

# na per feature |Positive Class
p_train<-x_train[y_train=="pos",]
nacl_p_train <-colSums(is.na(p_train))

# data.frame with combined pos and neg na info
nacl <-data.frame(features, p_train=nacl_p_train/dim(p_train)[1], 
                  n_train = nacl_n_train/dim(n_train)[1])
#---------------------------------------------

# na percentage of features for pos and neg class
#---------------------------------------------
ggplot(nacl,aes(x=features))+
  geom_point(aes(y=n_train),colour="#F8766D", size=3)+
  geom_point(aes(y=p_train),colour="#00BFC4", size=3)+
  xlab("Features")+
  ylab("Percentage of missing values")+
  #theme_bw() +
  theme(axis.text.x = element_text(angle=90, hjust=1))+
  theme(axis.text=element_text(size=7),
        axis.title=element_text(size=12,face="bold"))
#---------------------------------------------

#---------------------------------------------
# Select features with high na perc in pos class
nacl <-nacl[nacl$p_train>0.3,]
# Keep features with low na perc in neg class
nacl <-nacl[nacl$n_train<0.1,]
#---------------------------------------------

# Plot na percentage of features selected
#---------------------------------------------
ggplot(nacl,aes(x=features))+
  geom_point(aes(y=n_train),colour="#F8766D", size=3)+
  geom_point(aes(y=p_train),colour="#00BFC4", size=3)+
  xlab("Selected Features")+
  ylab("Percentage of missing values")+
  #theme_bw() +
  theme(axis.text.x = element_text(angle=90, hjust=1))+
  theme(axis.text=element_text(size=7),
        axis.title=element_text(size=12,face="bold"))
#---------------------------------------------

#---------------------------------------------
# predict failure if there is na in selected features
index<-features%in%nacl$features
x_test_na<-x_test[,index]
train_na<-rowSums(is.na(x_test_na))
pred_na<-ifelse(train_na>0,"pos","neg")
pred_na<-factor(pred_na, levels= levels(y_test))
cm_na <- confusionMatrix(pred_na, y_test)
#---------------------------------------------

# Store results for Simplified NA method
results<-tibble(Method="Simplified NA method", 
                Dataset = "Raw",
                Specificity = cm_na$byClass["Specificity"],
                Sensitivity = cm_na$byClass["Sensitivity"],
                FN = cm_na$table[1,2], 
                FP = cm_na$table[2,1],
                Cost = FN*500 +FP*10)
#---------------------------------------------

#---------------------------------------------
#results %>% knitr::kable(booktabs = T, linesep = "")%>%
#  kable_styling(position = "center")
#---------------------------------------------

##  II. k-means Clustering
##########################################################

#-----------------------------------------------------------------------
# Perform k-means clustering on scaled train set with 2 clusters 
set.seed(1989, sample.kind = "Rounding")
k <- kmeans(x2s, centers = 2)
#-----------------------------------------------------------------------

# The defined predict_kmeans() function takes two arguments 
# 1.a matrix of observations x
# 2.a k-means object k 
# Assigns each row of x to a cluster from k ->Prediction
predict_kmeans <- function(x, k) {
  centers <- k$centers    # extract cluster centers
  # calculate distance to cluster centers
  distances <- sapply(1:nrow(x), function(i){
    apply(centers, 1, function(y) dist(rbind(x[i,], y)))
  })
  max.col(-t(distances))  # select cluster with min distance to center
}
#-----------------------------------------------------------------------

# Use the predict_kmeans() function to make predictions on the test set
pred_kmeans <- ifelse(predict_kmeans(x2s_ts, k) == 2, "neg", "pos")
pred_kmeans <- factor(pred_kmeans, levels= levels(y_test))
cm_kmeans <- confusionMatrix(pred_kmeans, y_test)
#-----------------------------------------------------------------------

# Store results for kmeans clustering method
results<-bind_rows(results,tibble(Method="k-means Clustering",
                                  Dataset = "Scaled",
                                  Specificity = cm_kmeans$byClass["Specificity"],            
                                  Sensitivity = cm_kmeans$byClass["Sensitivity"],
                                  FN = cm_kmeans$table[1,2], 
                                  FP = cm_kmeans$table[2,1],
                                  Cost = FN*500 +FP*10))
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
#results %>% knitr::kable(booktabs = T, linesep = "")%>%
#  kable_styling(position = "center")
#-----------------------------------------------------------------------

#----------------------------------------------
# Remove unnecessary data
rm(x_train_m11, x_test_m11, x_train_m12, x_test_m12, x_train_m22, x_test_m22, 
   x_train_m14, x_test_m14, x_train_m24, x_test_m24, train_cor1, train_cor2,
   pca_train_m1, pca_train_m2, pca_test1, pca_test2, pca_test_m1, pca_test_m2)

rm(clsds1, clsds2, nacls1, nacls2, nacls1_p, nacls1_n, nacls2_p, nacls2_n, 
   k11, k21, k12, k22, n_train, nacl_n_train, p_train, nacl_p_train, nacl)
#----------------------------------------------

##  III. Logistic Regression
##########################################################

## Logistic Regression Reduced
#-----------------------------------------------------------------------
set.seed(1989, sample.kind = "Rounding")
train_glm1 <- train(x_train_f1, y_train_f1,
                    method = "glm", 
                    family = "binomial")


pred_glm1 <- predict(train_glm1, x_test_f1)
cm_glm1 <- confusionMatrix(pred_glm1, y_test)
#-----------------------------------------------------------------------

results<-bind_rows(results,tibble(Method="Logistic Regression", 
                                  Dataset = "Reduced",
                                  Specificity = cm_glm1$byClass["Specificity"],                  
                                  Sensitivity = cm_glm1$byClass["Sensitivity"],
                                  FN = cm_glm1$table[1,2], 
                                  FP = cm_glm1$table[2,1],
                                  Cost = FN*500 +FP*10))
#-----------------------------------------------------------------------

## Logistic Regression Extended
#-----------------------------------------------------------------------
set.seed(1989, sample.kind = "Rounding")
train_glm2 <- train(x_train_f2, y_train_f2,
                    method = "glm", 
                    family = "binomial")


pred_glm2 <- predict(train_glm2, x_test_f2)
cm_glm2 <- confusionMatrix(pred_glm2, y_test)
#-----------------------------------------------------------------------

results<-bind_rows(results,tibble(Method="Logistic Regression",
                                  Dataset = "Extended",
                                  Specificity = cm_glm2$byClass["Specificity"],                  
                                  Sensitivity = cm_glm2$byClass["Sensitivity"],
                                  FN = cm_glm2$table[1,2], 
                                  FP = cm_glm2$table[2,1],
                                  Cost = FN*500 +FP*10))
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
#results %>% knitr::kable(booktabs = T, linesep = "")%>%
#  kable_styling(position = "center")%>%
#  row_spec(2, hline_after = TRUE)
#-----------------------------------------------------------------------

##  IV. Boosted Logistic Regression
##########################################################

## Boosted Logistic Regression Reduced
#-----------------------------------------------------------------------
# Parallel Processing settings
cl <- makePSOCKcluster(6)
registerDoParallel(cl)
#getDoParWorkers()
#-----------------------------------------------------------------------

set.seed(1989, sample.kind = "Rounding")

# Initial Tuning Grid for optimization
#tuning_lgb1 <- data.frame(nIter = c(15, 21, 25 ))

# Optimized parameter for Specificity metric
tuning_lgb1 <- data.frame(nIter = c( 25 ))

control_lgb1 <- trainControl(method= "cv", number = 6, p= 0.8, 
                             #verboseIter = TRUE, 
                             classProbs = TRUE, 
                             summaryFunction = twoClassSummary,
                             allowParallel = TRUE)
#-----------------------------------------------------------------------

set.seed(1989, sample.kind = "Rounding")

train_lgb1 <- train(x_train_f1, y_train_f1,
                    method = "LogitBoost",
                    trControl = control_lgb1,
                    tuneGrid = tuning_lgb1,
                    metric = "Spec",
                    allowParallel = TRUE)
#verbose = TRUE)

stopCluster(cl)
registerDoSEQ()

pred_lgb1<- predict(train_lgb1, x_test_f1)
cm_lgb1<- confusionMatrix(pred_lgb1, y_test)
#-----------------------------------------------------------------------

results<-bind_rows(results,tibble(Method=" Boosted Logistic Regression",
                                  Dataset = "Reduced",
                                  Specificity = cm_lgb1$byClass["Specificity"],                  
                                  Sensitivity = cm_lgb1$byClass["Sensitivity"],
                                  FN = cm_lgb1$table[1,2], 
                                  FP = cm_lgb1$table[2,1],
                                  Cost = FN*500 +FP*10))
#-----------------------------------------------------------------------

## Boosted Logistic Regression Extended
#-----------------------------------------------------------------------
# Parallel Processing settings
cl <- makePSOCKcluster(6)
registerDoParallel(cl)
#getDoParWorkers()
#-----------------------------------------------------------------------

set.seed(1989, sample.kind = "Rounding")

# Initial Tuning Grid for optimization
#tuning_lgb2 <- data.frame(nIter = c(15, 21, 25 ))

# Optimized parameter for Specificity metric
tuning_lgb2 <- data.frame(nIter = c( 25 ))

control_lgb2 <- trainControl(method= "cv", number = 6, p= 0.8, 
                             #verboseIter = TRUE, 
                             classProbs = TRUE, 
                             summaryFunction = twoClassSummary,
                             allowParallel = TRUE)
#-----------------------------------------------------------------------

set.seed(1989, sample.kind = "Rounding")

train_lgb2 <- train(x_train_f2, y_train_f2,
                    method = "LogitBoost",
                    trControl = control_lgb2,
                    tuneGrid = tuning_lgb2,
                    metric = "Spec",
                    allowParallel = TRUE)
#verbose = TRUE)

stopCluster(cl)
registerDoSEQ()

pred_lgb2 <- predict(train_lgb2, x_test_f2)
cm_lgb2 <- confusionMatrix(pred_lgb2, y_test)
#-----------------------------------------------------------------------

results<-bind_rows(results,tibble(Method=" Boosted Logistic Regression",
                                  Dataset = "Extended",
                                  Specificity = cm_lgb2$byClass["Specificity"],
                                  Sensitivity = cm_lgb2$byClass["Sensitivity"],
                                  FN = cm_lgb2$table[1,2], 
                                  FP = cm_lgb2$table[2,1],
                                  Cost = FN*500 +FP*10))
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
#results %>% knitr::kable(booktabs = T, linesep = "")%>%
#  kable_styling(position = "center")%>%
#  row_spec(c(2,4), hline_after = TRUE)
#-----------------------------------------------------------------------

##  V. Random Forest
##########################################################

## Random Forest Reduced
#-----------------------------------------------------------------------
# Parallel Processing settings
cl <- makePSOCKcluster(6)
registerDoParallel(cl)
#getDoParWorkers()
#-----------------------------------------------------------------------

set.seed(1989, sample.kind = "Rounding")

control_rf1<- trainControl(method= "cv", number = 6, p= 0.8, 
                           #verboseIter = TRUE, 
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary,
                           allowParallel = TRUE)

# Initial Tuning Grid for optimization
#tuning_rf1 <- data.frame(mtry = c( 5, 7, 9))

# Optimized parameter for Specificity metric
tuning_rf1 <- data.frame(mtry = 9)
#-----------------------------------------------------------------------

set.seed(1989, sample.kind = "Rounding")

train_rf1 <- train(x_train_f1, y_train_f1,
                   method = "rf",
                   trControl = control_rf1,
                   tuneGrid = tuning_rf1,
                   metric = "Spec",
                   importance = TRUE,
                   allowParallel = TRUE)
#verbose = TRUE)
# default ntrees=500
# default p=0.75

stopCluster(cl)
registerDoSEQ()

pred_rf1<- predict(train_rf1, x_test_f1)
cm_rf1 <- confusionMatrix(pred_rf1, y_test)
#-----------------------------------------------------------------------

results<-bind_rows(results,tibble(Method="Random Forest",
                                  Dataset = "Reduced",
                                  Specificity = cm_rf1$byClass["Specificity"],
                                  Sensitivity = cm_rf1$byClass["Sensitivity"],
                                  FN = cm_rf1$table[1,2], 
                                  FP = cm_rf1$table[2,1],
                                  Cost = FN*500 +FP*10))
#-----------------------------------------------------------------------

## Random Forest Extended
#-----------------------------------------------------------------------
# Parallel Processing settings
cl <- makePSOCKcluster(6)
registerDoParallel(cl)
#getDoParWorkers()
#-----------------------------------------------------------------------

set.seed(1989, sample.kind = "Rounding")

control_rf2<- trainControl(method= "cv", number = 6, p= 0.8, 
                           #verboseIter = TRUE, 
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary,
                           allowParallel = TRUE)

# Initial Tuning Grid for optimization
#tuning_rf2 <- data.frame(mtry = c( 5, 7, 9))

# Optimized parameter for Specificity metric
tuning_rf2 <- data.frame(mtry = 9)
#-----------------------------------------------------------------------

set.seed(1989, sample.kind = "Rounding")

train_rf2 <- train(x_train_f2, y_train_f2,
                   method = "rf",
                   trControl = control_rf2,
                   tuneGrid = tuning_rf2,
                   metric = "Spec",
                   importance = TRUE,
                   allowParallel = TRUE)
#verbose = TRUE)

stopCluster(cl)
registerDoSEQ()

pred_rf2<- predict(train_rf2, x_test_f2)
cm_rf2 <- confusionMatrix(pred_rf2, y_test)
#-----------------------------------------------------------------------

results<-bind_rows(results,tibble(Method="Random Forest",
                                  Dataset = "Extended",
                                  Specificity = cm_rf2$byClass["Specificity"],
                                  Sensitivity = cm_rf2$byClass["Sensitivity"],
                                  FN = cm_rf2$table[1,2], 
                                  FP = cm_rf2$table[2,1],
                                  Cost = FN*500 +FP*10))
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
#results %>% knitr::kable(booktabs = T, linesep = "")%>%
#  kable_styling(position = "center")%>%
#  row_spec(c(2,4,6), hline_after = TRUE)
#-----------------------------------------------------------------------

##  VI. Extreme Gradient Boosting
##########################################################

## XGBoost Reduced
#-----------------------------------------------------------------------
# Parallel Processing settings
cl <- makePSOCKcluster(10)
registerDoParallel(cl)
#getDoParWorkers()
#-----------------------------------------------------------------------

set.seed(1989, sample.kind = "Rounding")

control_xgb1 <- trainControl(method= "cv", number = 4, p= 0.8, 
                             #verboseIter = TRUE, 
                             classProbs = TRUE, 
                             summaryFunction = twoClassSummary,
                             allowParallel = TRUE)

# Initial Tuning Grid for optimization
#tuning_xgb1 <- expand.grid(nrounds = c(150, 200, 250),
#                           max_depth =  c(10, 15, 20),
#                           eta = c(0.2, 0.3, 0.4),
#                           gamma = c(5 , 20 ),
#                           colsample_bytree = c(0.3, 0.5, 0.7, 0.9),
#                           min_child_weight = 2,
#                           subsample = c(0.7, 0.8, 0.9))

# Optimized parameters for Specificity metric
tuning_xgb1 <- expand.grid(nrounds = c(250),
                           max_depth =  15,
                           eta = c(0.3),
                           gamma = c(5),
                           colsample_bytree = c(0.5),
                           min_child_weight = 2,
                           subsample = c(0.8))
#-----------------------------------------------------------------------

set.seed(1989, sample.kind = "Rounding")

train_xgb1 <- train(x_train_f1, y_train_f1,
                    method = "xgbTree",
                    trControl = control_xgb1,
                    tuneGrid = tuning_xgb1,
                    metric = "Spec",
                    importance = TRUE,
                    allowParallel = TRUE)
#verbose = TRUE)

stopCluster(cl)
registerDoSEQ()

pred_xgb1<- predict(train_xgb1, x_test_f1)
cm_xgb1 <- confusionMatrix(pred_xgb1, y_test)
#-----------------------------------------------------------------------

results<-bind_rows(results,tibble(Method="Extreme Gradient Boosting",
                                  Dataset = "Reduced",
                                  Specificity = cm_xgb1$byClass["Specificity"],
                                  Sensitivity = cm_xgb1$byClass["Sensitivity"],
                                  FN = cm_xgb1$table[1,2], 
                                  FP = cm_xgb1$table[2,1],
                                  Cost = FN*500 +FP*10))
#-----------------------------------------------------------------------

## XGBoost Extended
#-----------------------------------------------------------------------
# Parallel Processing settings
cl <- makePSOCKcluster(10)
registerDoParallel(cl)
#getDoParWorkers()
#-----------------------------------------------------------------------

set.seed(1989, sample.kind = "Rounding")

control_xgb2 <- trainControl(method= "cv", number = 4, p= 0.8, 
                             #verboseIter = TRUE, 
                             classProbs = TRUE, 
                             summaryFunction = twoClassSummary,
                             allowParallel = TRUE)

# Initial Tuning Grid for optimization
#tuning_xgb2 <- expand.grid(nrounds = c(150, 200),
#                           max_depth = c(10, 15, 20),
#                           eta = c(0.2, 0.3, 0.4),
#                           gamma = 5,
#                           colsample_bytree = c(0.5, 0.7, 0.9),
#                           min_child_weight = c( 2),
#                           subsample = c( 0.7, 0.8))

# Optimized parameters for Specificity metric
tuning_xgb2 <- expand.grid(nrounds = c(200),
                           max_depth = c(10),
                           eta = c(0.3),
                           gamma = 5,
                           colsample_bytree = c(0.9),
                           min_child_weight =  2,
                           subsample = c(0.8))
#-----------------------------------------------------------------------

set.seed(1989, sample.kind = "Rounding")

train_xgb2 <- train(x_train_f2, y_train_f2,
                    method = "xgbTree",
                    trControl = control_xgb2,
                    tuneGrid = tuning_xgb2,
                    metric = "Spec",
                    importance = TRUE,
                    allowParallel = TRUE)
#verbose = TRUE)

stopCluster(cl)
registerDoSEQ()

pred_xgb2 <- predict(train_xgb2, x_test_f2)
cm_xgb2 <- confusionMatrix(pred_xgb2, y_test)
#-----------------------------------------------------------------------

results<-bind_rows(results,tibble(Method="Extreme Gradient Boosting", 
                                  Dataset = "Extended",
                                  Specificity = cm_xgb2$byClass["Specificity"],
                                  Sensitivity = cm_xgb2$byClass["Sensitivity"],
                                  FN = cm_xgb2$table[1,2], 
                                  FP = cm_xgb2$table[2,1],
                                  Cost = FN*500 +FP*10))
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
#results %>% knitr::kable(booktabs = T, linesep = "")%>%
#  kable_styling(position = "center")%>%
#  row_spec(c(2,4,6,8), hline_after = TRUE)
#-----------------------------------------------------------------------

##  VII. Ensemble
##########################################################

## Ensemble Reduced
#-----------------------------------------

ensemble1 <- cbind(glm = pred_glm1, 
                   lgb = pred_lgb1, 
                   rf = pred_rf1, 
                   xgb2 = pred_xgb1)

pred_en1 <- ifelse(rowMeans(ensemble1) >= 1.5, "pos", "neg")

pred_en1<-factor(pred_en1, levels= levels(y_test))
cm_en1 <- confusionMatrix(pred_en1, y_test)

results<-bind_rows(results,tibble(Method="Ensemble", 
                                  Dataset = "Reduced",
                                  Specificity = cm_en1$byClass["Specificity"],
                                  Sensitivity = cm_en1$byClass["Sensitivity"],
                                  FN = cm_en1$table[1,2], 
                                  FP = cm_en1$table[2,1],
                                  Cost = FN*500 +FP*10))

#----------------------------------------

## Ensemble Extended
#-----------------------------------------

ensemble2 <- cbind(glm = pred_glm2, 
                   lgb = pred_lgb2, 
                   rf = pred_rf2, 
                   xgb2 = pred_xgb2)

pred_en2 <- ifelse(rowMeans(ensemble2) >= 1.5, "pos", "neg")

pred_en2<-factor(pred_en2, levels= levels(y_test))
cm_en2 <- confusionMatrix(pred_en2, y_test)

results<-bind_rows(results,tibble(Method="Ensemble", 
                                  Dataset = "Extended",
                                  Specificity = cm_en2$byClass["Specificity"],
                                  Sensitivity = cm_en2$byClass["Sensitivity"],
                                  FN = cm_en2$table[1,2], 
                                  FP = cm_en2$table[2,1],
                                  Cost = FN*500 +FP*10))

#----------------------------------------

#-----------------------------------------------------------------------
#results %>% knitr::kable(booktabs = T, linesep = "")%>%
#  kable_styling(position = "center")%>%
#  row_spec(c(2,4,6,8,10), hline_after = TRUE)
#-----------------------------------------------------------------------

results
