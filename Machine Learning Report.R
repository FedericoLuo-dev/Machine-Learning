library(patchwork)
library(reshape2)
library(ROCR)
library(MASS)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(plyr)
library(VIM)
library(mice)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(Boruta)
library(car)
#install.packages("factorMerger")
devtools::install_github("ModelOriented/factorMerger", build_vignettes = FALSE)
library(factorMerger)
library(parsnip)
library(caretEnsemble)
library(caTools)
library(funModeling)
library(DALEX)
library(breakDown)
library(visdat)
set.seed(1)
options(scipen = 999)

table(train$repay_fail)

train <- read.csv("Anonymize_Loan_Default_data.csv", sep=",",dec = ".",stringsAsFactors=TRUE, na.strings=c("NA","NaN", "","n/a"))
head(train)
summary(train)
#CHECK OVERALL ####
table(train$repay_fail)
conteggio <- table(train$repay_fail)
train$repay_fail<-as.factor(ifelse(train$repay_fail==1,"Si","No"))
# Calcola la percentuale per ciascun valore
percentuale <- (conteggio / sum(conteggio)) * 100

# Crea un nuovo dataframe o un vettore con i risultati
risultati <- data.frame(Valore = names(conteggio), Conteggio = as.numeric(conteggio), Percentuale = percentuale)

# Stampa i risultati
print(risultati)

# MISSING VALUES ####
missingness<- aggr(train, col=c('navyblue','yellow'),numbers=TRUE, sortVars=TRUE,labels=names(train), cex.axis=.7,gap=2)
train2=train
train2$next_pymnt_d=NULL
train2$mths_since_last_delinq=NULL

missing = as.data.frame(sapply(train2, function(x)(sum(is.na(x)))))
names(missing) = "MISSING"
missing$PERC = (missing$MISSING / nrow(train2))*100
missing = missing[(missing$PERC != 0),]
missing

missingness = aggr(train2[, rownames(missing)],col=c('pink2','purple4'),numbers=TRUE,sortVars=TRUE,
                   labels=names(df),cex.axis=.7,gap=2)
which(is.na(train2$total_rec_int))

train2<-train2[-55,]
covdata <- train2[, c("int_rate", "term", "emp_length", "home_ownership", 
                      "annual_inc", "verification_status", "dti", 
                      "purpose", "total_acc", "revol_bal", "installment", 
                      "funded_amnt","total_pymnt")]
tempData = mice(covdata, m=1, maxit=10, meth='cart', seed=500)
train_imputed = complete(tempData,1) 
train_imputed = cbind(train_imputed, train2$repay_fail) 
colnames(train_imputed)
colnames(train_imputed)[14] = "repay_fail"
missingness = aggr(train_imputed,col=c('navyblue','yellow'),numbers=TRUE,sortVars=TRUE,
                   labels=names(df),cex.axis=.7,gap=2)
names(train_imputed)[14] = c("repay_fail")

#COLLINEARITA'####
numeric <- sapply(train_imputed, function(x) is.numeric(x))
numeric <-train_imputed[, numeric]
str(numeric)
R=cor(numeric)
R
correlatedPredictors = findCorrelation(R, cutoff = 0.95, names = TRUE)
correlatedPredictors #non abbiamo problemi di multicollinearity
#INSTALLMENT & FUNDED_AMNT SONO UN PROBLEMA  io penso di eliminare INSTALLMENT
train_imputed<-train_imputed[,-11]
summary(train_imputed)
#POTREI FARE LA DIFFERENZA DI FUNDED_AMNT E TOTAL PAYMNT COSI' LEVO LE DUE 
# VARIABILI
#Predittori con varianza zero e varianza vicino a zero#### 
nzv = nearZeroVar(train_selected, saveMetrics = TRUE)
nzv    #hanno varianza vicino allo zero capgain, caploss, country.
table(train2$repay_fail)



indice <- createDataPartition(train_imputed$repay_fail, p = 0.1, list = FALSE)
score_set <- train_imputed[indice, ]
write.csv(score_set, "score_set.csv", row.names = FALSE)
train_imputed <- train_imputed[-indice, ]
Trainindex = createDataPartition(y = train_imputed$repay_fail, p = .75, list = FALSE)
table(train_imputed$issue_d)
train = train_imputed[Trainindex,]
validation = train_imputed[-Trainindex,]

#MODEL SELECTION CON BORUTA ####
# CERCARE DI CAPIRE COME FARE LA SEPARAZIONE DEI DATI IN VALIDATION E ADDESTRAMENTO
# E IL DATASET DI SCORE
cvCtrl = trainControl(method = "cv", number=10, search="grid", classProbs = TRUE)
rpartTuneCvA = train(repay_fail ~ ., data = train_imputed, method = "rpart",tuneLength = 10,trControl = cvCtrl)

#sum(is.na(train_imputed))
rpartTuneCvA
getTrainPerf(rpartTuneCvA)

plot(varImp(object=rpartTuneCvA),main="train tuned - Variable Importance")
plot(rpartTuneCvA)

vi_t = as.data.frame(rpartTuneCvA$finalModel$variable.importance)
viname_t = row.names(vi_t)
head(viname_t)

#Random Forest
set.seed(123)  # Imposta un seme per la riproducibilità
sub_sample <- train[sample(1:nrow(train), 5000, replace = FALSE), ]
rfTune = train(repay_fail ~ ., data = sub_sample, method = "rf",
               tuneLength = 5,
               trControl = cvCtrl)

rfTune
getTrainPerf(rfTune)

plot(varImp(object=rfTune),main="train tuned - Variable Importance")
plot(rfTune)

vi_rf = data.frame(varImp(rfTune)[1])
vi_rf$var = row.names(vi_rf)
head(vi_rf)
viname_rf = vi_rf[,2]

#Boruta

boruta.train = Boruta(repay_fail ~., data = train_imputed, doTrace = 1)
plot(boruta.train, xlab = "features", xaxt = "n", ylab="MDI")
plot(boruta.train, las = 2, cex.axis = 0.7)

print(boruta.train)

boruta.metrics = attStats(boruta.train)
(boruta.metrics)
table(boruta.metrics$decision)

vi_bo = subset(boruta.metrics, decision == "Confirmed")
head(vi_bo)  
viname_bo = rownames(vi_bo)

viname_t
#viname_rf
viname_bo
#
selected = c("int_rate","term","emp_length","home_ownership","annual_inc",
             "verification_status","dti","purpose","total_acc","revol_bal",
             "funded_amnt","total_pymnt","repay_fail")
train_selected = train[,selected]
validation_selected = validation[,selected]

train_selected$repay_fail
#RIMOZIONE 
#train_selected<-train_selected[-11]
#validation_selected<-validation_selected[-11]
#CREAZIONE DEI MODELLI ####

table(train_selected$repay_fail)
#LOGISTICO####
metric <- "Sens"
control <- trainControl(method= "cv",number=10, summaryFunction = twoClassSummary, classProbs = TRUE ,savePrediction = TRUE)

glm=train(repay_fail~.,data=train_selected , method = "glm",
          trControl = control, tuneLength=5, trace=FALSE,preProcess=c("nzv","corr","scale"),metric=metric)
glm
confusionMatrix(glm)
getTrainPerf(glm)
#capire la separabilità e fixare questo problema
table(train$repay_fail)
confusionMatrix(glm)
confusionMatrix(predict(glm, newdata = train_selected), reference = train_selected$repay_fail, positive = "Si")


#NAIVE BAYES ####
ctrl =trainControl(method="cv", number = 10, classProbs = T,
                   summaryFunction=twoClassSummary)
naivebayes=train(repay_fail~.,data=train_selected,method = "naive_bayes",preProcess = c("corr", "nzv"),
                 trControl = ctrl, tuneLength=5, na.action = na.pass,
                 metric=metric) 
naivebayes
confusionMatrix(naivebayes)
confusionMatrix(predict(naivebayes, newdata = train_selected), reference = train_selected$repay_fail, positive = "Si")

getTrainPerf((naivebayes))

#LASSO####
ctrl =trainControl(method="cv", number = 10, classProbs = T,
                   summaryFunction=twoClassSummary)
grid = expand.grid(.alpha=1,.lambda=seq(0, 1, by = 0.01))
lasso=train(repay_fail~.,data=train_selected,method = "glmnet",
            trControl = ctrl, tuneLength=5, na.action = na.pass,
            tuneGrid=grid,metric=metric)
lasso
plot(lasso)
confusionMatrix(lasso)

#KNN ####
train_knn = train_selected
train_knn$term = as.integer(train_knn$term)
train_knn$emp_length = as.integer(train_knn$emp_length)
train_knn$home_ownership = as.integer(train_knn$home_ownership)
train_knn$verification_status = as.integer(train_knn$verification_status)
train_knn$purpose = as.integer(train_knn$purpose)
ctrl =trainControl(method="cv", number = 10, classProbs = T,
                   summaryFunction=twoClassSummary)
grid = expand.grid(k=seq(5,20,3))
knn=train(repay_fail~., data=train_knn,method = "knn",trControl = ctrl, tuneLength=5, na.action = na.pass,tuneGrid=grid, preProcess=c("scale","center"),metric=metric)
knn
plot(knn)
confusionMatrix(knn)
getTrainPerf(knn)
confusionMatrix(predict(knn, newdata = train_knn), reference = train_knn$repay_fail, positive = "Si")

#PLS ####
library(pls)
Control=trainControl(method= "cv",number=10, classProbs=TRUE,
                     summaryFunction=twoClassSummary)
pls=train(repay_fail~. , data=train_selected , method = "pls", 
          trControl = Control, tuneLength=5,metric=metric)
pls
plot(pls)
confusionMatrix(pls)
getTrainPerf(pls)
confusionMatrix(predict(pls, newdata = train_selected), reference = train_selected$repay_fail, positive = "Si")

#ALBERO####
cvCtrl <- trainControl(method = "cv", number=10, search="grid", classProbs = TRUE, summaryFunction=twoClassSummary)
tree <- train(repay_fail~. ,data=train_selected, 
              method = "rpart",
              tuneLength = 10,
              trControl = cvCtrl,metric=metric)
#purpose,total_payment ,int rate,term,annual_inc,home+verif+dti+purpose
tree
plot(tree)
confusionMatrix((tree))
confusionMatrix(predict(tree, newdata = train_selected), reference = train_selected$repay_fail, positive = "Si")




# RANDOM FOREST####
library(caret)
cvCtrl = trainControl(method = "cv", number=10, searc="grid",summaryFunction = twoClassSummary, 
                      classProbs = TRUE)

rf = train(repay_fail ~., data=train_selected,
           method = "rf", tuneLength = 5,
           metric="Sens",ntree = 100,
           trControl = cvCtrl,
           verbose = FALSE)

rf
confusionMatrix(rf)
getTrainPerf(rf)

table(train_selected)
#GRADIENT BOOSTING####
metric <- "Sens"   
control <- trainControl(method="cv", number=10, summaryFunction = twoClassSummary, classProbs = TRUE,savePrediction = TRUE)
gradient_boost <- train(repay_fail ~ ., data = train, method = "gbm", trControl = control, metric = metric, verbose=FALSE) 
plot(gradient_boost)
gradient_boost
confusionMatrix(gradient_boost)
getTrainPerf(gradient_boost)
confusionMatrix(predict(gradient_boost, newdata = train_selected), reference = train_selected$repay_fail, positive = "Si")

#cvCtrl = trainControl(method = "cv", number=10, searc="grid", 
#                      summaryFunction = twoClassSummary, 
#                      classProbs = TRUE)
#gbm_tune = expand.grid(
#  n.trees = 500,
# interaction.depth = 4,
# shrinkage = 0.1,
# n.minobsinnode = 10
#)

#gb = train(repay_fail ~., data=train_selected,
#          method = "gbm", tuneLength = 10,
#        metric="Sens", tuneGrid = gbm_tune,
#        trControl = cvCtrl)

#gb



#NEURAL NETWORKS####
cvCtrl = trainControl(method = "cv", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)
nn = train(repay_fail ~., data=train_selected,
           method = "nnet",
           preProcess = c("scale"), 
           tuneLength = 5, metric="Sens", trControl=cvCtrl, trace = TRUE,
           maxit = 200)
plot(nn)
nn
print(nn)
getTrainPerf(nn)
confusionMatrix(nn)

#ESEGUITO FINO A QUI####
# ROC ---------------------------------------------------------------------
predP=predict(gradient_boost,validation_selected, "prob")[,2]


library(pROC)
#LOGISTICO
glmpred_p = predict(glm, validation_selected, type = c("prob"))
y<-validation$repay_fail
glmpredR = prediction(glmpred_p[,"Si"], y)
roc_log = performance(glmpredR, measure = "tpr", x.measure = "fpr")
plot(roc_log)
abline(a=0, b= 1)


KNNPred_p = predict(knn, validation_selected, type = c("prob"))
knnpredR = prediction(KNNPred_p[,"Si"], y)
roc_knn = performance(knnpredR, measure = "tpr", x.measure = "fpr")
plot(roc_knn)
abline(a=0, b= 1)

lassoPred_p = predict(lasso, validation_selected, type = c("prob"))
lassoPredR = prediction(lassoPred_p[,"Si"], y)
roc_lasso = performance(lassoPredR, measure = "tpr", x.measure = "fpr")
plot(roc_lasso)
abline(a=0, b= 1)


plsPred_p = predict(pls, validation_selected, type = c("prob"))
plsPredR = prediction(plsPred_p[,"Si"], y)
roc_pls = performance(plsPredR, measure = "tpr", x.measure = "fpr")
plot(roc_pls)
abline(a=0, b= 1)

naivePred_p = predict(naivebayes, validation_selected, type = c("prob"))
naivePredR = prediction(naivePred_p[,"Si"], y)
roc_naive = performance(naivePredR, measure = "tpr", x.measure = "fpr")
plot(roc_naive)
abline(a=0, b= 1)


treePred_pruned_p = predict(tree, validation_selected, type = c("prob"))
treePredR = prediction(treePred_pruned_p[,"Si"], y)
roc_tree = performance(treePredR, measure = "tpr", x.measure = "fpr")
plot(roc_tree)
abline(a=0, b= 1)


gbPred_p = predict(gradient_boost, validation_selected, type = c("prob"))
gbPredR = prediction(gbPred_p[,"Si"], y)
roc_gb = performance(gbPredR, measure = "tpr", x.measure = "fpr")
plot(roc_gb)
abline(a=0, b= 1)


rfPred_p = predict(rf, validation_selected, type = c("prob"))
rfPredR = prediction(rfPred_p[,"Si"], y)
roc_rf = performance(rfPredR, measure = "tpr", x.measure = "fpr")
plot(roc_rf)
abline(a=0, b= 1)


nnPred_p = predict(nn, validation_selected, type = c("prob"))
nnPredR = prediction(nnPred_p[,"Si"], y)
roc_nn = performance(nnPredR, measure = "tpr", x.measure = "fpr")
plot(roc_nn)
abline(a=0, b= 1)


plot(roc_log, col = "dodgerblue", lwd = 2) 
par(new = TRUE)
plot(roc_gb, col = "darkorange", lwd = 2) 
par(new = TRUE)
plot(roc_rf, col = "green", lwd = 2) 
par(new = TRUE)
plot(roc_nn, col = "purple", lwd = 2) 
par(new = TRUE)
plot(roc_knn, col = "yellow4", lwd = 2)
par(new = TRUE)
plot(roc_tree, col = "red", lwd = 2) 
par(new = TRUE)
plot(roc_naive, col = "black", lwd = 2) 
par(new = TRUE)
plot(roc_pls, col = "brown3", lwd = 2) 
par(new = TRUE)
plot(roc_lasso, col = "darkgrey", lwd = 2) 
par(new = TRUE)

auc_log <- auc(roc_log)
auc_gb <- auc(roc_gb)
auc_rf <- auc(roc_rf)
auc_nn <- auc(roc_nn)
auc_knn <- auc(roc_knn)
auc_tree <- auc(roc_tree)
auc_naive <- auc(roc_naive)
auc_pls <- auc(roc_pls)
auc_lasso <- auc(roc_lasso)

legend("bottomright", legend=c("logit", "gb", "rf", "nn", "knn", "tree","naive","pls","lasso"),
       col=c("dodgerblue", "darkorange", "green", "purple", "yellow4", "red","black","brown3","darkgrey"),
       lty = 1, cex = 0.7, text.font=2, y.intersp=0.1, x.intersp=0.1, lwd = 3)

#PROBLEMA####
#HAI LE ROC ALTE COME UN PALO, 1) PROBLEMA SEPARABILITA' NON CREDO CHE HO GIà VISTO TABLE(...)
#MOLTO PROBABILMENTE SARA' UNA VARIABILE COLLINEARE FUNDED_AMNT . HO PROVATO A LEVARE EMP_LENGHT
# MA RIMANE UN PALO , QUINDI 90% FUNDED_AMNT


#LIFT CHARTS ####
copy = validation_selected
copy$gb = predict(gradient_boost, copy, type = c("prob"))[,"Si"]
gain_lift(data = copy, score='gb', target='repay_fail')


copy$glm = predict(glm, copy, type = c("prob"))[,"Si"]
gain_lift(data = copy, score='glm', target='repay_fail')


copy$nn = predict(nn, copy, type = c("prob"))[,"Si"]
gain_lift(data = copy, score='nn', target='repay_fail')



#predP <- predict(nn, validation_selected,type = "prob")
#head(predP)
#df=data.frame(cbind(validation_selected$repay_fail , predP))
#head(df)
#colnames(df)=c("repay_fail","ProbSi","ProbNo")
#head(df)
#df <- df[, c(1, 3)]
#head(df)


validation_selected$nn=predict(nn,validation_selected, "prob")[,2]
df=validation_selected
df$repay_fail
head(df)
df$repay_fail=ifelse(df$repay_fail=="Si","Si","No") # il nostro event c1 ? ora M
head(df)
df$ProbSi=validation_selected$nn
head(df$ProbSi)
head(df)




library(dplyr)
# for each threshold, find tp, tn, fp, fn and the sens=prop_true_M, spec=prop_true_R, precision=tp/(tp+fp)

thresholds <- seq(from = 0, to = 1, by = 0.01)
prop_table <- data.frame(threshold = thresholds, prop_true_Si = NA,  
                         prop_true_No = NA, true_Si = NA,  true_No = NA ,fn_Si=NA)
prop_table
for (threshold in thresholds) {
  pred <- ifelse(df$ProbSi > threshold, "Si", "No")  # be careful here!!!
  pred_t <- ifelse(pred == df$repay_fail, TRUE, FALSE)
  
  group <- data.frame(df, "pred" = pred_t) %>%
    group_by(repay_fail, pred) %>%
    dplyr::summarise(n = n())
  
  group_Si <- filter(group, repay_fail == "Si")
  
  true_Si=sum(filter(group_Si, pred == TRUE)$n)
  prop_Si <- sum(filter(group_Si, pred == TRUE)$n) / sum(group_Si$n)
  
  prop_table[prop_table$threshold == threshold, "prop_true_Si"] <- prop_Si
  prop_table[prop_table$threshold == threshold, "true_Si"] <- true_Si
  
  fn_Si=sum(filter(group_Si, pred == FALSE)$n)
  # true M predicted as R
  prop_table[prop_table$threshold == threshold, "fn_Si"] <- fn_Si
  
  
  group_No <- filter(group, repay_fail == "No")
  
  true_No=sum(filter(group_No, pred == TRUE)$n)
  prop_No <- sum(filter(group_No, pred == TRUE)$n) / sum(group_No$n)
  
  prop_table[prop_table$threshold == threshold, "prop_true_No"] <- prop_No
  prop_table[prop_table$threshold == threshold, "true_No"] <- true_No
  
}

head(prop_table, n=10)




prop_table$n=nrow(validation_selected)

# false positive (fp_M) by difference of   n and            tn,                 tp,         fn, 
prop_table$fp_Si=nrow(df)-prop_table$true_No-prop_table$true_Si-prop_table$fn_Si

# find accuracy
prop_table$acc=(prop_table$true_No+prop_table$true_Si)/nrow(df)

# find precision
prop_table$prec_Si=prop_table$true_Si/(prop_table$true_Si+prop_table$fp_Si)

# find F1 =2*(prec*sens)/(prec+sens)
# prop_true_M = sensitivity

prop_table$F1=2*(prop_table$prop_true_Si*prop_table$prec_Si)/(prop_table$prop_true_Si+prop_table$prec_Si)

# verify not having NA metrics at start or end of data 
tail(prop_table)
# we have typically some NA in the precision and F1 at the boundary..put,impute 1,0 respectively 

library(Hmisc)
#impute NA as 0, this occurs typically for precision
prop_table$prec_Si=impute(prop_table$prec_Si, 1)
prop_table$F1=impute(prop_table$F1, 0)
tail(prop_table)

colnames(prop_table)

# drop counts, PLOT only metrics
# drop counts, PLOT only metrics
prop_table2 = prop_table[,-c(4:8)] 
head(prop_table2)

# plot measures vs soglia##########
# before we must impile data vertically: one block for each measure
library(dplyr)
library(tidyr)

gathered=prop_table2 %>%
  gather(x, y, prop_true_Si:F1)

head(gathered)

# plot measures 
library(ggplot2)
gathered %>%
  ggplot(aes(x = threshold, y = y, color = x)) +
  geom_point() +
  geom_line() +
  scale_color_brewer(palette = "Set1") +
  labs(y = "measures",
       color = "Si: event\nNo: nonevent")
# zoom
gathered %>%
  ggplot(aes(x = threshold, y = y, color = x)) +
  geom_point() +
  geom_line() +
  scale_color_brewer(palette = "Set1") +
  labs(y = "measures",
       color = "Si: event\n No: nonevent") +
  coord_cartesian(xlim = c(0.2, 0.3))


library(ggplot2)

# Assicurati che le colonne siano numeriche
df$ProbSi <- as.numeric(df$ProbSi)

# Crea l'istogramma
ggplot(df, aes(x = ProbSi, fill = repay_fail)) +
  geom_histogram(binwidth = 0.02, position = "identity", alpha = 0.7) +
  labs(x = "Probabilità previste", y = "Frequenza", fill = "Classe") +
  theme_minimal()



#SCORE####
score_set$gd=predict(gradient_boost,score_set, "prob")[,2]
pred <- ifelse(score_set$gd > 0.05, "Si", "No")
head(pred)
head(score_set$repay_fail)
score_set$pred=pred
head(score_set$pred)
#train$repay_fail<-as.factor(ifelse(train$repay_fail==1,"Si","No"))
score_set$pred <- as.factor(ifelse(score_set$pred=="Si","Si","No"))
levels(score_set$pred)
confusionMatrix(score_set$pred, score_set$repay_fail,positive="Si")
score_set <- subset(score_set, select = -nn)
levels(score_set$repay_fail)
levels(pred)

