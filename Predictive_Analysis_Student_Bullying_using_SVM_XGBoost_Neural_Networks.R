
df <- read.csv('D:/course/CS699/project/project_dataset.csv')
set.seed(123)
which(is.na(df))
# no NA values found

cor_matrix <- data.frame(cor(df))
cor_matrix
library(ggcorrplot)
ggcorrplot(cor_matrix, 
           lab = TRUE,               
           lab_size = 0.1,           
           tl.cex = 1,               
           tl.srt = 45              
)
#find y name
last_col_name <- tail(names(df), 1)
print(last_col_name)
#o_bullied
# 假设你的结果变量列名是 "y"
y_cor <- cor_matrix[,"y", drop = FALSE] # drop = FALSE 保持结果为data.frame格式
# 找出相关性大于0.5的特征
high_cor_with_y <- y_cor[abs(y_cor[, "o_bullied"]) > 0.08, , drop = FALSE]
print(high_cor_with_y)
#            o_bullied
#V3034      -0.08455973 SOMETHING STOLEN OR ATTEMPT
#V3035      -0.08455167 NO. TIMES SOMETHING STOLEN OR ATTEMPT
#V3040      -0.13766259 ATTACK, THREAT, THEFT: LOCATION CUES
#V3041      -0.13766667 NO. TIMES ATTACK, LOCATION CUES
#V3042      -0.14879908 ATTACK, THREAT: WEAPON, ATTACK CUES 
#V3043      -0.14875719 NO. TIMES ATTACK, WEAPON CUES
#V3044      -0.11061339 STOLEN, ATTACK, THREAT: OFFENDER KNOWN
#VS0002      0.16748583 NUMBER OF INCIDENTS 
#VS0006      0.16748583 THE TOTAL NUMBER OF INCIDENTS FOR THAT PERSON.
#VS0010     -0.08917020 RESPONDENT AGE (ALLOCATED) 
#VS0131     -0.13816337 HAVE YOU ACTUALLY SEEN ANOTHER STUDENT WITH A GUN AT 188 SCHOOL DURING THIS SCHOOL YEAR?
#V4526AA_1  -0.17398776 SUSPECT INCIDENT JUST DISCUSSED WAS HATE CRIME OR CRIME 420 OF PREJUDICE OR BIGOTRY (START 2010 Q1)
#V4526H3A_1 -0.17279949 ARE YOU DEAF OR DO YOU HAVE SERIOUS DIFFICULTY HEARING? 
#V4526H3B_1 -0.17266292 ARE YOU BLIND OR DO YOU HAVE SERIOUS DIFFICULTY SEEING 429 EVEN WHEN WEARING GLASSES
#V4526H4_1  -0.17271242 LONG LAST CONDTN: LIMITS PHYSICAL ACTIVITIES
#V4526H5_1  -0.17411072 CONDTN 6 MO+ DIFFICULT: LEARN, REMEMBER, CONCENTRATE
#V4526H6_1  -0.17257575 CONDTN 6 MO+ DIFFICULT: DRESSING, BATHING, GET AROUND HOME
#V4526H7_1  -0.17257575 CONDTN 6 MO+ DIFFICULT: GO OUTSIDE HOME TO SHOP OR DR 431 OFFICE
threshold <- 0.5
high_cor_features <- which(abs(cor_matrix) > threshold, arr.ind = TRUE)
high_cor_features <- high_cor_features[high_cor_features[, 1] != high_cor_features[, 2], ]
feature_names <- rownames(cor_matrix)
for (i in seq_len(nrow(high_cor_features))) {
     row <- high_cor_features[i, "row"]
     col <- high_cor_features[i, "col"]
     cat(sprintf("Feature 1: %s - Feature 2: %s - Correlation: %.2f\n", 
                                   feature_names[row], feature_names[col], cor_matrix[row, col]))
 }
#Feature 1: V3034 - Feature 2: V3035 - Correlation: 1.00
#Feature 1: V3040 - Feature 2: V3041 - Correlation: 1.00
#Feature 1: V3042 - Feature 2: V3043 - Correlation: 1.00
#Feature 1: V3044 - Feature 2: V3045 - Correlation: 1.00
#Feature 1: V3034 - Feature 2: VS0002 - Correlation: -0.57
#Feature 1: VS0006 - Feature 2: VS0002 - Correlation: 1.00
#Feature 1: V4526AA_1 - Feature 2: VS0002 - Correlation: -0.87
#Feature 1: V4526H3A_1 - Feature 2: VS0002 - Correlation: -0.87
#Feature 1: V4526H3B_1 - Feature 2: VS0002 - Correlation: -0.87
#Feature 1: V4526H4_1 - Feature 2: VS0002 - Correlation: -0.88
#Feature 1: V4526H5_1 - Feature 2: VS0002 - Correlation: -0.88
#Feature 1: V4526H6_1 - Feature 2: VS0002 - Correlation: -0.88
#Feature 1: V4526H7_1 - Feature 2: VS0002 - Correlation: -0.88
#AFTER I FILTER SOME HIGH-CORRELATED FEATURES
#
#自己选VS0019, VS0043, VS0046, VS0058, VS0070, VS0103, VS0126, V2026, V2031, V2047, V2076, V2034, V3039, V3045, V3047, V3054, V3074, V3035, V3041, V3043, V3045, VS0002, VS0010, VS0131
df$o_bullied <- as.factor(df$o_bullied)
levels(df$o_bullied) <- c("No","Yes")
library(lattice)
library(caret)
train_control <- trainControl(method = "repeatedcv", number = 10,repeats = 5,classProbs = TRUE)
temp = c()
f1 = c()
set.seed(123)
library(pROC)
df_0 = df[,c("VS0019", "VS0043", "VS0046", "VS0058", "V2026", "V2034", "V2076", "V2047", "V3035", "V3041", "V3043", "V3045", "V3047", "V3054", "V3074", "o_bullied")]




#1 boruta
install.packages("Boruta")
library(Boruta)
boruta_output <- Boruta(df$o_bullied ~ ., data=na.omit(df), doTrace=0)  
boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(boruta_signif)  
#"V2038" "V2042CAT" "V2047" "V2050" "V3020" "V3024" "V3034" "V3035" "V3040" "V3041" "V3042" "V3043" "V3044" "V3045" "V3071" "V3072" "VS0002" "VS0006" "VS0010" "VS0017"    
#"VS0022" "VS0023" "VS0027" "VS0031" "VS0046" "VS0047" "VS0048" "VS0049" "VS0050" "VS0051" "VS0052" "VS0053" "VS0054" "VS0055" "VS0146" "VS0147" "VS0148" "VS0149" "VS0150" "VS0151"    
#"VS0057" "VS0153" "VS0154" "VS0155" "VS0058" "VS0059" "VS0060" "VS0061" "VS0062" "VS0063" "VS0064" "VS0065" "VS0066" "VS0067" "VS0068" "VS0069" "VS0070" "VS0112" "VS0113" "VS0114"    
#"VS0115" "VS0116" "VS0117" "VS0118" "VS0119" "VS0120" "VS0157" "VS0121" "VS0122" "VS0123" "VS0124" "VS0125" "VS0126" "VS0129" "VS0130" "VS0131" "VS0132" "VS0133" "VS0135" "VS0136"    
#"VS0137" "VS0138" "V4526AA_1" "V4526H3A_1" "V4526H3B_1" "V4526H4_1"  "V4526H5_1"  "V4526H6_1"  "V4526H7_1"  "o_bullied" 
roughFixMod <- TentativeRoughFix(boruta_output)
boruta_signif <- getSelectedAttributes(roughFixMod)
print(boruta_signif)
#"V3020" "V3034" "V3035" "V3040" "V3041" "V3042" "V3043" "V3044" "V3045" "V3071" "VS0002" "VS0006" "VS0010" "VS0017" "VS0022" "VS0023" "VS0027" "VS0046" "VS0047" "VS0048"    
#"VS0049" "VS0050" "VS0051" "VS0052" "VS0053" "VS0054" "VS0055" "VS0146" "VS0148" "VS0149" "VS0150" "VS0151" "VS0057" "VS0153" "VS0154" "VS0155" "VS0058" "VS0059" "VS0060" "VS0061"    
#"VS0062" "VS0063" "VS0064" "VS0065" "VS0066" "VS0067" "VS0069" "VS0070" "VS0112" "VS0113" "VS0114" "VS0115" "VS0116" "VS0117" "VS0118" "VS0119" "VS0120" "VS0157" "VS0121" "VS0122"    
#"VS0123" "VS0124" "VS0125" "VS0126"  "VS0130" "VS0131" "VS0133" "VS0135" "VS0136" "VS0137" "V4526AA_1"  "V4526H3A_1" "V4526H3B_1" "V4526H4_1"  "V4526H5_1"  "V4526H6_1"  "V4526H7_1"  "o_bullied" 
imps <- attStats(roughFixMod)
imps2 = imps[imps$decision != 'Rejected', c('meanImp', 'decision')]
head(imps2[order(-imps2$meanImp), ])  # descending sort
#            meanImp  decision
#o_bullied 73.497781 Confirmed
#VS0124    11.014375 Confirmed
#VS0070    10.699580 Confirmed
#VS0157    10.678672 Confirmed
#VS0069    10.634956 Confirmed
#VS0046     9.855043 Confirmed
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance(boruta)") 
#"VS0135" "VS0054""VS0047" "VS0155" "V4526H7_1""V4526H3A_1""VS0057""VS0048""V3034""VS0121""VS0010""V3020""VS0131""VS0112""VS0117""VS0123""VS0157"
df_1 = df[,c("VS0135","VS0054","VS0047","VS0155","V4526H7_1","V4526H3A_1","VS0057","VS0048","V3034","VS0121","VS0010","V3020","VS0131","VS0112","VS0117","VS0123","VS0157","o_bullied")]
set.seed(123)
library(rsample)
split <- initial_split(df_1, prop = 0.66, strata = o_bullied)
train <- training(split)
test <- testing(split)

#support vector machine
library(caret)

result <- train(o_bullied ~ .,data = train, trControl = train_control,
                method = "svmLinear")
pred <- predict(result, test)
test_labels <- test$o_bullied
positive_class_label <- 'Yes'
cm0 <- confusionMatrix(pred, test$o_bullied)
cm0
#Confusion Matrix and Statistics

#Reference
#Prediction   No  Yes
#No  1280  352
#Yes   18   33

#Accuracy : 0.7802          
#95% CI : (0.7596, 0.7997)
#No Information Rate : 0.7712          
#P-Value [Acc > NIR] : 0.2005          

#Kappa : 0.1034          

#Mcnemar's Test P-Value : <2e-16          
                                          
#            Sensitivity : 0.98613         
#            Specificity : 0.08571         
#         Pos Pred Value : 0.78431         
#         Neg Pred Value : 0.64706         
#             Prevalence : 0.77124         
#         Detection Rate : 0.76055         
#   Detection Prevalence : 0.96970         
#      Balanced Accuracy : 0.53592         
                                          
#       'Positive' Class : No 
cm1 <- confusionMatrix(pred, test_labels, positive = positive_class_label)
cm1
#Confusion Matrix and Statistics

#Reference
#Prediction   No  Yes
#No  1278  340
#Yes   20   45

#Accuracy : 0.7861          
#95% CI : (0.7657, 0.8055)
#No Information Rate : 0.7712          
#P-Value [Acc > NIR] : 0.07673         

#Kappa : 0.1434          

#Mcnemar's Test P-Value : < 2e-16         
                                          
#            Sensitivity : 0.11688         
#            Specificity : 0.98459         
#         Pos Pred Value : 0.69231         
#         Neg Pred Value : 0.78986         
#             Prevalence : 0.22876         
#         Detection Rate : 0.02674         
#   Detection Prevalence : 0.03862         
#      Balanced Accuracy : 0.55074         
                                          
#       'Positive' Class : Yes   
library(caret)
library(pROC)
train_control <- trainControl(
     method = "cv",
     number = 10,
     summaryFunction = twoClassSummary,
     classProbs = TRUE, 
     savePredictions = TRUE
 )
set.seed(123)
svm_model <- train(
     o_bullied ~ .,
     data = train,
     method = "svmLinear",
     trControl = train_control,
     metric = "ROC"
 )
predictions <- predict(svm_model, test, type = "prob")
roc_obj <- roc(test_labels, predictions[, positive_class_label])
print(auc(roc_obj))
#Area under the curve: 0.6502positive的roc,negative是1-0.6502

#模型的整体Accuracy
temp = c(temp,cm$overall["Accuracy"])
temp
#0.7801545 
f1 = c(f1,cm$byClass["F1"])
f1
#F1 
#0.8737201

#naive bayes
result <- train(o_bullied ~ .,data = train, trControl = train_control,
                method = "naive_bayes")
pred <- predict(result, test)
pred <- predict(result, test, type="prob")
cm <- confusionMatrix(pred, test$o_bullied)
cm
#Confusion Matrix and Statistics

#Reference
#Prediction   No  Yes
#No     1    1
#Yes 1297  384

#Accuracy : 0.2288          
#95% CI : (0.2089, 0.2496)
#No Information Rate : 0.7712          
#P-Value [Acc > NIR] : 1               

#Kappa : -8e-04          

#Mcnemar's Test P-Value : <2e-16          
                                          
#            Sensitivity : 0.0007704       
#            Specificity : 0.9974026       
#         Pos Pred Value : 0.5000000       
#         Neg Pred Value : 0.2284355       
#             Prevalence : 0.7712418       
#         Detection Rate : 0.0005942       
#   Detection Prevalence : 0.0011884       
#      Balanced Accuracy : 0.4990865       
                                          
#       'Positive' Class : No           
roc_obj <- roc(test$o_bullied, pred$Yes)
auc_value <- auc(roc_obj)
print(auc_value)
#Area under the curve: 0.6109

temp = c(temp,cm$overall["Accuracy"])
temp
#Accuracy  Accuracy 
#0.7801545 0.2287582 
f1 = c(f1,cm$byClass["F1"])
#f1
#         F1          F1 
#0.873720137 0.001538462 

#xgboost
result <- train(o_bullied ~ .,data = train, trControl = train_control,
                method = "xgbLinear")
pred <- predict(result, test)
pred <- predict(result, test, type="prob")
cm <- confusionMatrix(pred, test$o_bullied)
cm
#Confusion Matrix and Statistics

#Reference
#Prediction   No  Yes
#No  1229  302
#Yes   69   83

#Accuracy : 0.7796         
#95% CI : (0.759, 0.7992)
#No Information Rate : 0.7712         
#P-Value [Acc > NIR] : 0.2173         

#Kappa : 0.2063         

#Mcnemar's Test P-Value : <2e-16         
                                         
#            Sensitivity : 0.9468         
#            Specificity : 0.2156         
#         Pos Pred Value : 0.8027         
#         Neg Pred Value : 0.5461         
#             Prevalence : 0.7712         
#         Detection Rate : 0.7302         
#   Detection Prevalence : 0.9097         
#      Balanced Accuracy : 0.5812         
                                         
#       'Positive' Class : No 
roc_obj <- roc(test$o_bullied, pred$Yes)
auc_value <- auc(roc_obj)
print(auc_value)
#Area under the curve: 0.6717

temp = c(temp,cm$overall["Accuracy"])
temp
#Accuracy  Accuracy  Accuracy 
#0.7801545 0.2287582 0.7795603 
f1 = c(f1,cm$byClass["F1"])
f1
#         F1          F1          F1 
#0.873720137 0.001538462 0.868858254

#neural net
nnetGrid <- expand.grid(size = 1:10, decay = c(0, .1, 1, 2))
ncol(train)
#18
result <- train(x = train[,-18], y = train$o_bullied,method = "nnet",tuneGrid = nnetGrid,
                trace = FALSE,maxit = 100,MaxNWts = 1000,trControl = train_control)
pred <- predict(result, test)
pred <- predict(result, test, type = "prob")
roc_obj <- roc(response = test$o_bullied, predictor = pred$Yes)
auc_value <- auc(roc_obj)
print(auc_value)
#Area under the curve: 0.6832

cm <- confusionMatrix(pred, test$o_bullied)
cm
#Confusion Matrix and Statistics

#Reference
#Prediction   No  Yes
#No  1276  332
#Yes   22   53

#Accuracy : 0.7897          
#95% CI : (0.7694, 0.8089)
#No Information Rate : 0.7712          
#P-Value [Acc > NIR] : 0.0374          

#Kappa : 0.1684          

#Mcnemar's Test P-Value : <2e-16          
                                          
#            Sensitivity : 0.9831          
#            Specificity : 0.1377          
#         Pos Pred Value : 0.7935          
#         Neg Pred Value : 0.7067          
#             Prevalence : 0.7712          
#         Detection Rate : 0.7582          
#   Detection Prevalence : 0.9554          
#      Balanced Accuracy : 0.5604          
                                          
#       'Positive' Class : No      
temp = c(temp,cm$overall["Accuracy"])
temp 
#Accuracy  Accuracy  Accuracy  Accuracy 
#0.7801545 0.2287582 0.7795603 0.7896613 
f1 = c(f1,cm$byClass["F1"])
f1
#         F1          F1          F1          F1 
#0.873720137 0.001538462 0.868858254 0.878183070 

# random forest
mtryValues <- seq(2, ncol(df_1)-1, by = 1)
result <- train(x = train[, -18], y = train$o_bullied, method = "rf",ntree = 500,
                tuneGrid = data.frame(mtry = mtryValues),importance = TRUE,metric = "ROC",
                trControl = train_control)
pred <- predict(result, test)
pred <- predict(result, test, type = "prob")

cm <- confusionMatrix(pred, test$o_bullied)
cm
#Confusion Matrix and Statistics

#Reference
#Prediction   No  Yes
#No  1262  320
#Yes   36   65

#Accuracy : 0.7885          
#95% CI : (0.7682, 0.8078)
#No Information Rate : 0.7712          
#P-Value [Acc > NIR] : 0.04811         

#Kappa : 0.1905          

#Mcnemar's Test P-Value : < 2e-16         
                                          
#            Sensitivity : 0.9723          
#            Specificity : 0.1688          
#         Pos Pred Value : 0.7977          
#         Neg Pred Value : 0.6436          
#             Prevalence : 0.7712          
#         Detection Rate : 0.7499          
#   Detection Prevalence : 0.9400          
#      Balanced Accuracy : 0.5705          
                                          
#       'Positive' Class : No              
roc_obj <- roc(response = test$o_bullied, predictor = pred[,2])
auc_value <- auc(roc_obj)
print(auc_value)
#Area under the curve: 0.6716

temp = c(temp,cm$overall["Accuracy"])
temp
#Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
#0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 
f1 = c(f1,cm$byClass["F1"])
f1
#         F1          F1          F1          F1          F1 
#0.873720137 0.001538462 0.868858254 0.878183070 0.876388889

#2 lasso
set.seed(123)
install.packages("glmnet")
library(glmnet)
x = as.matrix(df[,1:203])
y = as.matrix(df[,204])
cv.lasso <- cv.glmnet(x, y, family='binomial', alpha=1, parallel=TRUE, standardize=TRUE, type.measure='auc')
plot(cv.lasso)
df_coef <- round(as.matrix(coef(cv.lasso, s=cv.lasso$lambda.min)), 2)
df_coef_abs <- abs(df_coef)
threshold <- 0.2
df_coef_significant <- df_coef[df_coef_abs[, 1] > threshold, ]
print(df_coef_significant)
#(Intercept) V2025A      V2025B       V2050       V2119    V3014CAT       V3015       V3034       V3040       V3042       V3044 
#31.61        0.23        0.36        0.28       -0.22       -0.27       -0.32       -0.22       -0.79       -2.02       -2.05 
#V3046       V3054       V3065       V3067      VS0007      VS0010      VS0031      VS0053      VS0055      VS0150      VS0059 
#-3.97       -1.02       -0.27       -0.25        0.26       -0.21       -0.43        0.29       -0.28       -0.30       -0.33 
#VS0067      VS0115      VS0117      VS0157      VS0123      VS0124 
#-0.23       -0.36       -0.48       -1.03       -0.45        0.73 
df_2 = df[,c("V2025A","V2025B","V2050","V2119","V3014CAT","V3015","V3034","V3040","V3042","V3044","V3046","V3054","V3065","V3067","VS0007","VS0010","VS0031","VS0053","VS0055","VS0150","VS0059","VS0067","VS0115","VS0117","VS0157","VS0123","VS0124","o_bullied")]
set.seed(123)
split <- initial_split(df_2, prop = 0.66, strata = o_bullied)
train <- training(split)
test <- testing(split)

#support vector machine
result <- train(o_bullied ~ ., data = train, trControl = train_control, method = "svmLinear")
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
#Confusion Matrix and Statistics

#Reference
#Prediction   No  Yes
#No  1283  333
#Yes   15   52

#Accuracy : 0.7932          
#95% CI : (0.7731, 0.8123)
#No Information Rate : 0.7712          
#P-Value [Acc > NIR] : 0.01628         

#Kappa : 0.1741          

#Mcnemar's Test P-Value : < 2e-16         
                                          
#            Sensitivity : 0.9884          
#            Specificity : 0.1351          
#         Pos Pred Value : 0.7939          
#         Neg Pred Value : 0.7761          
#             Prevalence : 0.7712          
#         Detection Rate : 0.7623          
#   Detection Prevalence : 0.9602          
#      Balanced Accuracy : 0.5618          
                                          
#       'Positive' Class : No  
pred_probs <- predict(result, test, type = "prob")[, "Yes"]
roc_curve <- roc(test$o_bullied, pred_probs)
auc_value <- auc(roc_curve)
print(auc_value)
#Area under the curve: 0.6393
#模型准确率
temp = c(temp,cm$overall["Accuracy"])
temp
#Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
#0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 
f1 = c(f1,cm$byClass["F1"])
f1
#         F1          F1          F1          F1          F1          F1 
#0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 

#naive bayes
result <- train(o_bullied ~ .,data = train, trControl = train_control,
                method = "naive_bayes")
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
#Confusion Matrix and Statistics

#Reference
#Prediction   No  Yes
#No  1296  370
#Yes    2   15

#Accuracy : 0.779           
#95% CI : (0.7584, 0.7986)
#No Information Rate : 0.7712          
#P-Value [Acc > NIR] : 0.2349          

#Kappa : 0.0564          

#Mcnemar's Test P-Value : <2e-16          
                                          
#            Sensitivity : 0.99846         
#            Specificity : 0.03896         
#         Pos Pred Value : 0.77791         
#         Neg Pred Value : 0.88235         
#             Prevalence : 0.77124         
#         Detection Rate : 0.77005         
#   Detection Prevalence : 0.98990         
#      Balanced Accuracy : 0.51871         
                                          
#       'Positive' Class : No   
temp = c(temp,cm$overall["Accuracy"])
temp
#Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
#0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 0.7789661 
f1 = c(f1,cm$byClass["F1"])
f1
#         F1          F1          F1          F1          F1          F1          F1 
#0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 0.874493927 
pred_probs <- predict(result, test, type = "prob")[, "Yes"]
roc_curve <- roc(test$o_bullied, pred_probs)
auc_value <- auc(roc_curve)
print(auc_value)
#Area under the curve: 0.6587
coords(roc_curve, "best", ret = c("sensitivity", "specificity"))
#    sensitivity specificity
#1   0.5662338   0.6895223

#xgboost
result <- train(o_bullied ~ .,data = train, trControl = train_control,
                method = "xgbLinear")
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
#Confusion Matrix and Statistics

#Reference
#Prediction   No  Yes
#No  1236  291
#Yes   62   94

#Accuracy : 0.7903        
#95% CI : (0.77, 0.8095)
#No Information Rate : 0.7712        
#P-Value [Acc > NIR] : 0.03282       

#Kappa : 0.2483        

#Mcnemar's Test P-Value : < 2e-16       
                                        
#            Sensitivity : 0.9522        
#            Specificity : 0.2442        
#         Pos Pred Value : 0.8094        
#         Neg Pred Value : 0.6026        
#             Prevalence : 0.7712        
#         Detection Rate : 0.7344        
#   Detection Prevalence : 0.9073        
#      Balanced Accuracy : 0.5982        
                                        
#       'Positive' Class : No    
temp = c(temp,cm$overall["Accuracy"])
temp
#Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
#0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 0.7789661 0.7902555 
f1 = c(f1,cm$byClass["F1"])
f1
#F1          F1          F1          F1          F1          F1          F1          F1 
#0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 0.874493927 0.875044248 
pred_probs <- predict(result, test, type = "prob")[, "Yes"]
roc_curve <- roc(test$o_bullied, pred_probs)
auc_value <- auc(roc_curve)
print(auc_value)
#Area under the curve: 0.709
coords_optimal <- coords(roc_curve, "best", ret = c("sensitivity", "specificity"))
print(coords_optimal)
#sensitivity specificity
#1   0.5558442   0.7580894

#neural net
nnetGrid <- expand.grid(size = 1:10, decay = c(0, .1, 1, 2))
result <- train(x = train[,-28], y = train$o_bullied,method = "nnet",tuneGrid = nnetGrid,
                trace = FALSE,maxit = 100,MaxNWts = 1000,trControl = train_control)
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
#Confusion Matrix and Statistics

#Reference
#Prediction   No  Yes
#No  1264  288
#Yes   34   97

#Accuracy : 0.8087          
#95% CI : (0.7891, 0.8272)
#No Information Rate : 0.7712          
#P-Value [Acc > NIR] : 0.0001092       

#Kappa : 0.294           

#Mcnemar's Test P-Value : < 2.2e-16       
                                          
#            Sensitivity : 0.9738          
#            Specificity : 0.2519          
#         Pos Pred Value : 0.8144          
#         Neg Pred Value : 0.7405          
#             Prevalence : 0.7712          
#         Detection Rate : 0.7510          
#   Detection Prevalence : 0.9222          
#      Balanced Accuracy : 0.6129          
                                          
#       'Positive' Class : No  
temp = c(temp,cm$overall["Accuracy"])
temp
#Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
#0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 0.7789661 0.7902555 0.8086750 
f1 = c(f1,cm$byClass["F1"])
f1
#         F1          F1          F1          F1          F1          F1          F1          F1          F1 
#0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 0.874493927 0.875044248 0.887017544 
pred_probs <- predict(result, test, type = "prob")
roc_curve <- roc(test$o_bullied, pred_probs[,2])
auc_value <- auc(roc_curve)
print(paste("AUC for nnet model:", auc_value))
#"AUC for nnet model: 0.717631521021352"
coords_optimal <- coords(roc_curve, "best", ret = c("sensitivity", "specificity"))
print(coords_optimal)
#   sensitivity specificity
#1   0.6753247   0.6355932

# random forest
train_control <- trainControl(method = "cv", number = 10, summaryFunction = twoClassSummary, classProbs = TRUE, savePredictions = TRUE)
mtryValues <- seq(2, ncol(df_2)-1, by = 1)
result <- train(x = train[, -28], y = train$o_bullied, method = "rf",ntree = 500,
                tuneGrid = data.frame(mtry = mtryValues),importance = TRUE,metric = "ROC",
                trControl = train_control)
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
#Confusion Matrix and Statistics

#Reference
#Prediction   No  Yes
#No  1267  298
#Yes   31   87

#Accuracy : 0.8045          
#95% CI : (0.7847, 0.8232)
#No Information Rate : 0.7712          
#P-Value [Acc > NIR] : 0.0005301       

#Kappa : 0.2673          

#Mcnemar's Test P-Value : < 2.2e-16       
                                          
#            Sensitivity : 0.9761          
#            Specificity : 0.2260          
#         Pos Pred Value : 0.8096          
#         Neg Pred Value : 0.7373          
#             Prevalence : 0.7712          
#         Detection Rate : 0.7528          
#   Detection Prevalence : 0.9299          
#      Balanced Accuracy : 0.6010          
                                          
#       'Positive' Class : No      
temp = c(temp,cm$overall["Accuracy"])
temp
#Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
#0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 0.7789661 0.7902555 0.8086750 0.8045157
f1 = c(f1,cm$byClass["F1"])
f1
#F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
#0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 0.874493927 0.875044248 0.887017544 0.885085575 
pred_probs <- predict(result, test, type = "prob")
roc_curve <- roc(test$o_bullied, pred_probs[,2])
auc_value <- auc(roc_curve)
print(paste("AUC for Random Forest model:", auc_value))
#"AUC for Random Forest model: 0.698609248994457"
coords_optimal <- coords(roc_curve, "best", ret = c("sensitivity", "specificity"))
print(coords_optimal)
#   sensitivity specificity
#1   0.5454545   0.7734977

#3 variable importance
set.seed(123)
rPartMod <- train(o_bullied ~ ., data=df, method="rpart")
rpartImp <- varImp(rPartMod)
print(rpartImp)
df_3 = df[,c("VS0046","VS0070","VS0069","VS0112","VS0124","VS0157","VS0115","VS0002","VS0006","V4526AA_1","VS0116","V3042","V3043","VS0148","VS0061","VS0023","V3024","V3012","V3048","o_bullied")]
set.seed(123)
split <- initial_split(df_3, prop = 0.66, strata = o_bullied)
train <- training(split)
test <- testing(split)

#support vector machine
# Pre-process your data
preProcValues <- preProcess(train[, -ncol(train)], method = c("center", "scale"))
train_scaled <- predict(preProcValues, train[, -ncol(train)])
train_scaled$o_bullied <- train$o_bullied  # Add the outcome variable back to the scaled data

# Define the control parameters for the train function
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid")

# Define a tuning grid for svmLinear (only need the 'C' parameter)
tune_grid <- expand.grid(C = 10^(-2:2))  # You can change the range based on your problem

# Train the model using the 'svmLinear' method with the defined tuning grid
set.seed(123)
result <- train(o_bullied ~ ., data = train_scaled, method = "svmLinear",
                trControl = train_control, tuneLength = 5, preProcess = c("center", "scale"),
                tuneGrid = tune_grid)
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
#Confusion Matrix and Statistics

#Reference
#Prediction   No  Yes
#No  1298  385
#Yes    0    0

#Accuracy : 0.7712          
#95% CI : (0.7504, 0.7911)
#No Information Rate : 0.7712          
#P-Value [Acc > NIR] : 0.5137          

#Kappa : 0               

#Mcnemar's Test P-Value : <2e-16          
                                          
#            Sensitivity : 1.0000          
#            Specificity : 0.0000          
#         Pos Pred Value : 0.7712          
#        Neg Pred Value :    NaN          
#             Prevalence : 0.7712          
#         Detection Rate : 0.7712          
#   Detection Prevalence : 1.0000          
#      Balanced Accuracy : 0.5000          
                                          
#       'Positive' Class : No 
# Define a tuning grid for svmRadial
tune_grid <- expand.grid(C = 10^(-2:2), sigma = 10^(-2:2))  # Adjust the ranges as necessary

# Make sure the response variable is a factor
train$o_bullied <- factor(train$o_bullied, levels = c("No", "Yes"))

# Update train_control to save the best tuning parameters
train_control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3,
                              search = "grid",
                              classProbs = TRUE, # Request class probabilities
                              summaryFunction = twoClassSummary)  # twoClassSummary is needed for ROC metric

# Train the model using the 'svmRadial' method with the new tuning grid
set.seed(123)
result <- train(o_bullied ~ ., data = train,
                method = "svmRadial",
                trControl = train_control,
                preProcess = c("center", "scale"),
                tuneGrid = tune_grid,
                metric = "ROC")
pred_probs <- predict(result, test, type = "prob")[, "Yes"]
roc_curve <- roc(test$o_bullied, pred_probs)
auc_value <- auc(roc_curve)
print(auc_value)
#Area under the curve: 0.72
#模型准确率
temp = c(temp,cm$overall["Accuracy"])
temp
# Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
# 0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 0.7789661 0.7902555 0.8086750 0.8045157 0.7712418 
f1 = c(f1,cm$byClass["F1"])
f1
#         F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
#0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 0.874493927 0.875044248 0.887017544 0.885085575 0.870848708  

#naive bayes
result <- train(o_bullied ~ .,data = train, trControl = train_control,
                method = "naive_bayes")
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
#Confusion Matrix and Statistics

#Reference
#Prediction  No Yes
#No  790 117
#Yes 508 268

#Accuracy : 0.6286          
#95% CI : (0.6051, 0.6518)
#No Information Rate : 0.7712          
#P-Value [Acc > NIR] : 1               

#Kappa : 0.2245          

#Mcnemar's Test P-Value : <2e-16          
                                          
#            Sensitivity : 0.6086          
#            Specificity : 0.6961          
#         Pos Pred Value : 0.8710          
#         Neg Pred Value : 0.3454          
#             Prevalence : 0.7712          
#         Detection Rate : 0.4694          
#   Detection Prevalence : 0.5389          
#      Balanced Accuracy : 0.6524          
                                          
#       'Positive' Class : No            
temp = c(temp,cm$overall["Accuracy"])
temp
#Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
#0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 0.7789661 0.7902555 0.8086750 0.8045157 0.7712418 0.6286393 
f1 = c(f1,cm$byClass["F1"])
f1
#        F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1   F1 
#0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 0.874493927 0.875044248 0.887017544 0.885085575 0.870848708 0.716553288

pred_probs <- predict(result, test, type = "prob")[, "Yes"]
roc_curve <- roc(test$o_bullied, pred_probs)
auc_value <- auc(roc_curve)
print(auc_value)
#Area under the curve: 0.6944
coords(roc_curve, "best", ret = c("sensitivity", "specificity"))
#    sensitivity specificity
#1   0.6181818   0.6910632

#xgboost
result <- train(o_bullied ~ .,data = train, trControl = train_control,
                method = "xgbLinear")
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
#Confusion Matrix and Statistics

#Reference
#Prediction   No  Yes
#No  1231  270
#Yes   67  115

#Accuracy : 0.7998          
#95% CI : (0.7798, 0.8186)
#No Information Rate : 0.7712          
#P-Value [Acc > NIR] : 0.002606        

#Kappa : 0.3033          

#Mcnemar's Test P-Value : < 2.2e-16       
                                          
#            Sensitivity : 0.9484          
#            Specificity : 0.2987          
#         Pos Pred Value : 0.8201          
#         Neg Pred Value : 0.6319          
#             Prevalence : 0.7712          
#         Detection Rate : 0.7314          
#   Detection Prevalence : 0.8919          
#      Balanced Accuracy : 0.6235          
                                          
#       'Positive' Class : No           
temp = c(temp,cm$overall["Accuracy"])
temp
#Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
#0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 0.7789661 0.7902555 0.8086750 0.8045157 0.7712418 0.6286393 0.7997623
f1 = c(f1,cm$byClass["F1"])
f1
#F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
#0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 0.874493927 0.875044248 0.887017544 0.885085575 0.870848708 
#F1          F1 
#0.716553288 0.879599857  
pred_probs <- predict(result, test, type = "prob")[, "Yes"]
roc_curve <- roc(test$o_bullied, pred_probs)
auc_value <- auc(roc_curve)
print(auc_value)
#Area under the curve: 0.7249
coords_optimal <- coords(roc_curve, "best", ret = c("sensitivity", "specificity"))
print(coords_optimal)
#sensitivity specificity
#1   0.6207792   0.7241911

#neural net
nnetGrid <- expand.grid(size = 1:10, decay = c(0, .1, 1, 2))
ncol(train)
result <- train(x = train[,-20], y = train$o_bullied,method = "nnet",tuneGrid = nnetGrid,
                trace = FALSE,maxit = 100,MaxNWts = 1000,trControl = train_control)
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
#Confusion Matrix and Statistics

#Reference
#Prediction   No  Yes
#No  1241  278
#Yes   57  107

#Accuracy : 0.801           
#95% CI : (0.7811, 0.8198)
#No Information Rate : 0.7712          
#P-Value [Acc > NIR] : 0.001787        

#Kappa : 0.2932          

#Mcnemar's Test P-Value : < 2.2e-16       
                                          
#            Sensitivity : 0.9561          
#            Specificity : 0.2779          
#         Pos Pred Value : 0.8170          
#         Neg Pred Value : 0.6524          
#             Prevalence : 0.7712          
#         Detection Rate : 0.7374          
#   Detection Prevalence : 0.9026          
#      Balanced Accuracy : 0.6170          
                                          
#       'Positive' Class : No       
temp = c(temp,cm$overall["Accuracy"])
temp
#Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
#0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 0.7789661 0.7902555 0.8086750 0.8045157 0.7712418 0.6286393 0.7997623 
#Accuracy 
#0.8009507 
f1 = c(f1,cm$byClass["F1"])
f1
#                  F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
#0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 0.874493927 0.875044248 0.887017544 0.885085575 0.870848708 
#F1          F1          F1 
#0.716553288 0.879599857 0.881079162 
pred_probs <- predict(result, test, type = "prob")
roc_curve <- roc(test$o_bullied, pred_probs[,2])
auc_value <- auc(roc_curve)
print(paste("AUC for nnet model:", auc_value))
#"AUC for nnet model: 0.747808816761051"
coords_optimal <- coords(roc_curve, "best", ret = c("sensitivity", "specificity"))
print(coords_optimal)
#   sensitivity specificity
#1   0.6363636   0.7550077

# random forest
train_control <- trainControl(method = "cv", number = 10, summaryFunction = twoClassSummary, classProbs = TRUE, savePredictions = TRUE)
mtryValues <- seq(2, ncol(df_3)-1, by = 1)
result <- train(x = train[, -20], y = train$o_bullied, method = "rf",ntree = 500,
                tuneGrid = data.frame(mtry = mtryValues),importance = TRUE,metric = "ROC",
                trControl = train_control)
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No  Yes
# No  1251  305
# Yes   47   80
# 
# Accuracy : 0.7908          
# 95% CI : (0.7706, 0.8101)
# No Information Rate : 0.7712          
# P-Value [Acc > NIR] : 0.02871         
# 
# Kappa : 0.2245          
# 
# Mcnemar's Test P-Value : < 2e-16         
#                                           
#             Sensitivity : 0.9638          
#             Specificity : 0.2078          
#          Pos Pred Value : 0.8040          
#          Neg Pred Value : 0.6299          
#              Prevalence : 0.7712          
#          Detection Rate : 0.7433          
#    Detection Prevalence : 0.9245          
#       Balanced Accuracy : 0.5858          
#                                           
#        'Positive' Class : No             
temp = c(temp,cm$overall["Accuracy"])
temp
# Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
# 0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 0.7789661 0.7902555 0.8086750 0.8045157 0.7712418 0.6286393 0.7997623 0.8009507 
# Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
# 0.8021390 0.7777778 0.5995247 0.7997623 0.7979798 0.8009507 0.7795603 0.5995247 0.7765894 0.7712418 0.7777778 0.7944147 0.7896613 0.7908497 
f1 = c(f1,cm$byClass["F1"])
f1
# F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
# 0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 0.874493927 0.875044248 0.887017544 0.885085575 0.870848708 0.716553288 
# F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
# 0.879599857 0.881079162 0.883525708 0.874074074 0.675649663 0.880538816 0.879261364 0.882083773 0.874873524 0.675649663 0.865905849 0.870848708 
# F1          F1          F1          F1 
# 0.864197531 0.876340243 0.877168633 0.876664331 
pred_probs <- predict(result, test, type = "prob")
roc_curve <- roc(test$o_bullied, pred_probs[,2])
auc_value <- auc(roc_curve)
print(paste("AUC for Random Forest model:", auc_value))
#"AUC for Random Forest model: 0.708981850199108"
coords_optimal <- coords(roc_curve, "best", ret = c("sensitivity", "specificity"))
print(coords_optimal)
#   sensitivity specificity
#1   0.6   0.7442219

#4 Recursive Feature Elimination
set.seed(123)

ctrl_param <- rfeControl(functions = rfFuncs,
                         method = "repeatedcv",
                         repeats = 5,
                         number =10,
                         verbose = FALSE,
                         returnResamp = "all")
ncol(df)
# 204
rfe_lm_profile <- rfe(df[,-204], df[, 204],
                      sizes = c(2,3),
                      rfeControl = ctrl_param)
importance <- varImp(rfe_lm_profile, scale = FALSE)
print(importance)
#从列表里选出了6以上的
#             Overall
#VS0046     17.36780433
#VS0069     14.87907239
#VS0124     13.52726550
#VS0070     13.12996078
#VS0157     12.95367399
#VS0112     10.87132585
#VS0123      9.01859206
#VS0051      8.46523593
#VS0115      8.45256998
#V3043       8.02584210
#VS0116      7.93401306
#VS0117      7.88256208
#V3042       7.78846212
#VS0053      7.49683487
#VS0131      7.35777623
#VS0130      7.18797261
#V3041       6.65228603
#V3040       6.59509761
#V3044       6.15186592
#V3045       6.07359000
#VS0049      5.92345259
#VS0059      5.84433913
#VS0067      5.46528068
#VS0022      5.31706978
#VS0055      5.25490944
#V3020       5.20124486
#VS0017      5.00805120
#VS0031      4.97332147
#VS0050      4.97290978
#VS0125      4.89455316
#VS0126      4.85008593
#VS0063      4.66657330
#VS0121      4.62935625
#VS0047      4.51014125
#V4526AA_1   4.44445657
#V4526H5_1   4.38171851
#VS0057      4.34598830
#VS0006      4.31034168
#VS0002      4.29439562
#V4526H7_1   4.27851031
#V4526H3A_1  4.23533933
#V4526H3B_1  4.21656320
#V4526H4_1   4.20049130
#V4526H6_1   4.14853639
#VS0062      4.12411975
#VS0052      4.06569990
#VS0010      4.04872607
#VS0023      4.01773894
#VS0137      3.97842016
#VS0048      3.77334493
#VS0122      3.76708438
#VS0136      3.64569832
#VS0061      3.31395199
#VS0133      3.29523753
#VS0058      3.27997214
#VS0154      3.10220543
#VS0065      3.09730364
#VS0155      3.09705287
#VS0138      3.00704740
#VS0132      2.98269345
#VS0135      2.89125145
#VS0148      2.84541011
#VS0119      2.80116336
#VS0033      2.77844309
#VS0060      2.74061658
#VS0114      2.74005968
#VS0128      2.67704166
#VS0113      2.66492455
#VS0054      2.57680706
#VS0147      2.57443469
#VS0064      2.57103254
#V2038       2.55739614
#V2042CAT    2.49539580
#VS0120      2.48087612
#VS0149      2.42094011
#VS0153      2.36979515
#VS0151      2.34616883
#V2047       2.33285430
#VS0118      2.26972315
#V3023A      2.23311841
#VS0150      2.22975695
#VS0152      2.19500870
#VS0146      2.17513472
#VS0027      2.16745510
#VS0035      2.08910358
#V2041       1.97885508
#V3032       1.96475665
#VS0129      1.96340383
#V2050       1.95271859
#VS0066      1.92520278
#V3012       1.92219637
#V3024       1.91011201
#V2026       1.90726853
#V2049A      1.90589545
#V2033CAT    1.90586483
#V2040A      1.89886827
#VS0068      1.88190586
#VS0034      1.81416536
#V3019       1.76423662
#V3071       1.70649969
#V2024       1.65117967
#V3072       1.64734724
#VS0024      1.64262372
#V3014CAT    1.63551938
#V3054       1.61155638
#V3058       1.58744397
#VS0026      1.58109162
#V2122       1.55014216
#V2034       1.51077573
#VS0134      1.43760146
#V3018       1.36352147
#VS0045      1.34599445
#V2127B      1.31574215
#V2129       1.26947047
#V2043       1.26177959
#V3035       1.23912865
#VS0028      1.22019513
#V2126B      1.17656731
#VS0007      1.12354548
#V3034       1.09829990
#VS0005      1.06953710
#V2125       1.06331333
#V2078       1.04346321
#V2025       1.00219416
#V3033       0.99562230
#V2124       0.96869682
#V2032       0.95886313
#VS0041      0.94954900
#VS0025      0.94026949
#VS0141      0.93538141
#VS0140      0.88839652
#V3073       0.83105574
#VS0021      0.80311312
#V2121       0.76420132
#V2120       0.72464451
#V3048       0.71684804
#VS0032      0.71529456
#V3061       0.71241140
#V3070       0.70887036
#VS0019      0.67189145
#V3052       0.66169953
#VS0029      0.59057846
#VS0020      0.57811668
#VS0039      0.56971310
#V3049       0.56470690
#V2025B      0.54326486
#VS0016      0.53412110
#VS0011      0.50597011
#V2037       0.49623341
#VS0030      0.47670440
#V3062       0.47525012
#V2045       0.43568571
#VS0037      0.38368956
#V2133       0.36687157
#V2036       0.34891528
#V2023       0.30442682
#VS0139      0.30316027
#VS0127      0.27595338
#VS0038      0.26890331
#VS0043      0.25718770
#V3064       0.23554059
#V2025A      0.21670836
#VS0036      0.20319187
#VS0040      0.19691708
#VS0015      0.18138319
#V3074       0.17780015
#V2022       0.17109464
#VS0014      0.15053918
#V2046       0.13951535
#V2128B      0.12799447
#V3047       0.07422922
#V3046       0.04004006
#V3075       0.03310574
#V3076       0.01533865
#V2077       0.00000000
#V2115       0.00000000
#V3039       0.00000000
#V3050       0.00000000
#V3053       0.00000000
#V3060       0.00000000
#V3069       0.00000000
#VS0008      0.00000000
#VS0013      0.00000000
#V3066      -0.02253932
#VS0042     -0.03816732
#V3078      -0.06090597
#V2074      -0.07912986
#V3059      -0.12680215
#V2119      -0.16535892
#V3068      -0.21881173
#V2132      -0.22470103
#VS0044     -0.23422216
#V3079      -0.34515642
#V2075      -0.39942712
#V3063      -0.47844128
#V3065      -0.56831122
#V2021      -0.66018504
#V3015      -0.68321551
#V2121B     -0.74481236
#V2076      -0.76063463
#V3067      -0.76427333
#V3077      -0.77307740
#V3038      -0.98825243
df_4 = df[,c("VS0046","VS0069","VS0124","VS0070","VS0157","VS0112","VS0123","VS0051","VS0115",
             "V3043","VS0116","VS0117","V3042","VS0053","VS0131","VS0130","V3041","V3040","V3044","V3045",
             "o_bullied")]
set.seed(123)
split <- initial_split(df_4, prop = 0.66, strata = o_bullied)
train <- training(split)
test <- testing(split)

#support vector machine
result <- train(o_bullied ~ ., data = train, trControl = train_control, method = "svmLinear")
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
#Confusion Matrix and Statistics

#Reference
#Prediction   No  Yes
#No  1298  374
#Yes    0   11

#Accuracy : 0.7778          
#95% CI : (0.7571, 0.7974)
#No Information Rate : 0.7712          
#P-Value [Acc > NIR] : 0.2722          

#Kappa : 0.0434          

#Mcnemar's Test P-Value : <2e-16          
                                          
#            Sensitivity : 1.00000         
#            Specificity : 0.02857         
#         Pos Pred Value : 0.77632         
#         Neg Pred Value : 1.00000         
#             Prevalence : 0.77124         
#         Detection Rate : 0.77124         
#   Detection Prevalence : 0.99346         
#      Balanced Accuracy : 0.51429         
                                          
#       'Positive' Class : No              
                                        
pred_probs <- predict(result, test, type = "prob")[, "Yes"]
roc_curve <- roc(test$o_bullied, pred_probs)
auc_value <- auc(roc_curve)
print(auc_value)
#Area under the curve: 0.6488
#模型准确率
temp = c(temp,cm$overall["Accuracy"])
temp
#Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
#0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 0.7789661 0.7902555 0.8086750 0.8045157 0.7712418 0.6286393 0.7997623 0.8009507 
#Accuracy  Accuracy 
#0.8021390 0.7777778 
f1 = c(f1,cm$byClass["F1"])
f1
#         F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
#0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 0.874493927 0.875044248 0.887017544 0.885085575 0.870848708 
#F1          F1          F1          F1          F1 
#0.716553288 0.879599857 0.881079162 0.883525708 0.874074074

#naive bayes
result <- train(o_bullied ~ .,data = train, trControl = train_control,
                method = "naive_bayes")
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
#Confusion Matrix and Statistics

#Reference
#Prediction  No Yes
#No  702  78
#Yes 596 307

#Accuracy : 0.5995         
#95% CI : (0.5757, 0.623)
#No Information Rate : 0.7712         
#P-Value [Acc > NIR] : 1              

#Kappa : 0.2296         

#Mcnemar's Test P-Value : <2e-16         
                                         
#            Sensitivity : 0.5408         
#            Specificity : 0.7974         
#         Pos Pred Value : 0.9000         
#         Neg Pred Value : 0.3400         
#             Prevalence : 0.7712         
#         Detection Rate : 0.4171         
#   Detection Prevalence : 0.4635         
#      Balanced Accuracy : 0.6691         
                                         
#       'Positive' Class : No                        
temp = c(temp,cm$overall["Accuracy"])
temp
#Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
#0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 0.7789661 0.7902555 0.8086750 0.8045157 0.7712418 0.6286393 0.7997623 0.8009507 
#Accuracy  Accuracy  Accuracy 
#0.8021390 0.7777778 0.5995247 
f1 = c(f1,cm$byClass["F1"])
f1
#         F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
#0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 0.874493927 0.875044248 0.887017544 0.885085575 0.870848708 
#F1          F1          F1          F1          F1          F1 
#0.716553288 0.879599857 0.881079162 0.883525708 0.874074074 0.675649663 

pred_probs <- predict(result, test, type = "prob")[, "Yes"]
roc_curve <- roc(test$o_bullied, pred_probs)
auc_value <- auc(roc_curve)
print(auc_value)
#Area under the curve: 0.7174
coords(roc_curve, "best", ret = c("sensitivity", "specificity"))
#    sensitivity specificity
#1   0.7974026    0.540832

#xgboost
result <- train(o_bullied ~ .,data = train, trControl = train_control,
                method = "xgbLinear")
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
#Confusion Matrix and Statistics

#Reference
#Prediction   No  Yes
#No  1242  281
#Yes   56  104

#Accuracy : 0.7998          
#95% CI : (0.7798, 0.8186)
#No Information Rate : 0.7712          
#P-Value [Acc > NIR] : 0.002606        

#Kappa : 0.2857          

#Mcnemar's Test P-Value : < 2.2e-16       
                                          
#            Sensitivity : 0.9569          
#            Specificity : 0.2701          
#         Pos Pred Value : 0.8155          
#         Neg Pred Value : 0.6500          
#             Prevalence : 0.7712          
#         Detection Rate : 0.7380          
#   Detection Prevalence : 0.9049          
#      Balanced Accuracy : 0.6135          
                                          
#       'Positive' Class : No             
temp = c(temp,cm$overall["Accuracy"])
temp
#Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
#0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 0.7789661 0.7902555 0.8086750 0.8045157 0.7712418 0.6286393 0.7997623 0.8009507 
#Accuracy  Accuracy  Accuracy  Accuracy 
#0.8021390 0.7777778 0.5995247 0.7997623 
f1 = c(f1,cm$byClass["F1"])
f1
#         F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
#0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 0.874493927 0.875044248 0.887017544 0.885085575 0.870848708 
#F1          F1          F1          F1          F1          F1          F1 
#0.716553288 0.879599857 0.881079162 0.883525708 0.874074074 0.675649663 0.880538816 
pred_probs <- predict(result, test, type = "prob")[, "Yes"]
roc_curve <- roc(test$o_bullied, pred_probs)
auc_value <- auc(roc_curve)
print(auc_value)
#Area under the curve: 0.7535
coords_optimal <- coords(roc_curve, "best", ret = c("sensitivity", "specificity"))
print(coords_optimal)
#sensitivity specificity
#1   0.6077922   0.7742681

#neural net
nnetGrid <- expand.grid(size = 1:10, decay = c(0, .1, 1, 2))
ncol(train)
result <- train(x = train[,-21], y = train$o_bullied,method = "nnet",tuneGrid = nnetGrid,
                trace = FALSE,maxit = 100,MaxNWts = 1000,trControl = train_control)
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
#Confusion Matrix and Statistics

#Reference
#Prediction   No  Yes
#No  1238  280
#Yes   60  105

#Accuracy : 0.798          
#95% CI : (0.778, 0.8169)
#No Information Rate : 0.7712         
#P-Value [Acc > NIR] : 0.004472       

#Kappa : 0.2835         

#Mcnemar's Test P-Value : < 2.2e-16      
                                         
#            Sensitivity : 0.9538         
#            Specificity : 0.2727         
#         Pos Pred Value : 0.8155         
#         Neg Pred Value : 0.6364         
#             Prevalence : 0.7712         
#         Detection Rate : 0.7356         
#   Detection Prevalence : 0.9020         
#      Balanced Accuracy : 0.6133         
                                         
#       'Positive' Class : No            
temp = c(temp,cm$overall["Accuracy"])
temp
#Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
#0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 0.7789661 0.7902555 0.8086750 0.8045157 0.7712418 0.6286393 0.7997623 0.8009507 
#Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
#0.8021390 0.7777778 0.5995247 0.7997623 0.7979798 
f1 = c(f1,cm$byClass["F1"])
f1
#F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
#0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 0.874493927 0.875044248 0.887017544 0.885085575 0.870848708 
#F1          F1          F1          F1          F1          F1          F1          F1 
#0.716553288 0.879599857 0.881079162 0.883525708 0.874074074 0.675649663 0.880538816 0.879261364 
pred_probs <- predict(result, test, type = "prob")
roc_curve <- roc(test$o_bullied, pred_probs[,2])
auc_value <- auc(roc_curve)
print(paste("AUC for nnet model:", auc_value))
#"AUC for nnet model: 0.73923518700098"
coords_optimal <- coords(roc_curve, "best", ret = c("sensitivity", "specificity"))
print(coords_optimal)
#   sensitivity specificity
#1   0.6   0.7742681

# random forest
train_control <- trainControl(method = "cv", number = 10, summaryFunction = twoClassSummary, classProbs = TRUE, savePredictions = TRUE)
mtryValues <- seq(2, ncol(df_4)-1, by = 1)
result <- train(x = train[, -21], y = train$o_bullied, method = "rf",ntree = 500,
                tuneGrid = data.frame(mtry = mtryValues),importance = TRUE,metric = "ROC",
                trControl = train_control)
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
#Confusion Matrix and Statistics

#Reference
#Prediction   No  Yes
#No  1253  290
#Yes   45   95

#Accuracy : 0.801           
#95% CI : (0.7811, 0.8198)
#No Information Rate : 0.7712          
#P-Value [Acc > NIR] : 0.001787        

#Kappa : 0.2732          

#Mcnemar's Test P-Value : < 2.2e-16       
                                          
#            Sensitivity : 0.9653          
#            Specificity : 0.2468          
#         Pos Pred Value : 0.8121          
#         Neg Pred Value : 0.6786          
#             Prevalence : 0.7712          
#         Detection Rate : 0.7445          
#   Detection Prevalence : 0.9168          
#      Balanced Accuracy : 0.6060          
                                          
#       'Positive' Class : No                      
temp = c(temp,cm$overall["Accuracy"])
temp
#Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
#0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 0.7789661 0.7902555 0.8086750 0.8045157 0.7712418 0.6286393 0.7997623 0.8009507 
#Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
#0.8021390 0.7777778 0.5995247 0.7997623 0.7979798 0.8009507
f1 = c(f1,cm$byClass["F1"])
f1
#         F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
#0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 0.874493927 0.875044248 0.887017544 0.885085575 0.870848708 0.716553288 
#F1          F1          F1          F1          F1          F1          F1          F1 
#0.879599857 0.881079162 0.883525708 0.874074074 0.675649663 0.880538816 0.879261364 0.882083773 
pred_probs <- predict(result, test, type = "prob")
roc_curve <- roc(test$o_bullied, pred_probs[,2])
auc_value <- auc(roc_curve)
print(paste("AUC for Random Forest model:", auc_value))
#"AUC for Random Forest model: 0.73412842935185"
coords_optimal <- coords(roc_curve, "best", ret = c("sensitivity", "specificity"))
print(coords_optimal)
#   sensitivity specificity
#1   0.6519481    0.748074

#5 Random forest importance
set.seed(123)

rfModel <-randomForest(o_bullied ~ ., data = df)
importance_matrix <- importance(rfModel)
importance_values <- as.numeric(importance_matrix[, 1])
importance_values <- as.numeric(importance_matrix[, 1])
importance_df <- data.frame(Variable = rownames(importance_matrix), Importance = importance_values)
importance_df_sorted <- importance_df[order(-importance_df$Importance), ]
print(importance_df_sorted)
# Variable  Importance
# 132     VS0046 59.17824714
# 51       V3032 34.51788500
# 164     VS0069 31.81697092
# 179     VS0124 30.82826379
# 8        V2026 28.43626218
# 166     VS0112 28.15834279
# 48       V3020 27.16934969
# 14       V2038 27.15257278
# 104     VS0017 26.60713868
# 21       V2047 26.44583866
# 108     VS0022 26.13919883
# 34       V2122 25.17231354
# 37      V2126B 24.30770607
# 165     VS0070 24.15595812
# 137     VS0051 23.68937589
# 98      VS0010 23.29131643
# 133     VS0047 19.09611906
# 28       V2078 18.90015472
# 196     VS0141 18.83950110
# 141     VS0055 18.43857016
# 139     VS0053 18.41668185
# 38      V2127B 18.14105704
# 136     VS0050 18.11951205
# 193     VS0138 17.47764083
# 162     VS0067 17.44851689
# 127     VS0041 15.88665131
# 149     VS0152 15.79607309
# 138     VS0052 15.45537932
# 111     VS0025 15.44992789
# 135     VS0049 15.36087202
# 158     VS0063 15.16475269
# 17    V2042CAT 14.99741249
# 154     VS0059 14.80499053
# 112     VS0026 14.53451780
# 43       V3012 14.53244022
# 157     VS0062 14.35603046
# 151     VS0154 14.23965021
# 94      VS0005 13.62750076
# 134     VS0048 13.59662566
# 110     VS0024 13.56703314
# 148     VS0057 13.38864562
# 52       V3033 13.15725208
# 10    V2033CAT 13.12354448
# 117     VS0031 12.52646558
# 175     VS0157 12.26422048
# 160     VS0065 12.06814704
# 190     VS0135 12.02457744
# 109     VS0023 11.94173417
# 40       V2129 11.86505052
# 142     VS0146 11.75996998
# 18       V2043 11.64754298
# 189     VS0134 11.54063708
# 11       V2034 11.40339590
# 181     VS0126 11.29615394
# 49      V3023A 11.18691413
# 159     VS0064 11.06883513
# 153     VS0058 11.03341325
# 188     VS0133 10.92367805
# 143     VS0147 10.91308436
# 152     VS0155 10.90289744
# 144     VS0148 10.55530659
# 156     VS0061 10.32053858
# 99      VS0011 10.26078367
# 155     VS0060 10.20117452
# 150     VS0153 10.07395333
# 3        V2023  9.88035218
# 180     VS0125  9.75809724
# 161     VS0066  9.64319775
# 140     VS0054  9.62730971
# 145     VS0149  9.56758750
# 131     VS0045  9.50412422
# 15      V2040A  9.30877846
# 122     VS0036  9.16601577
# 125     VS0039  9.16318711
# 39      V2128B  9.16298881
# 147     VS0151  9.09444917
# 22      V2049A  8.90705874
# 129     VS0043  8.80534724
# 115     VS0029  8.74039991
# 46       V3018  8.72418413
# 195     VS0140  8.61107181
# 113     VS0027  8.59152638
# 4        V2024  8.54606173
# 146     VS0150  8.23364924
# 106     VS0020  8.19241113
# 186     VS0131  8.07155494
# 163     VS0068  7.64760294
# 124     VS0038  7.57403009
# 120     VS0034  7.35409928
# 169     VS0115  7.34791487
# 192     VS0137  7.34152912
# 74       V3061  7.32958793
# 96      VS0007  7.32547719
# 32       V2121  7.19037713
# 19       V2045  7.18003724
# 185     VS0130  7.13868838
# 128     VS0042  7.07454235
# 84       V3071  7.06960109
# 118     VS0032  7.05966294
# 35       V2124  7.01114642
# 31       V2120  6.94828229
# 9        V2032  6.87813092
# 170     VS0116  6.86539648
# 114     VS0028  6.84285220
# 77       V3064  6.81319724
# 171     VS0117  6.75955142
# 36       V2125  6.68172326
# 85       V3072  6.63857640
# 119     VS0033  6.35512228
# 16       V2041  6.04640177
# 2        V2022  5.94784316
# 23       V2050  5.85645779
# 123     VS0037  5.78565988
# 50       V3024  5.76264679
# 178     VS0123  5.74462680
# 75       V3062  5.58349747
# 60       V3043  5.54009368
# 191     VS0136  5.43462493
# 194     VS0139  5.41372039
# 5        V2025  5.36968313
# 197  V4526AA_1  5.28153502
# 42       V2133  5.13794423
# 59       V3042  4.89404117
# 198 V4526H3A_1  4.69013795
# 107     VS0021  4.62046488
# 41       V2132  4.61964746
# 187     VS0132  4.59831706
# 116     VS0030  4.54931417
# 202  V4526H6_1  4.45703208
# 201  V4526H5_1  4.33103415
# 95      VS0006  4.32921659
# 199 V4526H3B_1  4.32634458
# 200  V4526H4_1  4.31166747
# 93      VS0002  4.18418716
# 12       V2036  4.02479112
# 58       V3041  4.02291911
# 57       V3040  4.01273019
# 24       V2074  3.94562159
# 105     VS0019  3.77866200
# 25       V2075  3.76715383
# 89       V3076  3.74810656
# 87       V3074  3.74320934
# 203  V4526H7_1  3.70589819
# 121     VS0035  3.68945052
# 54       V3035  3.38600887
# 126     VS0040  3.32245524
# 167     VS0113  3.25017419
# 20       V2046  3.13253272
# 183     VS0128  3.12351796
# 176     VS0121  2.91133591
# 92       V3079  2.79763402
# 47       V3019  2.78015753
# 130     VS0044  2.77346046
# 88       V3075  2.73713765
# 53       V3034  2.67435297
# 61       V3044  2.57386094
# 78       V3065  2.55458043
# 44    V3014CAT  2.54247128
# 6       V2025A  2.46823030
# 86       V3073  2.37566044
# 91       V3078  2.23663502
# 62       V3045  2.21840701
# 13       V2037  2.15559099
# 7       V2025B  1.97182984
# 168     VS0114  1.81441576
# 83       V3070  1.79873925
# 173     VS0119  1.66926043
# 103     VS0016  1.65841417
# 184     VS0129  1.60689066
# 177     VS0122  1.42727347
# 172     VS0118  1.37186819
# 174     VS0120  1.26558663
# 30       V2119  1.20640352
# 101     VS0014  1.14623537
# 1        V2021  1.13044507
# 102     VS0015  1.04692411
# 71       V3058  0.85582038
# 70       V3054  0.82957019
# 182     VS0127  0.76938867
# 79       V3066  0.69906831
# 55       V3038  0.68590064
# 45       V3015  0.61813156
# 76       V3063  0.61502328
# 26       V2076  0.59196601
# 90       V3077  0.53727854
# 66       V3049  0.52689678
# 68       V3052  0.43200268
# 80       V3067  0.41650585
# 33      V2121B  0.41089096
# 65       V3048  0.38959419
# 81       V3068  0.25089588
# 64       V3047  0.10883529
# 63       V3046  0.07091112
# 72       V3059  0.05818336
# 27       V2077  0.00000000
# 29       V2115  0.00000000
# 56       V3039  0.00000000
# 67       V3050  0.00000000
# 69       V3053  0.00000000
# 73       V3060  0.00000000
# 82       V3069  0.00000000
# 97      VS0008  0.00000000
# 100     VS0013  0.00000000

# VS0046 V3032 VS0069 VS0124 V2026 VS0112 V3020 V2038 VS0017 V2047 VS0022 V2122 V2126B VS0070 VS0051 VS0010 VS0047 V2078 VS0141 VS0055 VS0053 V2127B VS0050

df_5 = df[,c("VS0046","V3032","VS0069",
             "VS0124","V2026","VS0112","V3020", "V2038", "VS0017", "V2047", "VS0022", "V2122", "V2126B", 
             "VS0070", "VS0051", "VS0010", "VS0047", "V2078", "VS0141", "VS0055", "VS0053", "V2127B", "VS0050","o_bullied")]


set.seed(123)
library(rsample)
split <- initial_split(df_5, prop = 0.66, strata = o_bullied)
train <- training(split)
test <- testing(split)

#support vector machine
result <- train(o_bullied ~ ., data = train, trControl = train_control, method = "svmLinear")
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No  Yes
# No  1298  385
# Yes    0    0
# 
# Accuracy : 0.7712          
# 95% CI : (0.7504, 0.7911)
# No Information Rate : 0.7712          
# P-Value [Acc > NIR] : 0.5137          
# 
# Kappa : 0               
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 1.0000          
#             Specificity : 0.0000          
#          Pos Pred Value : 0.7712          
#          Neg Pred Value :    NaN          
#              Prevalence : 0.7712          
#          Detection Rate : 0.7712          
#    Detection Prevalence : 1.0000          
#       Balanced Accuracy : 0.5000          
#                                           
#        'Positive' Class : No                     

pred_probs <- predict(result, test, type = "prob")[, "Yes"]
roc_curve <- roc(test$o_bullied, pred_probs)
auc_value <- auc(roc_curve)
print(auc_value)
#Area under the curve: 0.6752
#模型准确率
temp = c(temp,cm$overall["Accuracy"])
temp
# Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
# 0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 0.7789661 0.7902555 0.8086750 0.8045157 0.7712418 0.6286393 0.7997623 0.8009507 
# Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
# 0.8021390 0.7777778 0.5995247 0.7997623 0.7979798 0.8009507 0.7795603 0.5995247 0.7765894 0.7712418 
f1 = c(f1,cm$byClass["F1"])
f1
# F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
# 0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 0.874493927 0.875044248 0.887017544 0.885085575 0.870848708 0.716553288 
# F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
# 0.879599857 0.881079162 0.883525708 0.874074074 0.675649663 0.880538816 0.879261364 0.882083773 0.874873524 0.675649663 0.865905849 0.870848708

#naive bayes
result <- train(o_bullied ~ .,data = train, trControl = train_control,
                method = "naive_bayes")
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No  Yes
# No  1214  292
# Yes   84   93
# 
# Accuracy : 0.7766          
# 95% CI : (0.7559, 0.7963)
# No Information Rate : 0.7712          
# P-Value [Acc > NIR] : 0.3123          
# 
# Kappa : 0.2183          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.9353          
#             Specificity : 0.2416          
#          Pos Pred Value : 0.8061          
#          Neg Pred Value : 0.5254          
#              Prevalence : 0.7712          
#          Detection Rate : 0.7213          
#    Detection Prevalence : 0.8948          
#       Balanced Accuracy : 0.5884          
#                                           
#        'Positive' Class : No                                
temp = c(temp,cm$overall["Accuracy"])
temp
# Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
# 0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 0.7789661 0.7902555 0.8086750 0.8045157 0.7712418 0.6286393 0.7997623 0.8009507 
# Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
# 0.8021390 0.7777778 0.5995247 0.7997623 0.7979798 0.8009507 0.7795603 0.5995247 0.7765894 
f1 = c(f1,cm$byClass["F1"])
f1
# F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
# 0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 0.874493927 0.875044248 0.887017544 0.885085575 0.870848708 0.716553288 
# F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
# 0.879599857 0.881079162 0.883525708 0.874074074 0.675649663 0.880538816 0.879261364 0.882083773 0.874873524 0.675649663 0.865905849

pred_probs <- predict(result, test, type = "prob")[, "Yes"]
roc_curve <- roc(test$o_bullied, pred_probs)
auc_value <- auc(roc_curve)
print(auc_value)
#Area under the curve: 0.7071
coords(roc_curve, "best", ret = c("sensitivity", "specificity"))
#    sensitivity specificity
#1   0.5896104   0.7380586

#xgboost
result <- train(o_bullied ~ .,data = train, trControl = train_control,
                method = "xgbLinear")
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No  Yes
# No  1190  266
# Yes  108  119
# 
# Accuracy : 0.7778          
# 95% CI : (0.7571, 0.7974)
# No Information Rate : 0.7712          
# P-Value [Acc > NIR] : 0.2722          
# 
# Kappa : 0.264           
# 
# Mcnemar's Test P-Value : 4.728e-16       
#                                           
#             Sensitivity : 0.9168          
#             Specificity : 0.3091          
#          Pos Pred Value : 0.8173          
#          Neg Pred Value : 0.5242          
#              Prevalence : 0.7712          
#          Detection Rate : 0.7071          
#    Detection Prevalence : 0.8651          
#       Balanced Accuracy : 0.6129          
#                                           
#        'Positive' Class : No                    
temp = c(temp,cm$overall["Accuracy"])
temp
# Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
# 0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 0.7789661 0.7902555 0.8086750 0.8045157 0.7712418 0.6286393 0.7997623 0.8009507 
# Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
# 0.8021390 0.7777778 0.5995247 0.7997623 0.7979798 0.8009507 0.7795603 0.5995247 0.7765894 0.7712418 0.7777778 
f1 = c(f1,cm$byClass["F1"])
f1
# F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
# 0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 0.874493927 0.875044248 0.887017544 0.885085575 0.870848708 0.716553288 
# F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
# 0.879599857 0.881079162 0.883525708 0.874074074 0.675649663 0.880538816 0.879261364 0.882083773 0.874873524 0.675649663 0.865905849 0.870848708 
# F1 
# 0.864197531 
pred_probs <- predict(result, test, type = "prob")[, "Yes"]
roc_curve <- roc(test$o_bullied, pred_probs)
auc_value <- auc(roc_curve)
print(auc_value)
#Area under the curve: 0.7125
coords_optimal <- coords(roc_curve, "best", ret = c("sensitivity", "specificity"))
print(coords_optimal)
#sensitivity specificity
#1    0.561039   0.7619414

#neural net
nnetGrid <- expand.grid(size = 1:10, decay = c(0, .1, 1, 2))
ncol(train)
result <- train(x = train[,-24], y = train$o_bullied,method = "nnet",tuneGrid = nnetGrid,
                trace = FALSE,maxit = 100,MaxNWts = 1000,trControl = train_control)
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No  Yes
# No  1226  274
# Yes   72  111
# 
# Accuracy : 0.7944          
# 95% CI : (0.7743, 0.8135)
# No Information Rate : 0.7712          
# P-Value [Acc > NIR] : 0.01202         
# 
# Kappa : 0.2855          
# 
# Mcnemar's Test P-Value : < 2e-16         
#                                           
#             Sensitivity : 0.9445          
#             Specificity : 0.2883          
#          Pos Pred Value : 0.8173          
#          Neg Pred Value : 0.6066          
#              Prevalence : 0.7712          
#          Detection Rate : 0.7285          
#    Detection Prevalence : 0.8913          
#       Balanced Accuracy : 0.6164          
#                                           
#        'Positive' Class : No               
temp = c(temp,cm$overall["Accuracy"])
temp
# Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
# 0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 0.7789661 0.7902555 0.8086750 0.8045157 0.7712418 0.6286393 0.7997623 0.8009507 
# Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
# 0.8021390 0.7777778 0.5995247 0.7997623 0.7979798 0.8009507 0.7795603 0.5995247 0.7765894 0.7712418 0.7777778 0.7944147 
f1 = c(f1,cm$byClass["F1"])
f1
# F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
# 0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 0.874493927 0.875044248 0.887017544 0.885085575 0.870848708 0.716553288 
# F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
# 0.879599857 0.881079162 0.883525708 0.874074074 0.675649663 0.880538816 0.879261364 0.882083773 0.874873524 0.675649663 0.865905849 0.870848708 
# F1          F1 
# 0.864197531 0.876340243 
pred_probs <- predict(result, test, type = "prob")
roc_curve <- roc(test$o_bullied, pred_probs[,2])
auc_value <- auc(roc_curve)
print(paste("AUC for nnet model:", auc_value))
#"AUC for nnet model: 0.740268745122366"
coords_optimal <- coords(roc_curve, "best", ret = c("sensitivity", "specificity"))
print(coords_optimal)
#   sensitivity specificity
#1   0.6415584   0.7311248

# random forest
train_control <- trainControl(method = "cv", number = 10, summaryFunction = twoClassSummary, classProbs = TRUE, savePredictions = TRUE)
mtryValues <- seq(2, ncol(df_5)-1, by = 1)
result <- train(x = train[, -24], y = train$o_bullied, method = "rf",ntree = 500,
                tuneGrid = data.frame(mtry = mtryValues),importance = TRUE,metric = "ROC",
                trControl = train_control)
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   No  Yes
# No  1264  320
# Yes   34   65
# 
# Accuracy : 0.7897          
# 95% CI : (0.7694, 0.8089)
# No Information Rate : 0.7712          
# P-Value [Acc > NIR] : 0.0374          
# 
# Kappa : 0.1931          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.9738          
#             Specificity : 0.1688          
#          Pos Pred Value : 0.7980          
#          Neg Pred Value : 0.6566          
#              Prevalence : 0.7712          
#          Detection Rate : 0.7510          
#    Detection Prevalence : 0.9412          
#       Balanced Accuracy : 0.5713          
#                                           
#        'Positive' Class : No                          
temp = c(temp,cm$overall["Accuracy"])
temp
# Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
# 0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 0.7789661 0.7902555 0.8086750 0.8045157 0.7712418 0.6286393 0.7997623 0.8009507 
# Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
# 0.8021390 0.7777778 0.5995247 0.7997623 0.7979798 0.8009507 0.7795603 0.5995247 0.7765894 0.7712418 0.7777778 0.7944147 0.7896613 
f1 = c(f1,cm$byClass["F1"])
f1
# F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
# 0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 0.874493927 0.875044248 0.887017544 0.885085575 0.870848708 0.716553288 
# F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
# 0.879599857 0.881079162 0.883525708 0.874074074 0.675649663 0.880538816 0.879261364 0.882083773 0.874873524 0.675649663 0.865905849 0.870848708 
# F1          F1          F1 
# 0.864197531 0.876340243 0.877168633 
pred_probs <- predict(result, test, type = "prob")
roc_curve <- roc(test$o_bullied, pred_probs[,2])
auc_value <- auc(roc_curve)
print(paste("AUC for Random Forest model:", auc_value))
#"AUC for Random Forest model: 0.762449722850339"
coords_optimal <- coords(roc_curve, "best", ret = c("sensitivity", "specificity"))
print(coords_optimal)
#   sensitivity specificity
#1   0.7298701   0.6656394

#自己选的 features support vector machine
df_0$o_bullied <- as.factor(df_0$o_bullied)
levels(df_0$o_bullied) <- make.names(levels(df_0$o_bullied), unique = TRUE)
print(levels(df_0$o_bullied))
#"X0" "X1"
set.seed(123)
split <- initial_split(df_0, prop = 0.66, strata = o_bullied)
train <- training(split)
test <- testing(split)
train$o_bullied <- factor(train$o_bullied, levels = levels(df_0$o_bullied))
print(levels(train$o_bullied))
#"X0" "X1"
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
result <- train(o_bullied ~ ., data = train, trControl = train_control, method = "svmLinear", metric = "ROC")
pred <- predict(result, test)
cm <- confusionMatrix(pred, test$o_bullied)
cm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   X0   X1
# X0 1293  353
# X1    5   32
# 
# Accuracy : 0.7873          
# 95% CI : (0.7669, 0.8066)
# No Information Rate : 0.7712          
# P-Value [Acc > NIR] : 0.06113         
# 
# Kappa : 0.1162          
# 
# Mcnemar's Test P-Value : < 2e-16         
#                                           
#             Sensitivity : 0.99615         
#             Specificity : 0.08312         
#          Pos Pred Value : 0.78554         
#          Neg Pred Value : 0.86486         
#              Prevalence : 0.77124         
#          Detection Rate : 0.76827         
#    Detection Prevalence : 0.97802         
#       Balanced Accuracy : 0.53963         
#                                           
#        'Positive' Class : X0            
temp = c(temp,cm$overall["Accuracy"])
temp
#Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
#0.7801545 0.2287582 0.7795603 0.7896613 0.7884730 0.7932264 0.7789661 0.7902555 0.8086750 0.8045157 0.7712418 0.6286393 0.7997623 0.8009507 
#Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
#0.8021390 0.7777778 0.5995247 0.7997623 0.7979798 0.8009507 0.7795603 0.5995247 0.7765894 0.7712418 0.7777778 0.7944147 0.7896613 0.7908497 
#Accuracy  Accuracy 
#0.7718360 0.7872846 
f1 = c(f1,cm$byClass["F1"])
f1
#F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
#0.873720137 0.001538462 0.868858254 0.878183070 0.876388889 0.880576527 0.874493927 0.875044248 0.887017544 0.885085575 0.870848708 
#F1          F1          F1          F1          F1          F1          F1          F1          F1          F1          F1 
#0.716553288 0.879599857 0.881079162 0.883525708 0.874074074 0.675649663 0.880538816 0.879261364 0.882083773 0.874873524 0.675649663 
#F1          F1          F1          F1          F1          F1          F1 
#0.865905849 0.870848708 0.864197531 0.876340243 0.877168633 0.876664331 0.878396739 
plot(temp, ylab = "Accuracy", xlab = "Algorithm")
lines(temp, ylab = "Accuracy", xlab = "Algorithm")