suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(reshape2)
  library(corrplot)
  library(caret)
  library(randomForest)
  library(rpart)
  # rpart.plot not needed
  library(pROC)
  library(e1071)
})

set.seed(42)
setwd("C:/Users/Bruce/Downloads")

# ── Load data ─────────────────────────────────────────────────────────────────
df <- read.csv("diabetes.csv", header=TRUE)
cat("Rows:", nrow(df), "| Cols:", ncol(df), "\n")

# ── PART 2: Wrangling ─────────────────────────────────────────────────────────

# Missing values (biological zeros)
zero_cols <- c("Glucose","BloodPressure","SkinThickness","Insulin","BMI")
cat("\nZero counts:\n")
for(col in zero_cols) cat(col,":",sum(df[[col]]==0),"\n")

df[zero_cols] <- lapply(df[zero_cols], function(x) ifelse(x==0, NA, x))
for(col in zero_cols) {
  m <- median(df[[col]], na.rm=TRUE)
  df[[col]][is.na(df[[col]])] <- m
  cat("Imputed",col,"with median =",m,"\n")
}

# Outliers — manual IQR winsorize (no DescTools needed)
winsorize_manual <- function(x, lo=0.05, hi=0.95) {
  lb <- quantile(x, lo, na.rm=TRUE)
  ub <- quantile(x, hi, na.rm=TRUE)
  pmax(pmin(x, ub), lb)
}

for(col in c("Insulin","BMI","DiabetesPedigreeFunction","BloodPressure")) {
  df[[col]] <- winsorize_manual(df[[col]])
}
cat("\nWinsorizing complete (manual IQR method)\n")

# Class balance
cat("\nOutcome distribution:\n")
print(table(df$Outcome))
cat("Proportions:\n")
print(round(prop.table(table(df$Outcome))*100,1))

df$Outcome <- factor(df$Outcome, levels=c(0,1),
                     labels=c("Non.Diabetic","Diabetic"))

# Feature engineering
df$BMI_Category     <- as.integer(as.character(
  cut(df$BMI, breaks=c(0,18.5,25,30,Inf),
      labels=c(0,1,2,3), right=FALSE)))
df$Glucose_BMI_Score <- round((df$Glucose * df$BMI)/100, 4)
df$Young_High_Preg   <- as.integer(df$Age < 30 & df$Pregnancies > 3)

cat("\nNew features created: BMI_Category, Glucose_BMI_Score, Young_High_Preg\n")
write.csv(df, "diabetes_cleaned.csv", row.names=FALSE)
cat("Cleaned dataset saved\n")

# ── PART 3: EDA ───────────────────────────────────────────────────────────────

cat("\n=== SUMMARY STATISTICS BY OUTCOME ===\n")
stats <- df %>%
  group_by(Outcome) %>%
  summarise(
    n            = n(),
    Mean_Glucose = round(mean(Glucose),2),
    Mean_BMI     = round(mean(BMI),2),
    Mean_Age     = round(mean(Age),2),
    Mean_Insulin = round(mean(Insulin),2),
    Mean_BP      = round(mean(BloodPressure),2),
    Mean_Preg    = round(mean(Pregnancies),2),
    Mean_DPF     = round(mean(DiabetesPedigreeFunction),3)
  )
print(as.data.frame(stats))

# Correlation with numeric outcome
df_num <- df %>%
  mutate(Outcome_num = as.integer(Outcome)-1) %>%
  select(-Outcome)
cor_mat <- cor(df_num, use="complete.obs")
cat("\n=== CORRELATIONS WITH OUTCOME ===\n")
cor_outcome <- sort(cor_mat[,"Outcome_num"], decreasing=TRUE)
print(round(cor_outcome,4))

# ── PLOTS ─────────────────────────────────────────────────────────────────────

# Plot 1 - Histograms
feat <- c("Glucose","BMI","Age","Insulin")
df_long <- melt(df[,c(feat,"Outcome")], id.vars="Outcome")
p1 <- ggplot(df_long, aes(x=value, fill=Outcome)) +
  geom_histogram(bins=28, alpha=0.65, position="identity") +
  facet_wrap(~variable, scales="free") +
  scale_fill_manual(values=c("#4472C4","#C00000")) +
  labs(title="Distribution of Key Features by Outcome",
       x="Value", y="Count") +
  theme_minimal(base_size=11) +
  theme(plot.title=element_text(face="bold"))
ggsave("plot1_histograms.png", p1, width=8, height=5, dpi=150)

# Plot 2 - Boxplots
p2 <- ggplot(df_long, aes(x=Outcome, y=value, fill=Outcome)) +
  geom_boxplot(alpha=0.7, outlier.size=0.8) +
  facet_wrap(~variable, scales="free") +
  scale_fill_manual(values=c("#4472C4","#C00000")) +
  labs(title="Feature Spread by Outcome", x="", y="Value") +
  theme_minimal(base_size=11) +
  theme(legend.position="none", plot.title=element_text(face="bold"))
ggsave("plot2_boxplots.png", p2, width=8, height=5, dpi=150)

# Plot 3 - Class distribution
p3 <- ggplot(df, aes(x=Outcome, fill=Outcome)) +
  geom_bar(width=0.5) +
  geom_text(stat="count", aes(label=after_stat(count)), vjust=-0.4, fontface="bold") +
  scale_fill_manual(values=c("#4472C4","#C00000")) +
  labs(title="Class Distribution", x="", y="Count") +
  theme_minimal(base_size=11) +
  theme(legend.position="none", plot.title=element_text(face="bold"))
ggsave("plot3_class_dist.png", p3, width=5, height=4, dpi=150)

# Plot 4 - Scatter glucose vs BMI
p4 <- ggplot(df, aes(x=Glucose, y=BMI, color=Outcome)) +
  geom_point(alpha=0.45, size=1.5) +
  geom_smooth(method="lm", se=TRUE, linewidth=0.8) +
  scale_color_manual(values=c("#4472C4","#C00000")) +
  labs(title="Glucose vs BMI by Outcome",
       x="Glucose (mg/dL)", y="BMI") +
  theme_minimal(base_size=11) +
  theme(plot.title=element_text(face="bold"))
ggsave("plot4_scatter.png", p4, width=7, height=5, dpi=150)

# Plot 5 - Age density
p5 <- ggplot(df, aes(x=Age, fill=Outcome)) +
  geom_density(alpha=0.5) +
  scale_fill_manual(values=c("#4472C4","#C00000")) +
  labs(title="Age Distribution by Outcome", x="Age (years)", y="Density") +
  theme_minimal(base_size=11) +
  theme(plot.title=element_text(face="bold"))
ggsave("plot5_age_density.png", p5, width=7, height=4, dpi=150)

# Plot 6 - Correlation matrix
png("plot6_correlation.png", width=800, height=700, res=130)
corrplot(cor_mat, method="color", type="upper",
         tl.cex=0.75, addCoef.col="black", number.cex=0.55,
         col=colorRampPalette(c("#4472C4","white","#C00000"))(200),
         title="Correlation Matrix", mar=c(0,0,2,0))
dev.off()
cat("All 6 plots saved\n")

# ── PART 4: Probability & Statistics ──────────────────────────────────────────

n_total    <- nrow(df)
n_diab     <- sum(df$Outcome=="Diabetic")
n_nondiab  <- sum(df$Outcome=="Non.Diabetic")

P_diab    <- n_diab/n_total
P_nondiab <- n_nondiab/n_total

cat("\n=== MARGINAL PROBABILITIES ===\n")
cat("P(Diabetic)    =", round(P_diab,4),"\n")
cat("P(Non-Diabetic)=", round(P_nondiab,4),"\n")

# Conditional probability
df$HighGlucose <- ifelse(df$Glucose>=140,"High","Normal")
gt <- table(df$HighGlucose, df$Outcome)
cat("\n=== CONDITIONAL PROBABILITY TABLE ===\n"); print(gt)
P_d_high   <- gt["High","Diabetic"]  /sum(gt["High",])
P_d_normal <- gt["Normal","Diabetic"]/sum(gt["Normal",])
cat("P(Diabetic|High Glucose)  =",round(P_d_high,4),"\n")
cat("P(Diabetic|Normal Glucose)=",round(P_d_normal,4),"\n")
cat("Risk multiplier           =",round(P_d_high/P_d_normal,2),"x\n")

# Joint
P_joint <- sum(df$Glucose>=140 & df$BMI>=30 &
               df$Outcome=="Diabetic")/n_total
cat("\nP(Diabetic AND High Glucose AND High BMI)=",round(P_joint,4),"\n")

# Bayes
P_obese_given_diab <- sum(df$BMI>=30 & df$Outcome=="Diabetic")/n_diab
P_obese            <- sum(df$BMI>=30)/n_total
P_diab_given_obese <- (P_obese_given_diab * P_diab)/P_obese
cat("\n=== BAYES THEOREM ===\n")
cat("P(Obese|Diabetic)  =",round(P_obese_given_diab,4),"\n")
cat("P(Obese)           =",round(P_obese,4),"\n")
cat("P(Diabetic|Obese)  =",round(P_diab_given_obese,4),"\n")

# Hypothesis Test 1 - glucose t-test
g_diab   <- df$Glucose[df$Outcome=="Diabetic"]
g_ndiab  <- df$Glucose[df$Outcome=="Non.Diabetic"]
t1 <- t.test(g_diab, g_ndiab, alternative="greater", var.equal=FALSE)
cat("\n=== TEST 1: Glucose t-test ===\n")
cat("t =",round(t1$statistic,4),"| p =",format(t1$p.value, scientific=TRUE),"\n")
cat("Mean Diabetic =",round(mean(g_diab),2),"| Mean Non-Diabetic =",round(mean(g_ndiab),2),"\n")

# Hypothesis Test 2 - BMI t-test + Cohen's d
b_diab  <- df$BMI[df$Outcome=="Diabetic"]
b_ndiab <- df$BMI[df$Outcome=="Non.Diabetic"]
t2 <- t.test(b_diab, b_ndiab, alternative="two.sided", var.equal=FALSE)
cohens_d <- (mean(b_diab)-mean(b_ndiab)) /
            sqrt((sd(b_diab)^2 + sd(b_ndiab)^2)/2)
cat("\n=== TEST 2: BMI t-test ===\n")
cat("t =",round(t2$statistic,4),"| p =",format(t2$p.value, scientific=TRUE),"\n")
cat("Mean BMI Diabetic =",round(mean(b_diab),2),"| Non-Diabetic =",round(mean(b_ndiab),2),"\n")
cat("Cohen's d =",round(cohens_d,4),"\n")

# Hypothesis Test 3 - Chi-square + Cramer's V
chisq_r <- chisq.test(gt)
cramers_v <- sqrt(chisq_r$statistic / (n_total*(min(dim(gt))-1)))
cat("\n=== TEST 3: Chi-Square ===\n")
cat("X2 =",round(chisq_r$statistic,4),"| df =",chisq_r$parameter,
    "| p =",format(chisq_r$p.value, scientific=TRUE),"\n")
cat("Cramer's V =",round(cramers_v,4),"\n")

# ── PART 5: Machine Learning ───────────────────────────────────────────────────

# Remove helper columns
df_ml <- df %>% select(-HighGlucose)

train_idx  <- createDataPartition(df_ml$Outcome, p=0.80, list=FALSE)
train_data <- df_ml[train_idx,]
test_data  <- df_ml[-train_idx,]
cat("\nTrain rows:",nrow(train_data),"| Test rows:",nrow(test_data),"\n")

ctrl <- trainControl(method="cv", number=10,
                     classProbs=TRUE, summaryFunction=twoClassSummary,
                     savePredictions=TRUE)

feat_form <- Outcome ~ Glucose + BMI + Age + Pregnancies +
             Insulin + DiabetesPedigreeFunction +
             BloodPressure + SkinThickness +
             Glucose_BMI_Score + BMI_Category

# Logistic Regression
cat("\nTraining Logistic Regression...\n")
m_log <- train(feat_form, data=train_data, method="glm",
               family="binomial", trControl=ctrl, metric="ROC")
cat("LR CV ROC:",round(max(m_log$results$ROC),4),"\n")

# Decision Tree
cat("Training Decision Tree...\n")
m_tree <- train(feat_form, data=train_data, method="rpart",
                trControl=ctrl, metric="ROC", tuneLength=8)
cat("DT CV ROC:",round(max(m_tree$results$ROC),4),"\n")

# Random Forest
cat("Training Random Forest...\n")
m_rf <- train(feat_form, data=train_data, method="rf",
              trControl=ctrl, metric="ROC", tuneLength=4)
cat("RF CV ROC:",round(max(m_rf$results$ROC),4),"\n")

# Predictions
pred_log  <- predict(m_log,  test_data)
pred_tree <- predict(m_tree, test_data)
pred_rf   <- predict(m_rf,   test_data)

prob_log  <- predict(m_log,  test_data, type="prob")[,"Diabetic"]
prob_tree <- predict(m_tree, test_data, type="prob")[,"Diabetic"]
prob_rf   <- predict(m_rf,   test_data, type="prob")[,"Diabetic"]

# Confusion matrices
cm_log  <- confusionMatrix(pred_log,  test_data$Outcome, positive="Diabetic")
cm_tree <- confusionMatrix(pred_tree, test_data$Outcome, positive="Diabetic")
cm_rf   <- confusionMatrix(pred_rf,   test_data$Outcome, positive="Diabetic")

# ROC / AUC
roc_log  <- roc(test_data$Outcome, prob_log,
                levels=c("Non.Diabetic","Diabetic"), quiet=TRUE)
roc_tree <- roc(test_data$Outcome, prob_tree,
                levels=c("Non.Diabetic","Diabetic"), quiet=TRUE)
roc_rf   <- roc(test_data$Outcome, prob_rf,
                levels=c("Non.Diabetic","Diabetic"), quiet=TRUE)

cat("\n=== MODEL COMPARISON ===\n")
results <- data.frame(
  Model       = c("Logistic Regression","Decision Tree","Random Forest"),
  AUC         = round(c(auc(roc_log), auc(roc_tree), auc(roc_rf)),4),
  Accuracy    = round(c(cm_log$overall["Accuracy"],
                        cm_tree$overall["Accuracy"],
                        cm_rf$overall["Accuracy"]),4),
  Sensitivity = round(c(cm_log$byClass["Sensitivity"],
                        cm_tree$byClass["Sensitivity"],
                        cm_rf$byClass["Sensitivity"]),4),
  Specificity = round(c(cm_log$byClass["Specificity"],
                        cm_tree$byClass["Specificity"],
                        cm_rf$byClass["Specificity"]),4),
  F1          = round(c(cm_log$byClass["F1"],
                        cm_tree$byClass["F1"],
                        cm_rf$byClass["F1"]),4)
)
print(results)

# Threshold tuning on RF
thresh <- 0.40
pred_rf_tuned <- factor(ifelse(prob_rf>=thresh,"Diabetic","Non.Diabetic"),
                        levels=c("Non.Diabetic","Diabetic"))
cm_rf_tuned <- confusionMatrix(pred_rf_tuned, test_data$Outcome,
                               positive="Diabetic")
cat("\n=== RF THRESHOLD 0.40 ===\n")
cat("Sensitivity:",round(cm_rf_tuned$byClass["Sensitivity"],4),"\n")
cat("Specificity:",round(cm_rf_tuned$byClass["Specificity"],4),"\n")
cat("Accuracy:   ",round(cm_rf_tuned$overall["Accuracy"],4),"\n")
cat("F1:         ",round(cm_rf_tuned$byClass["F1"],4),"\n")

# Logistic regression odds ratios
cat("\n=== LOGISTIC REGRESSION ODDS RATIOS ===\n")
or <- round(exp(coef(m_log$finalModel)),4)
print(or)

# RF feature importance
cat("\n=== RF FEATURE IMPORTANCE ===\n")
fi <- varImp(m_rf)$importance
fi$Feature <- rownames(fi)
fi <- fi[order(fi$Overall, decreasing=TRUE),]
print(fi)

# Plot 7 - ROC curves
png("plot7_roc.png", width=700, height=600, res=130)
plot(roc_log, col="#4472C4", lwd=2,
     main="ROC Curves - Model Comparison")
plot(roc_tree, col="#70AD47", lwd=2, add=TRUE)
plot(roc_rf,   col="#C00000", lwd=2, add=TRUE)
abline(a=0, b=1, lty=2, col="grey60")
legend("bottomright",
       legend=c(paste0("Logistic Reg (AUC=",round(auc(roc_log),3),")"),
                paste0("Decision Tree (AUC=",round(auc(roc_tree),3),")"),
                paste0("Random Forest (AUC=",round(auc(roc_rf),3),")")),
       col=c("#4472C4","#70AD47","#C00000"), lwd=2, cex=0.85)
dev.off()

# Plot 8 - Feature importance
fi_plot <- fi[1:8,]
png("plot8_importance.png", width=700, height=500, res=130)
barplot(fi_plot$Overall[order(fi_plot$Overall)],
        names.arg=fi_plot$Feature[order(fi_plot$Overall)],
        horiz=TRUE, las=1, col="#4472C4",
        main="Random Forest Feature Importance",
        xlab="Importance Score", cex.names=0.8)
dev.off()

cat("\n=== ALL DONE — real numbers computed ===\n")

# Save all key numbers to a results file for the report
sink("real_results.txt")
cat("=== SUMMARY STATS ===\n"); print(as.data.frame(stats))
cat("\n=== CORRELATIONS ===\n"); print(round(cor_outcome,4))
cat("\n=== MARGINAL PROBS ===\n")
cat("P(Diabetic)=",round(P_diab,4),"\nP(Non-Diabetic)=",round(P_nondiab,4),"\n")
cat("\n=== CONDITIONAL PROBS ===\n")
cat("P(D|High Glucose)=",round(P_d_high,4),"\nP(D|Normal Glucose)=",round(P_d_normal,4),"\n")
cat("Risk multiplier=",round(P_d_high/P_d_normal,2),"x\n")
cat("P_joint=",round(P_joint,4),"\n")
cat("\n=== BAYES ===\n")
cat("P(D|Obese)=",round(P_diab_given_obese,4),"\n")
cat("\n=== HYPOTHESIS TESTS ===\n")
cat("T1: t=",round(t1$statistic,3),"p=",format(t1$p.value,scientific=TRUE),"\n")
cat("T1: Mean_Diab=",round(mean(g_diab),2),"Mean_NonDiab=",round(mean(g_ndiab),2),"\n")
cat("T2: t=",round(t2$statistic,3),"p=",format(t2$p.value,scientific=TRUE),"\n")
cat("T2: Mean_BMI_Diab=",round(mean(b_diab),2),"Mean_BMI_NonDiab=",round(mean(b_ndiab),2),"\n")
cat("T2: Cohen_d=",round(cohens_d,4),"\n")
cat("T3: X2=",round(chisq_r$statistic,3),"df=",chisq_r$parameter,
    "p=",format(chisq_r$p.value,scientific=TRUE),"\n")
cat("T3: Cramers_V=",round(cramers_v,4),"\n")
cat("\n=== MODEL RESULTS ===\n"); print(results)
cat("\n=== RF TUNED THRESHOLD 0.40 ===\n")
cat("Sensitivity=",round(cm_rf_tuned$byClass["Sensitivity"],4),"\n")
cat("Specificity=",round(cm_rf_tuned$byClass["Specificity"],4),"\n")
cat("Accuracy=",round(cm_rf_tuned$overall["Accuracy"],4),"\n")
cat("F1=",round(cm_rf_tuned$byClass["F1"],4),"\n")
cat("\n=== ODDS RATIOS ===\n"); print(or)
cat("\n=== FEATURE IMPORTANCE ===\n"); print(fi)
sink()
cat("Results saved to real_results.txt\n")
