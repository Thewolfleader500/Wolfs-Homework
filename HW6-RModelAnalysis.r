install.packages(c("randomForest", "caret", "pROC"))

library(randomForest)
library(caret)
library(pROC)

set.seed(1)

# youre going to have to set this up to the path on your computer
# setwd("C:/Users/YOURNAMEHERE/Downloads/bc_data"), I removed my own PC name from this line because it's explicit!

# train_file <- "C:/Users/YOURNAMEHERE/Downloads/bc_data/data.breast-cancer-cells.train.txt"
# test_file  <- "C:/Users/YOURNAMEHERE/Downloads/bc_data/data.breast-cancer-cells.test.txt"

features <- c("radius_mean", "texture_mean", "perimeter_mean", "area_mean","smoothness_mean", "compactness_mean", "concavity_mean","concave_points_mean", "symmetry_mean", "fractal_dimension_mean","radius_se", "texture_se", "perimeter_se", "area_se","smoothness_se", "compactness_se", "concavity_se", "concave_points_se","symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
  "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst","concavity_worst", "concave_points_worst", "symmetry_worst","fractal_dimension_worst")



read_cell_data <- function(file) {
  # Skip the entire header line, then read the data as whitespace-separated tokens.
  tokens <- scan(
    file,
    what = character(),
    sep = "",
    quote = "",
    comment.char = "",
    skip = 1,
    quiet = TRUE
  )

  # Each data record should contain 32 values:
  # ID + diagnosis + 30 numeric features.
  if (length(tokens) %% 32 != 0) {
    stop(
      "Malformed data in: ", file,
      "\nTokens after header: ", length(tokens),
      "\nRemainder after grouping into 32-column records: ",
      length(tokens) %% 32,
      "\nCheck whether the file has extra header lines, blank metadata, or a different format."
    )
  }

  d <- as.data.frame(
    matrix(tokens, ncol = 32, byrow = TRUE),
    stringsAsFactors = FALSE
  )

  names(d) <- c("id", "diagnosis", features)

  d$diagnosis <- trimws(d$diagnosis)

  if (any(!d$diagnosis %in% c("B", "M"))) {
    stop(
      "Unexpected diagnosis values. Check the header and first few data rows."
    )
  }

  d$diagnosis <- factor(d$diagnosis, levels = c("B", "M"))
  d[features] <- lapply(d[features], as.numeric)

  if (anyNA(d[features])) {
    stop("Missing or non-numeric feature values detected.")
  }

  d
}



train <- read_cell_data(train_file)
test <- read_cell_data(test_file)

x_train <- train[, features, drop = FALSE]
y_train <- train$diagnosis

x_test <- test[, features, drop = FALSE]
y_test <- test$diagnosis

cat("\nDATA DIMENSIONS\n")
cat("Training rows:", nrow(train), "\n")
print(table(y_train))
cat("Test rows:", nrow(test), "\n")
print(table(y_test))

folds <- caret::createFolds(y = droplevels(y_train),k = 5,list = TRUE,returnTrain = FALSE)
grid <- expand.grid(ntree = c(100, 300),mtry = c(5, 15),nodesize = c(1, 3))
grid$mean_AUROC <- NA_real_
grid$sd_AUROC <- NA_real_

for (g in seq_len(nrow(grid))) {
  fold_auc <- numeric(length(folds))
  for (i in seq_along(folds)) {
    valid_idx <- folds[[i]]
    fit_idx <- setdiff(seq_along(y_train), valid_idx)
    fit <- randomForest(x = x_train[fit_idx, , drop = FALSE],y = y_train[fit_idx],ntree = grid$ntree[g],mtry = grid$mtry[g],nodesize = grid$nodesize[g])
    p_malignant <- predict(fit, x_train[valid_idx, , drop = FALSE], type = "prob")[, "M"]
    fold_auc[i] <- as.numeric(auc(roc(y_train[valid_idx], p_malignant,levels = c("B", "M"), direction = "<", quiet = TRUE)))
  }
  grid$mean_AUROC[g] <- mean(fold_auc)
  grid$sd_AUROC[g] <- sd(fold_auc)
}

grid <- grid[order(-grid$mean_AUROC, grid$sd_AUROC), ]
rownames(grid) <- NULL

cat("\nQ1 ranked by mean AUROC\n")
print(grid, row.names = FALSE, digits = 5)
best <- grid[1, ]
cat("\nSelected parameters:\n")
print(best[, c("ntree", "mtry", "nodesize", "mean_AUROC", "sd_AUROC")])

# Refit model on all training data.
rf.final <- randomForest(x = x_train, y = y_train,ntree = best$ntree, mtry = best$mtry, nodesize = best$nodesize,importance = TRUE)

# Question 2, Grab all notable values
evaluate <- function(x, y, label) {
  prob_m <- predict(rf.final, x, type = "prob")[, "M"]
  pred <- factor(ifelse(prob_m >= 0.5, "M", "B"), levels = c("B", "M"))
  actual <- factor(y, levels = c("B", "M"))
  cm <- table(Actual = actual, Predicted = pred)

  TN <- as.numeric(cm["B", "B"])
  FP <- as.numeric(cm["B", "M"])
  FN <- as.numeric(cm["M", "B"])
  TP <- as.numeric(cm["M", "M"])

  accuracy <- (TP + TN) / sum(cm)
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  precision <- ifelse(TP + FP == 0, 0, TP / (TP + FP))
  f1 <- ifelse(precision + sensitivity == 0, 0, 2 * precision * sensitivity / (precision + sensitivity))
  balanced_accuracy <- (sensitivity + specificity) / 2
  roc_obj <- roc(actual, prob_m, levels = c("B", "M"), direction = "<", quiet = TRUE)

  cat("\n", label, "CONFUSION MATRIX (rows=actual, columns=predicted)\n", sep = "")
  print(cm)
  cat("\n", label, "METRICS\n", sep = "")
  print(data.frame( Accuracy = accuracy, Balanced_Accuracy = balanced_accuracy, Sensitivity = sensitivity, Specificity = specificity, Precision = precision, F1 = f1, AUROC = as.numeric(auc(roc_obj))), row.names = FALSE, digits = 5)

  list(roc = roc_obj, accuracy = accuracy, auroc = as.numeric(auc(roc_obj)))
}

train_eval <- evaluate(x_train, y_train, "TRAINING")
test_eval  <- evaluate(x_test, y_test, "HELD-OUT TEST")

# Build an ROC Curve
plot(train_eval$roc, col = "blue", lwd = 2, main = "Random Forest ROC Curves", legacy.axes = TRUE)
plot(test_eval$roc, col = "red", lwd = 2, add = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright", legend = c(sprintf("Training AUROC = %.4f", train_eval$auroc), sprintf("Test AUROC = %.4f", test_eval$auroc), "No-discrimination reference"), col = c("blue", "red", "gray"), lty = c(1, 1, 2), lwd = c(2, 2, 1), bty = "n")

# Question 3, Comparing training and our previous test performances
cat("\nQuestion 3, Model Fit\n")
cat(sprintf("Training accuracy = %.4f; test accuracy = %.4f\n", train_eval$accuracy, test_eval$accuracy))
cat(sprintf("Training AUROC = %.4f; test AUROC = %.4f\n", train_eval$auroc, test_eval$auroc))
if (abs(train_eval$accuracy - test_eval$accuracy) < 0.05 &&
    abs(train_eval$auroc - test_eval$auroc) < 0.05) {
  cat("No substantial evidence of overfitting or underfitting.\n")
} else if (
    train_eval$accuracy > test_eval$accuracy + 0.05 ||
    train_eval$auroc > test_eval$auroc + 0.05
) {
  cat("Possible overfitting for training performance.\n")
} else {
  cat("Possible underfitting for training performance.\n")
}

# Question 4, or, mean Gini Impurities/Importance Metrics
importance_scores <- importance(rf.final, type = 2)
importance_table <- data.frame(
  Feature = rownames(importance_scores),
  MeanDecreaseGini = importance_scores[, 1]
)
importance_table <- importance_table[order(-importance_table$MeanDecreaseGini), ]

cat("\nTOP FIVE FEATURES BY MEAN DECREASE IN GINI\n")
print(head(importance_table, 5), row.names = FALSE, digits = 6)
