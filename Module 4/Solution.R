# Import required libraries
library(ISLR)
library(glmnet)
library(caret)
library(dplyr)
library(tidyr)
library(officer)
library(flextable)

# Load the College dataset
data("College")
set.seed(42) # For reproducibility

# Feature Engineering: Adding new features
College$Apps_per_Enroll <- College$Apps / College$Enroll
College$AcceptRate <- College$Accept / College$Apps
College$Expend_per_Term <- College$Expend / College$Terminal

# Additional Features: Interaction and polynomial terms
College$Interaction1 <- College$AcceptRate * College$Outstate
College$Interaction2 <- College$PhD * College$Top10perc
College$Poly_Outstate <- College$Outstate^2
College$Log_Room_Board <- log(College$Room.Board + 1)

# Splitting the dataset into training and testing sets
train_index <- createDataPartition(College$Grad.Rate, p = 0.7, list = FALSE)
train_data <- College[train_index, ]
test_data <- College[-train_index, ]

# Prepare data for glmnet 
x_train <- model.matrix(Grad.Rate ~ ., data = train_data)[, -1]
y_train <- train_data$Grad.Rate
x_test <- model.matrix(Grad.Rate ~ ., data = test_data)[, -1]
y_test <- test_data$Grad.Rate

### Ridge Regression ###
# Ridge Regression (alpha = 0)
ridge_model <- cv.glmnet(x_train, y_train, alpha = 0)
lambda_min_ridge <- ridge_model$lambda.min
lambda_1se_ridge <- ridge_model$lambda.1se

# Save Ridge CV plot
png("ridge_cv_plot.png")
plot(ridge_model)
dev.off()

# Ridge iteration-wise coefficients
# Coefficients matrix
ridge_beta <- as.matrix(ridge_model$glmnet.fit$beta)
# Convert to data frame
ridge_iterations <- as.data.frame(ridge_beta)
# Add feature names
ridge_iterations <- tibble::rownames_to_column(ridge_iterations, "Feature")

# Reshape to long format for iteration-wise output
ridge_iterations_long <- ridge_iterations %>%
  pivot_longer(
    cols = -Feature,
    names_to = "Iteration",
    values_to = "Coefficient"
  )
ridge_iterations_long$Lambda <- rep(ridge_model$glmnet.fit$lambda,
                                    each = nrow(ridge_iterations))

# Fit Ridge model with lambda.min
ridge_fit <- glmnet(x_train, y_train, alpha = 0, lambda = lambda_min_ridge)
ridge_coefficients <- coef(ridge_fit)

# Ridge RMSE for training and testing sets
ridge_pred_train <- predict(ridge_fit, s = lambda_min_ridge, newx = x_train)
ridge_rmse_train <- sqrt(mean((y_train - ridge_pred_train)^2))

ridge_pred_test <- predict(ridge_fit, s = lambda_min_ridge, newx = x_test)
ridge_rmse_test <- sqrt(mean((y_test - ridge_pred_test)^2))
ridge_perf <- postResample(pred = ridge_pred_test, obs = y_test)

### LASSO ###
# LASSO Regression (alpha = 1)
lasso_model <- cv.glmnet(x_train, y_train, alpha = 1)
lambda_min_lasso <- lasso_model$lambda.min
lambda_1se_lasso <- lasso_model$lambda.1se

# Save LASSO CV plot
png("lasso_cv_plot.png")
plot(lasso_model)
dev.off()

# LASSO iteration-wise coefficients
# Coefficients matrix
lasso_beta <- as.matrix(lasso_model$glmnet.fit$beta)
# Convert to data frame
lasso_iterations <- as.data.frame(lasso_beta)
# Add feature names
lasso_iterations <- tibble::rownames_to_column(lasso_iterations, "Feature")

# Reshape to long format for iteration-wise output
lasso_iterations_long <- lasso_iterations %>%
  pivot_longer(
    cols = -Feature,
    names_to = "Iteration",
    values_to = "Coefficient"
  )
lasso_iterations_long$Lambda <- rep(lasso_model$glmnet.fit$lambda,
                                    each = nrow(lasso_iterations))

# Fit LASSO model with lambda.min
lasso_fit <- glmnet(x_train, y_train, alpha = 1, lambda = lambda_min_lasso)
lasso_coefficients <- coef(lasso_fit)

# LASSO RMSE for training and testing sets
lasso_pred_train <- predict(lasso_fit, s = lambda_min_lasso, newx = x_train)
lasso_rmse_train <- sqrt(mean((y_train - lasso_pred_train)^2))

lasso_pred_test <- predict(lasso_fit, s = lambda_min_lasso, newx = x_test)
lasso_rmse_test <- sqrt(mean((y_test - lasso_pred_test)^2))
lasso_perf <- postResample(pred = lasso_pred_test, obs = y_test)

### Comparison and Stepwise Feature Selection ###
# Compare Ridge and LASSO performance
comparison <- data.frame(
  Model = c("Ridge", "LASSO"),
  RMSE = c(ridge_perf[1], lasso_perf[1]),
  R_Squared = c(ridge_perf[2], lasso_perf[2]),
  MAE = c(ridge_perf[3], lasso_perf[3])
)

# Stepwise feature selection using AIC
stepwise_model <- step(lm(Grad.Rate ~ ., data = train_data), direction = "both")

# Evaluate Stepwise Model
stepwise_pred_train <- predict(stepwise_model, newdata = train_data)
stepwise_rmse_train <- sqrt(mean((y_train - stepwise_pred_train)^2))

stepwise_pred_test <- predict(stepwise_model, newdata = test_data)
stepwise_rmse_test <- sqrt(mean((y_test - stepwise_pred_test)^2))
stepwise_perf <- postResample(pred = stepwise_pred_test, obs = y_test)

# Add stepwise model to comparison
comparison <- rbind(
  comparison,
  data.frame(
    Model = "Stepwise",
    RMSE = stepwise_perf[1],
    R_Squared = stepwise_perf[2],
    MAE = stepwise_perf[3]
  )
)

### Save all results to Word file ###
# Create tables for coefficients and iteration-wise outputs
ridge_coefficients_df <- as.data.frame(as.matrix(ridge_coefficients))
colnames(ridge_coefficients_df) <- "Coefficient"
ridge_coef_df <- tibble::rownames_to_column(ridge_coefficients_df, "Variable")

lasso_coef_df <- as.data.frame(as.matrix(lasso_coefficients))
colnames(lasso_coef_df) <- "Coefficient"
lasso_coefficients_df <- tibble::rownames_to_column(lasso_coef_df, "Variable")

# Create Word document
doc <- read_docx()
doc <- doc %>%
  body_add_par("Model Performance Comparison", style = "heading 1") %>%
  body_add_flextable(flextable(comparison)) %>%
  body_add_par("Ridge Regression Coefficients", style = "heading 2") %>%
  body_add_flextable(flextable(ridge_coef_df)) %>%
  body_add_par("LASSO Regression Coefficients", style = "heading 2") %>%
  body_add_flextable(flextable(lasso_coefficients_df)) %>%
  body_add_par("Iteration-wise Coefficients for Ridge Regression",
               style = "heading 2") %>%
  # Add a sample of Ridge iterations
  body_add_flextable(flextable(head(ridge_iterations_long, 50))) %>%
  body_add_par("Iteration-wise Coefficients for LASSO Regression",
               style = "heading 2") %>%
  # Add a sample of LASSO iterations
  body_add_flextable(flextable(head(lasso_iterations_long, 50)))

print(doc, target = "model_comparison_with_iterations.docx")


### Results Output ###
print("Lambda values for Ridge:")
print(paste("Lambda.min:", lambda_min_ridge, "Lambda.1se:", lambda_1se_ridge))

print("Lambda values for LASSO:")
print(paste("Lambda.min:", lambda_min_lasso, "Lambda.1se:", lambda_1se_lasso))

print("Model Performance Comparison:")
print(comparison)
