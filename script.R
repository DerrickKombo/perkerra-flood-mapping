# =============================================================
# RF-FS FLOOD SUSCEPTIBILITY MAPPING: PERKERRA BASIN
# =============================================================

# =============================================================
# STEP 0: INSTALL AND LOAD LIBRARIES
# =============================================================

required_pkgs <- c(
  "terra", "sf", "randomForest", "caret",
  "ggplot2", "dplyr", "reshape2", "pROC",
  "corrplot", "ranger", "mlr3verse", "mlr3fselect",
  "gridExtra", "grid"
)

for(pkg in required_pkgs){
  if(!requireNamespace(pkg, quietly = TRUE)){
    install.packages(pkg, dependencies = TRUE)
  }
}

library(terra)
library(sf)
library(randomForest)
library(caret)
library(ggplot2)
library(dplyr)
library(reshape2)
library(pROC)
library(corrplot)
library(ranger)
library(mlr3verse)
library(mlr3fselect)
library(gridExtra)
library(grid)

# Determine the available FSelectInstance class name
fsi_class <- if(exists("FSelectInstanceBatchSingleCrit")){
  "FSelectInstanceBatchSingleCrit"
} else if(exists("FSelectInstanceSingleCrit")){
  "FSelectInstanceSingleCrit"
} else {
  "fallback"
}

setwd("D:/_PROJECT/DATA/rasters/R_raster")

# =============================================================
# STEP 1: LOAD FLOOD INVENTORY
# =============================================================

inventory <- read.csv("perkerra_flood_inventory.csv")
inventory_sf <- st_as_sf(
  inventory,
  coords = c("longitude","latitude"),
  crs    = 4326
) %>% st_transform(32637)

# =============================================================
# STEP 2: LOAD CONDITIONING FACTOR RASTERS
# =============================================================

EV   <- rast("factors/Elevation.tif")
Slope <- rast("factors/Slope.tif")
AS   <- rast("factors/Aspect.tif")
PC   <- rast("factors/Plan_curvature.tif")
PrC  <- rast("factors/Profile_curvature.tif")
TWI  <- rast("factors/TWI.tif")
TPI  <- rast("factors/TPI.tif")
SPI  <- rast("factors/SPI.tif")
TRI  <- rast("factors/TRI.tif")
DtoS <- rast("factors/Distance_river.tif")
DtoL <- rast("factors/Distance_Lake.tif")
DD   <- rast("factors/Drainage_Density.tif")
PPT  <- rast("factors/Avg_Rain.tif")
LULC <- rast("factors/LULC.tif")
NDVI <- rast("factors/NDVI.tif")
DtoR <- rast("factors/Distance_Road.tif")
TXT  <- rast("factors/Soil_Text.tif")
LIHO <- rast("factors/Lithology.tif")

reference <- EV

# Align Drainage Density to reference grid
DD <- project(DD, reference, method = "bilinear")

factor_stack <- c(
  EV, Slope, AS, PC, PrC,
  TWI, TPI, SPI, TRI,
  DtoS, DtoL, DD,
  PPT, LULC, NDVI, DtoR,
  TXT, LIHO
)

factor_names <- c(
  "EV","Slope","AS","PC","PrC",
  "TWI","TPI","SPI","TRI",
  "DtoS","DtoL","DD",
  "PPT","LULC","NDVI","DtoR",
  "TXT","LIHO"
)

names(factor_stack) <- factor_names

# =============================================================
# STEP 3: EXTRACT FACTOR VALUES AT INVENTORY POINTS
# =============================================================

extracted <- extract(factor_stack, inventory_sf, ID = FALSE)

dataset             <- cbind(flood_label = inventory$class, extracted)
dataset             <- na.omit(dataset)
dataset$flood_label <- as.factor(dataset$flood_label)

# =============================================================
# STEP 4: NORMALISE FACTORS
# =============================================================

norm_params <- list() 

normalize_col <- function(x){
  rng <- range(x, na.rm = TRUE)
  if(diff(rng) == 0) return(list(values = x, min = rng[1], max = rng[2]))
  list(values = (x - rng[1]) / (rng[2] - rng[1]),
       min    = rng[1],
       max    = rng[2])
}

dataset_norm <- dataset
for(col in factor_names){
  if(col %in% names(dataset_norm)){
    result               <- normalize_col(dataset_norm[[col]])
    dataset_norm[[col]]   <- result$values
    norm_params[[col]]    <- list(min = result$min, max = result$max)
  }
}

# =============================================================
# STEP 5: TRAIN AND TEST SPLIT
# =============================================================

set.seed(42)
train_idx  <- createDataPartition(dataset_norm$flood_label,
                                  p = 0.80, list = FALSE)
train_data <- dataset_norm[ train_idx, ]
test_data  <- dataset_norm[-train_idx, ]

# =============================================================
# STEP 6: MULTICOLLINEARITY CHECK 
# =============================================================

numeric_train <- train_data %>% dplyr::select(-flood_label)

# Remove zero variance predictors
zero_var      <- sapply(numeric_train,
                        function(x) sd(x, na.rm = TRUE) == 0)
if(any(zero_var)){
  numeric_train <- numeric_train[, !zero_var]
}

cor_matrix            <- cor(numeric_train,
                             use    = "complete.obs",
                             method = "pearson")
cor_matrix[is.na(cor_matrix)] <- 0

# Save Pearson Correlation Matrix
png("perkerra_correlation_matrix.png",
    width = 1400, height = 1200, res = 150)
corrplot(cor_matrix,
         method      = "color",
         type        = "upper",
         tl.cex      = 0.75,
         number.cex  = 0.50,
         addCoef.col = "black",
         title       = "Pearson Correlation Matrix: Conditioning Factors",
         mar         = c(0,0,3,0))
dev.off()

threshold <- 0.85
high_cor  <- findCorrelation(cor_matrix, cutoff = threshold, names = TRUE)

remove_vars      <- unique(c(high_cor, names(zero_var[zero_var])))
train_filtered   <- train_data   %>% dplyr::select(-any_of(remove_vars))
test_filtered    <- test_data    %>% dplyr::select(-any_of(remove_vars))
dataset_filtered <- dataset_norm %>% dplyr::select(-any_of(remove_vars))

remaining_factors <- names(train_filtered)[
  names(train_filtered) != "flood_label"]

# -------------------------------------------------------------
# STEP 6B: VIF AND TOLERANCE CHECK
# -------------------------------------------------------------

vif_data <- numeric_train[, remaining_factors]
vif_data$flood_label_num <- as.numeric(
  as.character(train_data$flood_label[rownames(numeric_train) %in%
                                        rownames(vif_data)]))

lm_vif <- lm(flood_label_num ~ ., data = vif_data)

if(!requireNamespace("car", quietly = TRUE)){
  install.packages("car")
}
library(car)

vif_vals <- car::vif(lm_vif)

vif_table <- data.frame(
  Factor    = names(vif_vals),
  VIF       = round(as.numeric(vif_vals), 4),
  Tolerance = round(1 / as.numeric(vif_vals), 4)
)

vif_table <- vif_table[order(-vif_table$VIF), ]
vif_table$Multicollinearity <- ifelse(vif_table$VIF >= 10, "High: Remove",
                                      ifelse(vif_table$VIF >= 5,  "Moderate",
                                             "Acceptable"))

write.csv(vif_table, "perkerra_vif_tolerance.csv", row.names = FALSE)

# -------------------------------------------------------------
# STEP 6C: RETAINED CONDITIONING FACTORS
# -------------------------------------------------------------

removed_vif      <- vif_table$Factor[vif_table$VIF >= 10]
all_removed <- unique(c(high_cor, names(zero_var[zero_var]), removed_vif))
final_retained <- remaining_factors[!remaining_factors %in% removed_vif]

if(length(removed_vif) > 0){
  remove_vars      <- unique(c(remove_vars, removed_vif))
  train_filtered   <- train_data   %>% dplyr::select(-any_of(remove_vars))
  test_filtered    <- test_data    %>% dplyr::select(-any_of(remove_vars))
  remaining_factors <- names(train_filtered)[
    names(train_filtered) != "flood_label"]
}

# =============================================================
# STEP 7: SHADOW VARIABLE SEARCH (SVS)
# =============================================================

svs_success  <- FALSE
svs_selected <- character(0)
svs_archive  <- NULL

tryCatch({
  
  task_svs <- TaskClassif$new(
    id       = "flood_svs",
    backend  = as.data.frame(train_filtered),
    target   = "flood_label",
    positive = "1"
  )
  
  learner_svs <- lrn("classif.ranger",
                     predict_type = "prob",
                     importance   = "impurity",
                     num.trees    = 200,
                     seed         = 42L)
  
  resampling_svs <- rsmp("cv", folds = 10L)
  measure_svs    <- msr("classif.acc")
  fselector_svs  <- fs("sequential",
                       strategy     = "sfs",
                       max_features = length(remaining_factors))
  
  fsi_svs <- if(fsi_class == "FSelectInstanceBatchSingleCrit"){
    FSelectInstanceBatchSingleCrit$new(
      task        = task_svs,
      learner     = learner_svs,
      resampling  = resampling_svs,
      measure     = measure_svs,
      terminator  = trm("stagnation", iters = 3L, threshold = 0.001)
    )
  } else {
    FSelectInstanceSingleCrit$new(
      task        = task_svs,
      learner     = learner_svs,
      resampling  = resampling_svs,
      measure     = measure_svs,
      terminator  = trm("stagnation", iters = 3L, threshold = 0.001)
    )
  }
  
  set.seed(42)
  fselector_svs$optimize(fsi_svs)
  
  svs_archive  <- as.data.frame(fsi_svs$archive)
  svs_selected <- fsi_svs$result_feature_set
  svs_success  <- TRUE
  
}, error = function(e){
  cat("Fallback to caret sequential CV for SVS\n")
})

# Fallback SVS processing
if(!svs_success){
  set.seed(42)
  rf_init <- randomForest::randomForest(
    x          = as.matrix(train_filtered %>% dplyr::select(-flood_label)),
    y          = train_filtered$flood_label,
    ntree      = 200,
    importance = TRUE
  )
  
  imp_matrix <- randomForest::importance(rf_init, type = 1) 
  
  if("MeanDecreaseAccuracy" %in% colnames(imp_matrix)){
    imp_vec <- imp_matrix[, "MeanDecreaseAccuracy"]
  } else {
    imp_vec <- imp_matrix[, ncol(imp_matrix)]
  }
  
  imp_order <- names(sort(imp_vec, decreasing = TRUE))
  imp_order <- imp_order[imp_order %in% remaining_factors]
  
  cv_ctrl_svs <- trainControl(
    method      = "cv",
    number      = 10,
    verboseIter = FALSE
  )
  
  svs_curve_rows <- list()
  
  for(n in seq_along(imp_order)){
    top_vars <- head(imp_order, n)
    temp_df  <- train_filtered %>%
      dplyr::select(all_of(c("flood_label", top_vars)))
    
    acc_n <- tryCatch({
      rf_n <- train(
        flood_label ~ .,
        data      = temp_df,
        method    = "rf",
        trControl = cv_ctrl_svs,
        tuneGrid  = data.frame(mtry = max(1L, floor(sqrt(n)))),
        ntree     = 200
      )
      max(rf_n$results$Accuracy)
    }, error = function(e) NA_real_)
    
    svs_curve_rows[[n]] <- data.frame(
      n_features        = n,
      Selected_Features = paste(sort(top_vars), collapse = ", "),
      Accuracy          = round(acc_n, 3)
    )
  }
  
  svs_table3_fallback <- do.call(rbind, svs_curve_rows)
  svs_table3_fallback <- svs_table3_fallback[!is.na(svs_table3_fallback$Accuracy), ]
  
  best_n       <- svs_table3_fallback$n_features[which.max(svs_table3_fallback$Accuracy)]
  svs_selected <- head(imp_order, best_n)
  svs_archive  <- svs_table3_fallback
  svs_success  <- TRUE
}

if(!is.null(svs_archive) && "classif.acc" %in% names(svs_archive)){
  feat_cols <- remaining_factors[remaining_factors %in% names(svs_archive)]
  svs_t3 <- svs_archive %>%
    mutate(n_features = rowSums(
      dplyr::select(., all_of(feat_cols)), na.rm = TRUE)) %>%
    group_by(n_features) %>%
    slice_max(classif.acc, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    arrange(n_features)
  
  svs_t3$Selected_Features <- apply(
    svs_t3[, feat_cols], 1,
    function(row) paste(sort(feat_cols[as.logical(row)]), collapse = ", ")
  )
  
  svs_table3_clean <- svs_t3 %>%
    dplyr::select(
      Number_of_Features = n_features,
      Selected_Features,
      Accuracy           = classif.acc
    ) %>%
    mutate(Accuracy = round(Accuracy, 3))
  
} else {
  svs_table3_clean <- svs_archive %>%
    dplyr::rename(Number_of_Features = n_features)
}

write.csv(svs_table3_clean, "perkerra_table3_svs_features.csv", row.names = FALSE)

p_svs_curve <- ggplot(svs_table3_clean,
                      aes(x = Number_of_Features, y = Accuracy)) +
  geom_line(color = "#1a7a1e", linewidth = 1.3) +
  geom_point(color = "#1a7a1e", size = 4) +
  geom_vline(
    xintercept = svs_table3_clean$Number_of_Features[which.max(svs_table3_clean$Accuracy)],
    linetype = "dashed", color = "grey40", linewidth = 0.8
  ) +
  scale_x_continuous(breaks = seq(1, max(svs_table3_clean$Number_of_Features), 1)) +
  scale_y_continuous(limits = c(0.60, 1.0), breaks = seq(0.60, 1.0, 0.05)) +
  labs(title = "RF-SVS Accuracy", x = "Number of Features", y = "Accuracy") +
  theme_classic(base_size = 13) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

# =============================================================
# STEP 8: RECURSIVE FEATURE ELIMINATION (RFE)
# =============================================================

X_rfe <- train_filtered %>% dplyr::select(-flood_label)
y_rfe <- train_filtered$flood_label

customRF      <- rfFuncs
customRF$fit  <- function(x, y, first, last, ...){
  randomForest(x, y, ntree = 200, importance = TRUE, ...)
}

rfe_ctrl <- rfeControl(
  functions    = customRF,
  method       = "cv",
  number       = 10,
  saveDetails  = TRUE,
  returnResamp = "all",
  verbose      = FALSE
)

set.seed(42)
all_sizes <- seq(1, length(remaining_factors))

rfe_result <- rfe(
  x          = X_rfe,
  y          = y_rfe,
  sizes      = all_sizes,
  rfeControl = rfe_ctrl
)

rfe_selected <- predictors(rfe_result)

if(!is.null(rfe_result$fit)){
  full_imp <- varImp(rfe_result$fit, scale = FALSE)
  if(is.data.frame(full_imp)){
    full_ranked <- rownames(full_imp)[order(full_imp[,1], decreasing = TRUE)]
  } else {
    full_ranked <- rownames(full_imp$importance)[
      order(full_imp$importance[,1], decreasing = TRUE)]
  }
} else {
  full_ranked <- rfe_result$variables %>%
    group_by(var) %>%
    summarise(MeanImp = mean(Overall, na.rm = TRUE)) %>%
    arrange(desc(MeanImp)) %>%
    pull(var)
}

rfe_table4 <- data.frame(
  Number_of_Selected_Features = rfe_result$results$Variables,
  Accuracy = round(rfe_result$results$Accuracy, 3)
)

rfe_table4$Selected_Features <- sapply(
  rfe_table4$Number_of_Selected_Features,
  function(n){
    paste(sort(head(full_ranked, n)), collapse = ", ")
  }
)

rfe_table4 <- rfe_table4[order(-rfe_table4$Number_of_Selected_Features), ]
write.csv(rfe_table4, "perkerra_table4_rfe_features.csv", row.names = FALSE)

rfe_curve_data <- data.frame(
  N_Features = rfe_result$results$Variables,
  Accuracy   = rfe_result$results$Accuracy
)

p_rfe_curve <- ggplot(rfe_curve_data,
                      aes(x = N_Features, y = Accuracy)) +
  geom_line(color = "#e65100", linewidth = 1.3) +
  geom_point(color = "#e65100", size = 4) +
  geom_vline(xintercept = rfe_result$optsize,
             linetype = "dashed", color = "grey40", linewidth = 0.8) +
  scale_x_reverse(breaks = seq(max(rfe_curve_data$N_Features), 1, -1)) +
  scale_y_continuous(limits = c(0.60, 1.0), breaks = seq(0.60, 1.0, 0.05)) +
  labs(title = "RF-RFE Accuracy", x = "Number of Features", y = "Accuracy") +
  theme_classic(base_size = 13) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

combined_fig4 <- grid.arrange(
  p_svs_curve, p_rfe_curve, ncol = 2,
  top = textGrob("Accuracy vs Number of Features: RF-FS Framework",
                 gp = gpar(fontface = "bold", fontsize = 14))
)

ggsave("perkerra_fig4_accuracy_vs_features.png",
       plot = combined_fig4, dpi = 300, width = 13, height = 6)

# =============================================================
# STEP 9: CONSENSUS KEY DRIVING FACTORS
# =============================================================

kdf_consensus <- intersect(svs_selected, rfe_selected)
svs_only      <- setdiff(svs_selected, rfe_selected)
rfe_only      <- setdiff(rfe_selected, svs_selected)

final_kdfs <- if(length(kdf_consensus) < 4){
  union(kdf_consensus, svs_only)
} else {
  kdf_consensus
}

train_kdf             <- train_filtered %>%
  dplyr::select(all_of(c("flood_label", final_kdfs)))
test_kdf              <- test_filtered  %>%
  dplyr::select(all_of(c("flood_label", final_kdfs)))
train_kdf$flood_label <- as.factor(train_kdf$flood_label)
test_kdf$flood_label  <- as.factor(test_kdf$flood_label)
X_train_kdf           <- train_kdf %>% dplyr::select(-flood_label)
y_train_kdf           <- train_kdf$flood_label
X_test_kdf            <- test_kdf  %>% dplyr::select(-flood_label)
y_test_kdf            <- test_kdf$flood_label

# =============================================================
# STEP 10: HYPERPARAMETER TUNING
# =============================================================

tuning_success <- FALSE
optimal_mtry   <- max(2L, floor(sqrt(length(final_kdfs))))
optimal_ntree  <- 200L

tryCatch({
  
  task_tune <- TaskClassif$new(
    id       = "perkerra_tune",
    backend  = as.data.frame(train_kdf),
    target   = "flood_label",
    positive = "1"
  )
  
  learner_tune    <- lrn("classif.ranger",
                         predict_type = "prob",
                         importance   = "impurity")
  resampling_tune <- rsmp("cv", folds = 10L)
  
  param_set <- tryCatch(
    ps(
      mtry      = p_int(lower = 2L, upper = length(final_kdfs)),
      num.trees = p_int(lower = 100L, upper = 500L)
    ),
    error = function(e){
      ParamSet$new(list(
        ParamInt$new("mtry",      lower = 2L,   upper = length(final_kdfs)),
        ParamInt$new("num.trees", lower = 100L, upper = 500L)
      ))
    }
  )
  
  instance <- ti(
    task         = task_tune,
    learner      = learner_tune,
    resampling   = resampling_tune,
    measure      = msr("classif.acc"),
    search_space = param_set,
    terminator   = trm("evals", n_evals = 50L)
  )
  
  tuner <- tnr("random_search")
  set.seed(42)
  tuner$optimize(instance)
  
  best_params    <- instance$result_learner_param_vals
  optimal_mtry   <- as.integer(best_params$mtry)
  optimal_ntree  <- as.integer(best_params$num.trees)
  tuning_archive <- as.data.frame(instance$archive$data)
  best_acc_tune  <- max(tuning_archive$classif.acc, na.rm = TRUE)
  best_ce_tune   <- 1 - best_acc_tune
  tuning_success <- TRUE
  
}, error = function(e){
  
  cv_tune <- trainControl(method = "cv", number = 10,
                          verboseIter = FALSE)
  mtry_grid <- data.frame(mtry = seq(2, length(final_kdfs)))
  
  set.seed(42)
  rf_tune <- train(
    flood_label ~ .,
    data      = train_kdf,
    method    = "rf",
    trControl = cv_tune,
    tuneGrid  = mtry_grid,
    ntree     = 200
  )
  
  optimal_mtry  <<- rf_tune$bestTune$mtry
  optimal_ntree <<- 200L
  best_acc_tune <<- max(rf_tune$results$Accuracy)
  best_ce_tune  <<- 1 - best_acc_tune
  tuning_success <<- TRUE
})

# =============================================================
# STEP 11: TRAIN FINAL MODEL ON FULL TRAINING SET
# =============================================================

set.seed(42)
rf_final <- ranger(
  flood_label ~ .,
  data        = train_kdf,
  num.trees   = optimal_ntree,
  mtry        = optimal_mtry,
  importance  = "impurity",
  probability = TRUE,
  seed        = 42L
)

# =============================================================
# STEP 12: TEN FOLD CROSS VALIDATION ON TRAINING SET
# =============================================================

y_cv <- factor(
  ifelse(as.character(y_train_kdf) == "1", "Flood", "NonFlood"),
  levels = c("Flood","NonFlood")
)

cv_ctrl <- trainControl(
  method          = "cv",
  number          = 10,
  savePredictions = "final",
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter     = FALSE
)

set.seed(42)
rf_cv <- train(
  x         = X_train_kdf,
  y         = y_cv,
  method    = "rf",
  ntree     = optimal_ntree,
  tuneGrid  = data.frame(mtry = optimal_mtry),
  trControl = cv_ctrl,
  metric    = "ROC"
)

fold_res       <- rf_cv$resample
fold_res$Fold  <- paste0("Fold", sprintf("%02d", seq_len(nrow(fold_res))))
fold_res$AUC  <- round(fold_res$ROC, 3)
fold_res$CE   <- round(1 - fold_res$Sens, 3) 
fold_res$ROC   <- round(fold_res$ROC, 3)
fold_res$Sens  <- round(fold_res$Sens, 3)
fold_res$Spec  <- round(fold_res$Spec, 3)

avg_fold <- data.frame(
  Fold = "Average",
  AUC  = round(mean(fold_res$AUC), 3),
  CE   = round(mean(fold_res$CE), 3),
  ROC  = round(mean(fold_res$ROC), 3),
  Sens = round(mean(fold_res$Sens), 3),
  Spec = round(mean(fold_res$Spec), 3)
)

fold_report <- rbind(
  fold_res[, c("Fold","AUC","CE","ROC","Sens","Spec")],
  avg_fold
)

write.csv(fold_report, "perkerra_cv_fold_results.csv", row.names = FALSE)

# =============================================================
# STEP 13: FINAL EVALUATION ON TEST SET 
# =============================================================

test_preds <- predict(rf_final,
                      data = X_test_kdf,
                      type = "response")$predictions

if(is.matrix(test_preds)){
  flood_col       <- which(colnames(test_preds) == "1")
  test_flood_prob <- test_preds[, flood_col]
} else {
  test_flood_prob <- test_preds
}

test_pred_class  <- factor(
  ifelse(test_flood_prob >= 0.5, "1", "0"),
  levels = c("0","1")
)
test_true_factor <- factor(
  as.character(y_test_kdf),
  levels = c("0","1")
)

cm_test    <- confusionMatrix(test_pred_class, test_true_factor,
                              positive = "1")
OA_test    <- cm_test$overall["Accuracy"]
Kappa_test <- cm_test$overall["Kappa"]
Sens_test  <- cm_test$byClass["Sensitivity"]
Spec_test  <- cm_test$byClass["Specificity"]
F1_test    <- cm_test$byClass["F1"]

roc_obj    <- roc(as.integer(as.character(test_true_factor)),
                  test_flood_prob, quiet = TRUE)
auc_val    <- auc(roc_obj)

# -------------------------------------------------------------
# STEP 13B: ROC CURVE PLOT
# -------------------------------------------------------------

ci_obj  <- ci.se(roc_obj,
                 specificities = seq(0, 1, by = 0.01),
                 conf.level    = 0.95,
                 boot.n        = 500,
                 progress      = "none")

ci_df <- data.frame(
  FPR   = 1 - as.numeric(rownames(ci_df <- as.data.frame(ci_obj))),
  TPR   = ci_df[, 2],            
  lower = ci_df[, 1],            
  upper = ci_df[, 3]             
)

roc_df <- data.frame(
  FPR = 1 - roc_obj$specificities,
  TPR = roc_obj$sensitivities
)

youden_idx   <- which.max(roc_obj$sensitivities +
                            roc_obj$specificities - 1)
opt_point    <- data.frame(
  FPR = 1 - roc_obj$specificities[youden_idx],
  TPR = roc_obj$sensitivities[youden_idx]
)

opt_label <- sprintf("Optimal threshold\nSens = %.3f, Spec = %.3f",
                     roc_obj$sensitivities[youden_idx],
                     roc_obj$specificities[youden_idx])

auc_ci     <- ci.auc(roc_obj, method = "delong")
auc_label  <- sprintf(
  "AUC = %.3f (95%% CI: %.3f - %.3f)",
  as.numeric(auc_val),
  as.numeric(auc_ci[1]),
  as.numeric(auc_ci[3])
)

p_roc_pub <- ggplot() +
  geom_ribbon(
    data    = ci_df,
    aes(x = FPR, ymin = lower, ymax = upper),
    fill    = "#1565C0",
    alpha   = 0.12
  ) +
  geom_abline(
    slope     = 1, intercept = 0,
    linetype  = "longdash",
    linewidth = 0.55,
    color     = "grey45"
  ) +
  geom_line(
    data      = roc_df,
    aes(x = FPR, y = TPR),
    color     = "#1565C0",
    linewidth = 0.9
  ) +
  geom_point(
    data  = opt_point,
    aes(x = FPR, y = TPR),
    shape = 21,
    size  = 4.5,
    fill  = "#e53935",
    color = "white",
    stroke = 1.4
  ) +
  annotate(
    "text",
    x     = opt_point$FPR + 0.04,
    y     = opt_point$TPR - 0.055,
    label = opt_label,
    size  = 3.2,
    hjust = 0,
    color = "#b71c1c",
    fontface = "italic"
  ) +
  annotate(
    "label",
    x         = 0.97,
    y         = 0.06,
    label     = auc_label,
    size      = 3.8,
    hjust     = 1,
    vjust     = 0,
    fontface  = "bold",
    color     = "#1565C0",
    fill      = "white",
    label.size = 0.4
  ) +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.2),
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.2),
    expand = expansion(mult = c(0.01, 0.02))
  ) +
  labs(
    title    = "Receiver Operating Characteristic (ROC) Curve",
    subtitle = "RF-SVS Model",
    x        = "1 - Specificity (False Positive Rate)",
    y        = "Sensitivity (True Positive Rate)"
  ) +
  theme_classic(base_size = 13)

ggsave("perkerra_roc_curve_publication.png",
       plot   = p_roc_pub,
       dpi    = 600,
       width  = 7,
       height = 7,
       bg     = "white")

# =============================================================
# STEP 14: FEATURE IMPORTANCE 
# =============================================================

imp_vals <- rf_final$variable.importance

imp_df <- data.frame(
  Factor     = names(imp_vals),
  Importance = as.numeric(imp_vals)
) %>%
  filter(!is.na(Importance)) %>%
  mutate(Imp_pct = round(Importance / sum(Importance) * 100, 1)) %>%
  arrange(desc(Imp_pct))

write.csv(imp_df, "perkerra_kdf_importance.csv", row.names = FALSE)

# =============================================================
# STEP 15: GENERATE FLOOD SUSCEPTIBILITY MAP
# =============================================================

kdf_stack <- factor_stack[[final_kdfs]]

for(nm in final_kdfs){
  if(!is.null(norm_params[[nm]])){
    rng_min <- norm_params[[nm]]$min
    rng_max <- norm_params[[nm]]$max
    if((rng_max - rng_min) > 0){
      kdf_stack[[nm]] <- (kdf_stack[[nm]] - rng_min) /
        (rng_max - rng_min)
    }
  }
}

predfun <- function(model, data){
  preds <- predict(model, data = data, type = "response")$predictions
  if(is.matrix(preds)){
    col_idx <- which(colnames(preds) == "1")
    return(preds[, col_idx])
  }
  return(preds)
}

flood_prob <- terra::predict(
  kdf_stack,
  rf_final,
  fun      = predfun,
  na.rm    = TRUE,
  filename = "perkerra_flood_probability.tif",
  overwrite= TRUE
)

names(flood_prob) <- "flood_probability"

# =============================================================
# STEP 16: CLASSIFY INTO SUSCEPTIBILITY ZONES
# =============================================================

prob_vals <- values(flood_prob, na.rm = TRUE)

if(!requireNamespace("classInt", quietly = TRUE)){
  install.packages("classInt")
}
library(classInt)

set.seed(42)
jenks <- classIntervals(
  prob_vals,
  n      = 5,
  style  = "jenks",
  warnSmallN = FALSE
)

breaks <- jenks$brks   

flood_zones <- classify(
  flood_prob,
  rcl = matrix(c(
    breaks[1], breaks[2], 1,  
    breaks[2], breaks[3], 2,   
    breaks[3], breaks[4], 3,   
    breaks[4], breaks[5], 4,   
    breaks[5], breaks[6], 5    
  ), ncol = 3, byrow = TRUE),
  include.lowest = TRUE
)

levels(flood_zones) <- data.frame(
  ID    = 1:5,
  Class = c("Extremely Low","Low","Moderate","High","Extremely High")
)

# =============================================================
# STEP 17: EXPORT RASTERS AND AREA STATISTICS
# =============================================================

writeRaster(flood_prob,
            "perkerra_flood_probability_final.tif",
            overwrite = TRUE, datatype = "FLT4S")
writeRaster(flood_zones,
            "perkerra_flood_susceptibility_zones_final.tif",
            overwrite = TRUE, datatype = "INT1U")

freq_z   <- freq(flood_zones)
pix_area <- (res(flood_zones)[1])^2

zlabels <- c("1" = "Extremely Low",
             "2" = "Low",
             "3" = "Moderate",
             "4" = "High",
             "5" = "Extremely High")

zone_stats <- data.frame(
  ID      = freq_z$value,
  Zone    = zlabels[as.character(freq_z$value)],
  Pixels  = freq_z$count,
  Area_ha  = round(freq_z$count * pix_area / 10000, 2),
  Area_km2 = round(freq_z$count * pix_area / 1e6,   4),
  Percent  = round(freq_z$count / sum(freq_z$count) * 100, 2)
)

write.csv(zone_stats, "perkerra_zone_areas.csv", row.names = FALSE)

# =============================================================
# STEP 18: SAVE MODEL 
# =============================================================

saveRDS(rf_final, "perkerra_rf_flood_model_final.rds")

saveRDS(list(
  final_kdfs     = final_kdfs,
  optimal_mtry   = optimal_mtry,
  optimal_ntree  = optimal_ntree,
  train_acc      = round(best_acc_tune, 3),
  train_ce       = round(best_ce_tune, 3),
  test_acc       = round(OA_test, 3),
  test_f1        = round(F1_test, 3),
  test_sens      = round(Sens_test, 3),
  test_spec      = round(Spec_test, 3),
  test_auc       = round(auc_val, 3),
  kappa          = round(Kappa_test, 4),
  zone_breaks    = round(breaks, 4),
  norm_params    = norm_params
), "perkerra_model_summary.rds")
