#Training results
train_results <- bind_rows("Random Forest" = rf_train_results,
                           #"XGboost" = xgb_train_results,
                           #"GAM" = gam_train_results,
                           #"KNN" = knn_train_results,
                           #"SVM" = svm_train_results,
                           "MARS" = mars_train_results,
                           "Decision Trees" = dt_train_results,
                           #"MLP" = mlp_train_results,
                           .id = "model")

# Plot results
train_results %>%
  mutate(model = fct_reorder(model, mean)) %>%
  ggplot(aes(.metric, mean, fill = model)) +
  geom_col(position = "dodge") + 
  geom_text(
    aes(x = .metric, y = mean , label = round(mean, 3), group = model),
    position = position_dodge(width = 1), vjust = -0.3, size = 3.5) +
  scale_fill_brewer(palette = "Set1") +
  labs(y = "value (mean)", x = "metric", title = "Results on training set") +
  facet_wrap(~ .metric, drop = T, scales = "free_y")
#--------------------------------------------------------------------
#Testing results
test_results <- bind_rows("Random Forest" = rf_results,
                          #"XGboost" = xgb_results,
                          #"GAM" = gam_results,
                          #"KNN" = knn_results,
                          #"SVM" = svm_results,
                          "MARS" = mars_results,
                          "Decision Trees" = dt_results,
                          #"MLP" = mlp_results,
                          .id = "model")

# Plot results
test_results %>%
  mutate(model = fct_reorder(model, .estimate)) %>%
  ggplot(aes(.metric, .estimate, fill = model)) +
  geom_col(position = "dodge") + 
  geom_text(
    aes(x = .metric, y = .estimate , label = round(.estimate, 3), group = model),
    position = position_dodge(width = 1), vjust = -0.3, size = 3.5) +
  scale_fill_brewer(palette = "Set1") +
  labs(y = "value", x = "metric", title = "Results on test set") +
  facet_wrap(~ .metric, drop = T, scales = "free_y")
