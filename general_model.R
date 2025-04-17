#🔢 1. Load Required Libraries
library(tidyverse)
library(tidymodels)

set.seed(123)  # For reproducibility

#📦 2. Load and Explore the Data

data(mtcars)
glimpse(mtcars)

#✂️ 3. Split the Data

data_split <- initial_split(mtcars, prop = 0.8)
train_data <- training(data_split)
test_data <- testing(data_split)

#🧹 4. Create a Recipe for Preprocessing

reg_recipe <- recipe(mpg ~ ., data = train_data) %>%
  step_normalize(all_predictors())

#🧠 5. Specify a Regression Model

lm_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

# rf_model <- rand_forest(trees = 500) %>%
#   set_engine("ranger") %>%
#   set_mode("regression")

#🔗 6. Create a Workflow

reg_workflow <- workflow() %>%
  add_model(lm_model) %>%
  add_recipe(reg_recipe)

#🚂 7. Fit the Model

reg_fit <- fit(reg_workflow, data = train_data)

#📊 8. Evaluate the Model

reg_preds <- predict(reg_fit, test_data) %>%
  bind_cols(test_data)

metrics(reg_preds, truth = mpg, estimate = .pred)
#------------------------------------------------
#🎯 9. Tune Hyperparameters (for models like RF)

# Use CV folds
folds <- vfold_cv(train_data, v = 5)

# Specify a tunable RF model
rf_tune_model <- rand_forest(mtry = tune(), min_n = tune(), trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# Update workflow
rf_tune_workflow <- workflow() %>%
  add_model(rf_tune_model) %>%
  add_recipe(reg_recipe)

# Create grid
rf_grid <- grid_regular(mtry(range = c(1, 10)),
                        min_n(range = c(2, 10)),
                        levels = 5)

# Tune
tune_results <- tune_grid(
  rf_tune_workflow,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(rmse)
)

# Best parameters
show_best(tune_results, "rmse")

#✅ 10. Finalize the Model with Best Params

best_params <- select_best(tune_results, "rmse")

final_rf <- finalize_workflow(rf_tune_workflow, best_params)

final_fit <- fit(final_rf, data = train_data)

#🧪 11. Test on Holdout Set

final_preds <- predict(final_fit, test_data) %>%
  bind_cols(test_data)

metrics(final_preds, truth = mpg, estimate = .pred)