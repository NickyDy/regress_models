library(tidyverse)
library(tidymodels)

df <- read_csv("islr2/Carseats.csv") %>% janitor::clean_names()

glimpse(df)
diagnose_numeric(df) %>% flextable()
plot_intro(df)
plot_histogram(df)
plot_correlate(df)
df %>% plot_bar_category(top = 15)
df %>% plot_bar(by  = "result")
df %>% plot_boxplot(by = "result")
df %>% map_dfr(~sum(is.na(.)))
#----------------------------------------------
set.seed(123)
df_split <- initial_split(df, strata = sales)
df_train <- training(df_split)
df_test <- testing(df_split)

# The validation set via K-fold cross validation of 5 validation folds
set.seed(2020)
folds <- vfold_cv(df_train, strata = sales)

# Recipe
lr_rec <-
	recipe(sales ~ ., data = df_train) %>%
	step_dummy(all_nominal_predictors())
	
train_prep <- rec %>% prep() %>% juice()
glimpse(train_prep)

# Control and metrics
model_control <- control_grid(save_pred = TRUE)
model_metrics <- metric_set(rmse, rsq)

# Specify LIN REG---------------------------------------------
lr_spec <- 
	linear_reg(
		mode = "regression", 
		engine = "glmnet", 
		penalty = tune(), 
		mixture = 1)

# Workflow
lr_wf <-
  workflow() %>%
  add_recipe(lr_rec) %>% 
  add_model(lr_spec)

# Grid
lr_grid <- 
  grid_latin_hypercube(
    penalty(),
    size = 15)

# Tune
doParallel::registerDoParallel()
set.seed(1234)
lr_tune <- lr_wf %>%
  tune_grid(folds,
            metrics = model_metrics,
            control = model_control,
            grid = lr_grid)

# Select best metric
lr_best <- lr_tune %>% select_best(metric = "rmse")
autoplot(lr_tune)

lr_best

# Train results
lr_train_results <- lr_tune %>%
  filter_parameters(parameters = lr_best) %>%
  collect_metrics()
lr_train_results

# Last fit
lr_test_results <- lr_wf %>% 
  finalize_workflow(lr_best) %>%
  last_fit(split = df_split, metrics = model_metrics)

lr_results <- lr_test_results %>% collect_metrics()
lr_results

library(vip)
lr_test_results %>%
  extract_fit_engine() %>%
  vi()

lr_test_results %>% 
  pluck(".workflow", 1) %>%
  extract_fit_parsnip() %>% 
  vip(geom = "col", num_features = 10, horiz = TRUE, aesthetics = list(size = 4)) +
  labs(title = "Variable Importance - LR")

library(vetiver)
v <- lr_test_results %>%
  extract_workflow() %>%
  vetiver_model("met_age")
v

augment(v, slice_sample(df_test, n = 10)) %>%
  select(sales, .pred)

library(plumber)
pr() %>% 
  vetiver_api(v) %>% 
  pr_run()

pr() %>% 
  vetiver_api(v, type = "prob") %>% 
  pr_run()
