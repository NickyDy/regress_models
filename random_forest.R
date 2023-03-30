library(tidyverse)
library(tidymodels)
#library(flextable)
#library(DataExplorer)
#library(dlookr)

df <- read_rds("met_age.rds") %>% select(1:100)

glimpse(diss)
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
df_split <- initial_split(df, strata = Age)
df_train <- training(df_split)
df_test <- testing(df_split)

# The validation set via K-fold cross validation of 5 validation folds
set.seed(2020)
folds <- vfold_cv(df_train, strata = Age)

# Recipe
clean_rec <- recipe(Age ~ ., data = df_train) %>%
	step_zv(all_predictors()) %>% 
  step_nzv(all_predictors()) %>% 
  step_corr(all_predictors()) %>% 
  step_log(base = exp(1), all_predictors()) %>% 
  step_normalize(all_predictors())
  
train_prep <- clean_rec %>% prep() %>% juice()
glimpse(train_prep)

# Control and metrics
model_control <- control_grid(save_pred = TRUE)
model_metrics <- metric_set(rmse, rsq)

# Specify RF--------------------------------------
rf_spec <- 
	rand_forest(mtry = tune(),
							min_n = tune(),
							trees = tune()) %>%
	set_engine("ranger") %>% 
	set_mode("regression")

# Workflow
rf_wflow <-
	workflow() %>%
	add_recipe(clean_rec) %>% 
	add_model(rf_spec)

# Grid
rf_grid <- grid_latin_hypercube(
	finalize(mtry(), df_train),
	trees(),
	min_n(),
	size = 15)

# Tune
doParallel::registerDoParallel()
set.seed(1234)
rf_res <-
	rf_wflow %>% 
	tune_grid(
		resamples = folds, 
		metrics = model_metrics,
		control = model_control,
		grid = rf_grid)

# Select best metric
rf_best <- rf_res %>% select_best(metric = "rmse")
autoplot(rf_res)

rf_best

# Train results
rf_train_results <- rf_res %>% 
	filter_parameters(parameters = rf_best) %>% 
	collect_metrics()
rf_train_results

# Last fit
rf_test_res <- rf_wflow %>% 
	finalize_workflow(rf_best) %>%
	last_fit(split = df_split, metrics = model_metrics)

rf_results <- rf_test_res %>% collect_metrics()
rf_results %>%
	select(-.config, -.estimator) %>%
	rename(metric = .metric,
				 Test_set = .estimate) %>% 
	arrange(desc(Test_set))

library(vetiver)
v <- rf_test_res %>%
	extract_workflow() %>%
	vetiver_model("price")
v

augment(v, slice_sample(df_test, n = 100)) %>%
	select(Age, .pred)

library(plumber)
pr() %>% 
	vetiver_api(v) %>% 
	pr_run()

pr() %>% 
	vetiver_api(v, type = "prob") %>% 
	pr_run()