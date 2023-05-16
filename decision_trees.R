library(tidyverse)
library(tidymodels)
#library(flextable)
#library(DataExplorer)
#library(dlookr)
library(rpart.plot)

df <- read_rds("~/Desktop/R/met_age.rds")

df <- df %>% pivot_longer(-Age, names_to = "gene", values_to = "value") %>% 
  group_by(gene) %>% mutate(v = var(value)) %>% arrange(desc(v)) %>% 
  mutate(id = row_number(), .before = Age) %>% 
  select(1:4) %>% 
  pivot_wider(names_from = "gene", values_from = "value")
df <- df %>% select(2:2002)

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
folds <- vfold_cv(df_train, v = 10, strata = Age)

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

# Specify Decision Trees-------------------------------------
dt_spec <- 
	decision_tree(
		mode = "regression",
		engine = "rpart",
		cost_complexity = tune(),
		tree_depth = tune(),
		min_n = tune())

# Workflow
dt_wf <-
	workflow() %>%
	add_recipe(clean_rec) %>% 
	add_model(dt_spec)

# Grid
dt_grid <- grid_latin_hypercube(
	cost_complexity(),
	tree_depth(),
	min_n(),
	size = 15)

# Tune
doParallel::registerDoParallel()
set.seed(1234)
dt_tune <- dt_wf %>%
	tune_grid(folds,
						metrics = model_metrics,
						control = model_control,
						grid = dt_grid)

# Select best metric
dt_best <- dt_tune %>% select_best(metric = "rmse")
autoplot(dt_tune)

dt_best

# Train results
dt_train_results <- dt_tune %>%
	filter_parameters(parameters = dt_best) %>%
	collect_metrics()
dt_train_results

# Last fit
dt_test_results <- dt_wf %>% 
	finalize_workflow(dt_best) %>%
	last_fit(split = df_split, metrics = model_metrics)

dt_results <- dt_test_results %>% collect_metrics()

dt_results %>%
	select(-.config, -.estimator) %>%
	rename(metric = .metric,
				 Test_set = .estimate) %>% 
	arrange(desc(Test_set))

dt_test_results %>%
	extract_fit_engine() %>%
	rpart.plot(roundint = FALSE, cex = 0.8)

library(vetiver)
v <- dt_test_results %>%
  extract_workflow() %>%
  vetiver_model("price")
v

augment(v, slice_sample(df_test, n = 50)) %>%
  select(Age, .pred) %>% View()

library(plumber)
pr() %>% 
  vetiver_api(v) %>% 
  pr_run()

pr() %>% 
  vetiver_api(v, type = "prob") %>% 
  pr_run()