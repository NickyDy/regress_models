library(tidyverse)
library(tidymodels)

df <- read_csv("mdata.csv") %>% janitor::clean_names()

df <- df %>% pivot_longer(2:34, names_to = "locality", values_to = "absorbance") %>% 
	mutate(locality = str_replace_all(locality, "^local_[:digit:]+", "local"),
				 locality = str_replace_all(locality, "^nonlocal_[:digit:]+", "nonlocal")) %>% 
	relocate(locality, .after = absorbance)

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
df_split <- initial_split(df, strata = locality)
df_train <- training(df_split)
df_test <- testing(df_split)

# The validation set via K-fold cross validation of 5 validation folds
set.seed(2020)
folds <- vfold_cv(df_train, v = 5, strata = fcr_metric)

# Recipe
rec <-
	recipe(fcr_metric ~ ., data = df_train) %>%
	step_rm(employee_id, employment_month) %>% 
	step_corr(all_numeric_predictors()) %>% 
	step_normalize(all_numeric_predictors()) %>% 
	step_dummy(all_nominal_predictors()) %>% 
	step_naomit(all_predictors())
	

train_prep <- rec %>% prep() %>% juice()
glimpse(train_prep)

# Control and metrics
model_control <- control_grid(save_pred = TRUE)
model_metrics <- metric_set(rmse, rsq)

# Specify LOG REG---------------------------------------------
spec <- 
	linear_reg(
		mode = "regression", 
		engine = "glmnet", 
		penalty = tune(), 
		mixture = 1)

# Workflow
wf <- workflow() %>%
	add_recipe(rec) %>%
	add_model(title_spec)

reg_grid <- tibble(penalty = 10^seq(-3, -1, length.out = 60))

# Resampling
doParallel::registerDoParallel()
set.seed(123)
res <- 
	wf %>% 
	tune_grid(
		folds, 
		grid = reg_grid,
		control = model_control,
		metrics = model_metrics)

autoplot(res)

show_best(res, metric = "rmse")

select_best(res, desc(penalty), metric = "rmse")

final <-
	wf %>%
	finalize_workflow(select_by_pct_loss(res, desc(penalty), metric = "rmse")) %>%
	last_fit(df_split)

final

collect_metrics(final)

library(vip)
final %>%
	extract_fit_engine() %>%
	vi()

final %>% 
	pluck(".workflow", 1) %>%
	extract_fit_parsnip() %>% 
	vip(geom = "col", num_features = 10, horiz = TRUE, aesthetics = list(size = 4)) +
	labs(title = "Variable Importance - Linear Regression")

library(vetiver)
v <- final %>%
	extract_workflow() %>%
	vetiver_model("price")
v

augment(v, slice_sample(df_test, n = 10))

library(plumber)
pr() %>% 
	vetiver_api(v) %>% 
	pr_run()

pr() %>% 
	vetiver_api(v, type = "prob") %>% 
	pr_run()
