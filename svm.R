library(tidyverse)
library(tidymodels)
library(lubridate)
library(tidytext)
library(textrecipes)
#library(flextable)
#library(DataExplorer)
#library(dlookr)
library(vip)
library(rpart.plot)
library(themis)
library(xgboost)

df <- read_csv("fcr.csv", col_types = "fcfffddddddddddddddd") %>% janitor::clean_names()
#------------------------------------------------
validate <- tibble(
	elev = 2327,
	slope = 13,
	invasive = 2,
	rare = 5,
	herb_cover = 92,
	perennials = 48,
	annuals = 29)

validate<-crossing(
	elev = c(1000, 1500, 2000, 2500),
	slope = c(10, 20, 30, 40),
	exp = c("N", "E", "S", "W"),
	top = c("concave", "flat", "convex"),
	rare = c(1,2,3,4),
	invasive = c(2,4,6,8),
	s_ha = c(10,20,30,40),
	s_m2 = c(5,10,15,20)) %>% slice_sample(n = 2000)

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
df_split <- initial_split(df, strata = fcr_metric)
df_train <- training(df_split)
df_test <- testing(df_split)

# The validation set via K-fold cross validation of 5 validation folds
set.seed(2020)
folds <- vfold_cv(df_train, v = 5, strata = fcr_metric)

# Recipe
title_rec <-
	recipe(fcr_metric ~ ., data = df_train) %>%
	step_rm(employee_id, employment_month) %>% 
	step_corr(all_numeric_predictors()) %>% 
	step_normalize(all_numeric_predictors()) %>% 
	step_dummy(all_nominal_predictors()) %>% 
	step_naomit(all_predictors())

train_prep <- clean_rec %>% prep() %>% juice()
glimpse(train_prep)

# Control and metrics
model_control <- control_grid(save_pred = TRUE)
model_metrics <- metric_set(rmse, rsq)

# Specify SVM-------------------------------------
svm_spec <- 
	svm_poly(
		mode = "regression",
		engine = "kernlab",
		cost = tune(),
		degree = tune(),
		scale_factor = tune())

# Workflow
svm_wf <-
	workflow() %>%
	add_recipe(title_rec) %>% 
	add_model(svm_spec)

# Grid
svm_grid <- grid_latin_hypercube(
	cost(),
	degree(),
	scale_factor(),
	size = 15)

# Tune
doParallel::registerDoParallel()
set.seed(1234)
svm_tune <- svm_wf %>%
	tune_grid(folds,
						metrics = model_metrics,
						control = model_control,
						grid = svm_grid)

# Select best metric
svm_best <- svm_tune %>% select_best(metric = "rmse")
autoplot(svm_tune)

svm_best

# Train results
svm_train_results <- svm_tune %>%
	filter_parameters(parameters = svm_best) %>%
	collect_metrics()
svm_train_results

# Last fit
svm_test_results <- svm_wf %>% 
	finalize_workflow(svm_best) %>%
	last_fit(split = df_split, metrics = model_metrics)

svm_results <- svm_test_results %>% collect_metrics()
svm_results
#----------
svm_fit <- svm_wf %>%
	finalize_workflow(svm_best) %>%
	fit(df_test)
svm_fit

pred_svm<-predict(svm_fit, new_data = validate, type = NULL)
pred_svm

#SVM Results
svm_results %>%
	select(-.config, -.estimator) %>%
	rename(metric = .metric,
				 Test_set = .estimate) %>% 
	arrange(desc(Test_set))