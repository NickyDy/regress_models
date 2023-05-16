library(tidyverse)
library(tidymodels)
#library(flextable)
#library(DataExplorer)
#library(dlookr)
library(vip)
library(rpart.plot)
library(themis)
library(xgboost)

df <- read_csv("diss/diss.csv") %>% select(s_ha, elev, slope, invasive, rare, herb_cover, perennials, annuals)

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
df_split <- initial_split(df, strata = s_ha)
df_train <- training(df_split)
df_test <- testing(df_split)

# The validation set via K-fold cross validation of 5 validation folds
set.seed(2020)
folds <- vfold_cv(df_train, strata = s_ha)

# Recipe
clean_rec <- recipe(s_ha ~ ., data = df_split) %>%
	step_normalize(all_numeric_predictors())

train_prep <- clean_rec %>% prep() %>% juice()
glimpse(train_prep)

# Control and metrics
model_control <- control_grid(save_pred = TRUE)
model_metrics <- metric_set(rmse, rsq, mae, huber_loss)

# Specify XGB-------------------------------------
xgb_spec <- 
	boost_tree(mtry = tune(),
						 trees = tune(),
						 tree_depth = tune(),
						 learn_rate = tune()) %>%
	set_engine("xgboost") %>% 
	set_mode("regression")

# Workflow
xgb_wf <-
	workflow() %>%
	add_recipe(clean_rec) %>% 
	add_model(xgb_spec) 

# Grid
xgb_grid <- grid_latin_hypercube(
	trees(),
	tree_depth(),
	learn_rate(),
	finalize(mtry(), df_train),
	size = 15)

# Tune
doParallel::registerDoParallel()
set.seed(1234)
xgb_tune <- xgb_wf %>%
	tune_grid(folds,
						metrics = model_metrics,
						control = model_control,
						grid = xgb_grid)

# Select best metric
xgb_best <- xgb_tune %>% select_best(metric = "rmse")
autoplot(xgb_tune)

xgb_best

# Train results
xgb_train_results <- xgb_tune %>%
	filter_parameters(parameters = xgb_best) %>%
	collect_metrics()
xgb_train_results

# Last fit
xgb_test_results <- xgb_wf %>% 
	finalize_workflow(xgb_best) %>%
	last_fit(split = df_split, metrics = model_metrics)

xgb_results <- xgb_test_results %>% collect_metrics()
xgb_results
#----------
xgb_fit <- xgb_wf %>%
	finalize_workflow(xgb_best) %>%
	fit(df_test)
xgb_fit

pred_xgb<-predict(xgb_fit, new_data = validate, type = NULL)
pred_xgb

#XGboost Results
xgb_results %>%
	select(-.config, -.estimator) %>%
	rename(metric = .metric,
				 Test_set = .estimate) %>% 
	arrange(desc(Test_set))
