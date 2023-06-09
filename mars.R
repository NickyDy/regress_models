library(tidyverse)
library(tidymodels)
#library(flextable)
#library(DataExplorer)
#library(dlookr)

df <- read_rds("~/Desktop/R/met_age.rds")

df <- df %>% pivot_longer(-Age, names_to = "gene", values_to = "value") %>% 
  group_by(gene) %>% mutate(v = var(value)) %>% arrange(desc(v)) %>% 
  mutate(id = row_number(), .before = Age) %>% 
  select(1:4) %>% 
  pivot_wider(names_from = "gene", values_from = "value")
df <- df %>% select(2:2002)

glimpse(df)
df %>% count(room_type_code, sort = )
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
df_split <- initial_split(df, prop = 0.7, strata = Age)
df_train <- training(df_split)
df_test <- testing(df_split)

# V-кратното кръстосано валидиране (известно също като k-кратно кръстосано валидиране) 
# произволно разделя данните на V (в случая V = 10) групи с приблизително еднакъв размер.
set.seed(2020)
folds <- vfold_cv(df_train, v = 10, strata = Age)

# Recipe
mars_rec <- recipe(Age ~ ., data = df_train) %>%
  step_zv(all_predictors()) %>% 
  step_nzv(all_predictors()) %>% 
  step_corr(all_predictors()) %>% 
  step_log(base = exp(1), all_predictors()) %>% 
  step_normalize(all_predictors())

train_prep <- clean_rec %>% prep() %>% juice()
glimpse(train_prep)

# Control and metrics
model_control <- control_grid(save_pred = TRUE)

# metric_set() ви позволява да комбинирате множество 
# метрични функции заедно в нова функция, която изчислява всички наведнъж.
model_metrics <- metric_set(rmse, rsq)

# Мултивариационни адаптивни регресионни сплайнове (MARS)
mars_spec <- 
	mars(
		mode = "regression",
		engine = "earth",
		num_terms = tune(),
		prod_degree = tune(),
		prune_method = tune())

# Workflow
mars_wf <-
	workflow() %>%
	add_recipe(mars_rec) %>% 
	add_model(mars_spec)

# Grid
mars_grid <- 
	grid_latin_hypercube(
	finalize(num_terms(), df_train),
	prod_degree(),
	prune_method(),
	size = 10)

# Tune
doParallel::registerDoParallel()
set.seed(1234)
mars_tune <- mars_wf %>%
	tune_grid(folds,
						metrics = model_metrics,
						control = model_control,
						grid = mars_grid)

# Select best metric
mars_best <- mars_tune %>% select_best(metric = "rmse")
autoplot(mars_tune)

mars_best

# Train results
mars_train_results <- mars_tune %>%
	filter_parameters(parameters = mars_best) %>%
	collect_metrics()
mars_train_results

# Last fit
mars_test_results <- mars_wf %>% 
	finalize_workflow(mars_best) %>%
	last_fit(split = df_split, metrics = model_metrics)

mars_results <- mars_test_results %>% collect_metrics()
mars_results

library(vip)
mars_test_results %>%
	extract_fit_engine() %>%
	vi()

mars_test_results %>% 
	pluck(".workflow", 1) %>%
	extract_fit_parsnip() %>% 
	vip(geom = "col", num_features = 10, horiz = TRUE, aesthetics = list(size = 4)) +
	labs(title = "Variable Importance - MARS")

library(vetiver)
v <- mars_test_results %>%
	extract_workflow() %>%
	vetiver_model("met_age")
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
