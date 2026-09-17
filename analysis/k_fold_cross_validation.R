library(reformulas, quietly = TRUE, warn.conflicts = FALSE)
library(lme4, quietly = TRUE, warn.conflicts = FALSE)
library(performance, quietly = TRUE, warn.conflicts = FALSE)
library(broom.mixed, quietly = TRUE, warn.conflicts = FALSE)

calc_test_loglik <- function(fit, newdata, response_var, family_obj) {
  mu <- predict(fit, newdata = newdata, type = "response",
                allow.new.levels = TRUE)
  y  <- newdata[[response_var]]
  fam <- family_obj$family
  
  ll <- switch(fam,
               gaussian = sum(dnorm(y, mean = mu, sd = sigma(fit), log = TRUE)),
               poisson = sum(dpois(y, lambda = mu, log = TRUE)),
               binomial = sum(dbinom(y, size = 1, prob = mu, log = TRUE)),
               Gamma = sum(dgamma(y, shape = 1 / sigma(fit)^2,
                                     rate = 1 / (sigma(fit)^2 * mu), log = TRUE)),
               beta = {
                 phi <- sigma(fit)
                 shape1 <- mu * phi
                 shape2 <- (1 - mu) * phi
                 sum(dbeta(y, shape1 = shape1, shape2 = shape2, log = TRUE))
               },
               stop("Add a case for family: ", fam)
  )
  ll
}

calc_test_rmse <- function(fit, newdata, response_var) {
  mu <- predict(fit, newdata = newdata, type = "response",
                allow.new.levels = TRUE)
  y  <- newdata[[response_var]]
  sqrt(mean((y - mu)^2, na.rm = TRUE))
}

make_null_formula <- function(model_formula) {
  re_terms <- reformulas::findbars(model_formula)
  re_part <- paste(sapply(re_terms, function(x) paste0("(", deparse(x), ")")),
                   collapse = " + ")
  as.formula(paste(all.vars(model_formula)[1], "~ 1 +", re_part))
}

model_cross_val <- function(base_formula, model_formula, df, group_var,
                            model_family, num_folds=10){
  
  # sample folds grouped by WorkerId
  groups <- unique(df[[group_var]])
  fold_map <- setNames(sample(rep(1:num_folds, length.out = length(groups))), groups)
  df$.fold <- fold_map[as.character(df[[group_var]])]
  
  cv_results <- vector("list", num_folds)
  fixef_results <- vector("list", num_folds)
  
  response_var <- all.vars(base_formula)[1]
  
  # create a dummy null model for r2 calculation
  null_formula <- make_null_formula(model_formula)
  
  for (i in 1:num_folds) {
    train_data <- df %>% filter(.fold != i)
    test_data  <- df %>% filter(.fold == i)
    
    null_fit <- glmmTMB(null_formula, data = train_data, family = model_family)
    base_fit <- glmmTMB(base_formula, data = train_data, family = model_family)
    model_fit <- glmmTMB(model_formula, data = train_data, family = model_family)
    
    ll0 <- calc_test_loglik(base_fit, test_data, response_var, model_family())/length(test_data)
    ll1 <- calc_test_loglik(model_fit, test_data, response_var, model_family())/length(test_data)
    delta_ll <- ll1 - ll0
    
    rmse0 <- calc_test_rmse(base_fit, test_data, response_var)
    rmse1 <- calc_test_rmse(model_fit, test_data, response_var)
    delta_rmse <- rmse1 - rmse0   # negative = m1 has lower (better) error
    
    r2_0 <- r2(base_fit, null_model = null_fit)
    r2_1 <- r2(model_fit, null_model = null_fit)
    
    cv_results[[i]] <- data.frame(
      fold = i,
      ll0 = ll0, ll1 = ll1, delta_ll = delta_ll,
      rmse0 = rmse0, rmse1 = rmse1, delta_rmse = delta_rmse,
      r2m_m0 = as.numeric(r2_0$R2_marginal),  r2c_m0 = as.numeric(r2_0$R2_conditional),
      r2m_m1 = as.numeric(r2_1$R2_marginal),  r2c_m1 = as.numeric(r2_1$R2_conditional),
      delta_r2m = as.numeric(r2_1$R2_marginal) - as.numeric(r2_0$R2_marginal),
      delta_r2c = as.numeric(r2_1$R2_conditional) - as.numeric(r2_0$R2_conditional)
    )
    
    fixef_results[[i]] <- tidy(model_fit, effects = "fixed", conf.int = TRUE) %>%
      transmute(fold = i, term, estimate, std.error, statistic, p.value, 
                conf.low, conf.high)
  }
  cv_summary <- do.call(rbind, cv_results)
  fixef_summary <- do.call(rbind, fixef_results)
  
  return(
    list(cv_summary = cv_summary, fixef_summary = fixef_summary)
  )
}

calc_test_loglik_lmer <- function(model, test_data, response_var,
                                  allow_new_levels = TRUE) {
  
  mu_pred <- predict(model, newdata = test_data,
                     allow.new.levels = allow_new_levels)
  
  # sd of residuals
  resid_sd <- sigma(model)
  
  observed <- test_data[[response_var]]
  n <- nrow(test_data)
  
  # residual sum of squares
  residual_ss <- sum((test_data$Reaction - mu_pred)^2)
  
  log_lik <- (n / 2) * log(2 * pi) - (n / 2) * log((resid_sd)^2) - (residual_ss / (2 * (resid_sd)^2))
  
  return(
    # sum(dnorm(observed, mean = mu_pred, sd = resid_sd, log = TRUE))
    log_lik
    )
}

model_cross_val_lmer <- function(base_formula, model_formula, df, group_var,
                                 num_folds=10){
  
  # sample folds grouped by WorkerId
  groups <- unique(df[[group_var]])
  fold_map <- setNames(sample(rep(1:num_folds, length.out = length(groups))), groups)
  df$.fold <- fold_map[as.character(df[[group_var]])]
  
  cv_results <- vector("list", num_folds)
  fixef_results <- vector("list", num_folds)
  
  response_var <- all.vars(base_formula)[1]
  
  # create a dummy null model for r2 calculation
  null_formula <- make_null_formula(model_formula)
  
  for (i in 1:num_folds) {
    train_data <- df %>% filter(.fold != i)
    test_data  <- df %>% filter(.fold == i)
    
    null_fit <- lmer(null_formula, data = train_data)
    base_fit <- lmer(base_formula, data = train_data)
    model_fit <- lmer(model_formula, data = train_data)
    
    ll0 <- calc_test_loglik_lmer(base_fit, test_data, response_var)/nrow(test_data)
    ll1 <- calc_test_loglik_lmer(model_fit, test_data, response_var)/nrow(test_data)
    delta_ll <- (ll1 - ll0)
    
    rmse0 <- calc_test_rmse(base_fit, test_data, response_var)
    rmse1 <- calc_test_rmse(model_fit, test_data, response_var)
    delta_rmse <- rmse1 - rmse0   # negative = m1 has lower (better) error
    
    r2_0 <- r2(base_fit, null_model = null_fit)
    r2_1 <- r2(model_fit, null_model = null_fit)
    
    cv_results[[i]] <- data.frame(
      fold = i,
      ll0 = ll0, ll1 = ll1, delta_ll = delta_ll,
      rmse0 = rmse0, rmse1 = rmse1, delta_rmse = delta_rmse,
      r2m_m0 = as.numeric(r2_0$R2_marginal),  r2c_m0 = as.numeric(r2_0$R2_conditional),
      r2m_m1 = as.numeric(r2_1$R2_marginal),  r2c_m1 = as.numeric(r2_1$R2_conditional),
      delta_r2m = as.numeric(r2_1$R2_marginal) - as.numeric(r2_0$R2_marginal),
      delta_r2c = as.numeric(r2_1$R2_conditional) - as.numeric(r2_0$R2_conditional)
    )
    
    fixef_results[[i]] <- tidy(model_fit, effects = "fixed", conf.int = TRUE) %>%
      transmute(fold = i, term, estimate, std.error, statistic, 
                conf.low, conf.high)
  }
  cv_summary <- do.call(rbind, cv_results)
  fixef_summary <- do.call(rbind, fixef_results)
  
  return(
    list(cv_summary = cv_summary, fixef_summary = fixef_summary)
    )
}