library(reformulas, quietly = TRUE, warn.conflicts = FALSE)
library(broom.mixed, quietly = TRUE, warn.conflicts = FALSE)

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