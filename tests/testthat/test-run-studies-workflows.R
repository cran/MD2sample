test_that("run.studies handles included and user-supplied tests", {
  set.seed(301)

  cases <- case.studies(ReturnCaseNames = TRUE)
  expect_true(is.character(cases))
  expect_true("NormalD2" %in% cases)

  included <- run.studies(
    study = "NormalD2",
    Continuous = TRUE,
    param_alt = 0.5,
    nsample = c(35, 40),
    B = 5,
    maxProcessor = 1,
    SuppressMessages = TRUE
  )
  expect_valid_power(included)

  extra_stat <- list(which = "statistic", nbins = c(3, 3))
  supplied_statistic <- run.studies(
    study = "NormalD2",
    Continuous = TRUE,
    TS = chiTS.cont,
    TSextra = extra_stat,
    param_alt = 0.5,
    nsample = c(35, 40),
    B = 5,
    maxProcessor = 1,
    SuppressMessages = TRUE
  )
  expect_valid_power(supplied_statistic)

  extra_pvalue <- list(which = "pvalue", nbins = c(3, 3))
  supplied_pvalue <- run.studies(
    study = "NormalD2",
    Continuous = TRUE,
    TS = chiTS.cont,
    TSextra = extra_pvalue,
    With.p.value = TRUE,
    param_alt = 0.5,
    nsample = c(35, 40),
    B = 5,
    maxProcessor = 1,
    SuppressMessages = TRUE
  )
  expect_valid_power(supplied_pvalue)
})
