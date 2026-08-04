test_that("discrete two-sample workflows run", {
  set.seed(201)
  f <- case.studies(1, nx = 80, ny = 90, nbins = c(5, 5))$f
  dta_null <- f(0)
  dta_alt <- f(0.5)

  null_result <- twosample_test(
    dta_null,
    B = 9,
    doMethods = c("KS", "CvM"),
    maxProcessor = 1,
    SuppressMessages = TRUE
  )
  expect_valid_test_result(null_result)

  alt_result <- twosample_test(
    dta_alt,
    B = 9,
    doMethods = c("KS", "CvM"),
    maxProcessor = 1,
    SuppressMessages = TRUE
  )
  expect_valid_test_result(alt_result)

  extra_stat <- list(which = "statistic")
  custom_result <- twosample_test(
    dta_null,
    TS = chiTS.disc,
    TSextra = extra_stat,
    B = 9,
    maxProcessor = 1,
    SuppressMessages = TRUE
  )
  expect_valid_test_result(custom_result)

  adjusted_output <- capture.output(
    adjusted_result <- suppressMessages(
      twosample_test_adjusted_pvalue(
        dta_null,
        B = c(9, 9),
        doMethods = c("KS", "CvM"),
        maxProcessor = 1,
        SuppressMessages = TRUE
      )
    ),
    type = "message"
  )
  expect_null(adjusted_result)
})

test_that("discrete power workflows return probabilities", {
  set.seed(202)
  f <- case.studies(1, nx = 70, ny = 80, nbins = c(5, 5))$f

  included <- twosample_power(
    f,
    c(0, 0.5),
    B = 7,
    doMethods = c("KS", "CvM"),
    maxProcessor = 1,
    SuppressMessages = TRUE
  )
  expect_valid_power(included)

  extra_stat <- list(which = "statistic")
  supplied_statistic <- twosample_power(
    f,
    c(0, 0.5),
    TS = chiTS.disc,
    TSextra = extra_stat,
    B = 7,
    maxProcessor = 1,
    SuppressMessages = TRUE
  )
  expect_valid_power(supplied_statistic)

  extra_pvalue <- list(which = "pvalue")
  supplied_pvalue <- twosample_power(
    f,
    c(0, 0.5),
    TS = chiTS.disc,
    TSextra = extra_pvalue,
    With.p.value = TRUE,
    B = 7,
    maxProcessor = 1,
    SuppressMessages = TRUE
  )
  expect_valid_power(supplied_pvalue)
})
