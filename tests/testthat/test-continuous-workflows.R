test_that("continuous two-sample workflows run", {
  set.seed(101)
  f <- case.studies(1, nx = 40, ny = 45)$f
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
  expect_named(null_result$p.values, c("KS", "CvM"))

  alt_result <- twosample_test(
    dta_alt,
    B = 9,
    doMethods = c("KS", "CvM"),
    maxProcessor = 1,
    SuppressMessages = TRUE
  )
  expect_valid_test_result(alt_result)

  extra_stat <- list(which = "statistics", nbins = rbind(c(3, 3), c(4, 4)))
  custom_result <- twosample_test(
    dta_null,
    TS = chiTS.cont,
    TSextra = extra_stat,
    B = 9,
    maxProcessor = 1,
    SuppressMessages = TRUE
  )
  expect_valid_test_result(custom_result)
  expect_length(custom_result$statistics, 2L)

  adjusted_output <- capture.output(
    adjusted_result <- suppressMessages(
      twosample_test_adjusted_pvalue(
        dta_null,
        B = c(9, 9),
        doMethods = c("CvM", "AZ"),
        maxProcessor = 1,
        SuppressMessages = TRUE
      )
    ),
    type = "message"
  )
  expect_null(adjusted_result)
})

test_that("continuous power workflows return probabilities", {
  set.seed(102)
  f <- case.studies(1, nx = 35, ny = 40)$f

  included <- twosample_power(
    f,
    c(0, 0.5),
    B = 7,
    doMethods = c("KS", "CvM"),
    maxProcessor = 1,
    SuppressMessages = TRUE
  )
  expect_valid_power(included)
  expect_equal(nrow(included), 2L)

  extra_stat <- list(which = "statistic", nbins = c(3, 3))
  supplied_statistic <- twosample_power(
    f,
    c(0, 0.5),
    TS = chiTS.cont,
    TSextra = extra_stat,
    B = 7,
    maxProcessor = 1,
    SuppressMessages = TRUE
  )
  expect_valid_power(supplied_statistic)

  extra_pvalue <- list(which = "pvalue", nbins = c(3, 3))
  supplied_pvalue <- twosample_power(
    f,
    c(0, 0.5),
    TS = chiTS.cont,
    TSextra = extra_pvalue,
    With.p.value = TRUE,
    B = 7,
    maxProcessor = 1,
    SuppressMessages = TRUE
  )
  expect_valid_power(supplied_pvalue)
})
