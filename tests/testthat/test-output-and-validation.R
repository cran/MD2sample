test_that("twosample_test has structured backward-compatible output", {
  set.seed(401)
  dta <- case.studies(1, nx=30, ny=35)$f(0)
  z <- twosample_test(dta, B=5, maxProcessor=1, doMethods=c("KS","CvM"), SuppressMessages=TRUE)
  expect_s3_class(z, "MD2sample_test")
  expect_named(z$statistics, c("KS","CvM"))
  expect_named(z$p.values, c("KS","CvM"))
  expect_equal(as.data.frame(z), z$results)
  expect_s3_class(summary(z), "summary.MD2sample_test")
})

test_that("B=0 retains structured output", {
  set.seed(402)
  dta <- case.studies(1, nx=25, ny=30)$f(0)
  z <- twosample_test(dta, B=0, maxProcessor=1, SuppressMessages=TRUE)
  expect_s3_class(z, "MD2sample_test")
  expect_true(all(is.na(z$p.values)))
})

test_that("input validation produces informative errors", {
  set.seed(403)
  dta <- case.studies(1, nx=25, ny=30)$f(0)
  expect_error(twosample_test(dta, B=-1, SuppressMessages=TRUE), "B")
  expect_error(twosample_test(dta, B=3, doMethods="not-a-method", maxProcessor=1, SuppressMessages=TRUE), "Unknown method")
  badTS <- function(x, y) c(1,2)
  expect_error(twosample_test(dta, TS=badTS, B=3, maxProcessor=1, SuppressMessages=TRUE), "named numeric vector")
})

test_that("twosample_power can return Wilson intervals", {
  set.seed(404)
  f <- case.studies(1, nx=25, ny=30)$f
  z <- twosample_power(f, c(0,0.4), B=8, doMethods=c("KS","CvM"),
                       maxProcessor=1, SuppressMessages=TRUE, CI=TRUE, seed=404)
  expect_s3_class(z, "MD2sample_power")
  expect_equal(dim(z$power), c(2L,2L))
  expect_true(all(z$lower <= z$power, na.rm=TRUE))
  expect_true(all(z$power <= z$upper, na.rm=TRUE))
  expect_equal(nrow(as.data.frame(z)), 4L)
})
