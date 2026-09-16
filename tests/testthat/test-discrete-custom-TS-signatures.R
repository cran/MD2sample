test_that("discrete custom TS accepts a matrix alone or matrix plus TSextra", {
  d <- cbind(
    vals_x=c(0, 1, 0, 1),
    vals_y=c(0, 0, 1, 1),
    x=c(10, 12, 8, 11),
    y=c(9, 13, 7, 12)
  )

  TS1 <- function(z) c(Custom=sum(abs(z[, "x"] - z[, "y"])))
  out1 <- twosample_test(d, TS=TS1, B=0, SuppressMessages=TRUE)
  expect_s3_class(out1, "MD2sample_test")
  expect_equal(unname(out1$statistics["Custom"]), sum(abs(d[, "x"] - d[, "y"])))

  TS2 <- function(z, TSextra)
    c(Custom=sum(abs(z[, "x"] - z[, "y"])) * TSextra$scale)
  out2 <- twosample_test(d, TS=TS2, TSextra=list(scale=2), B=0,
                         SuppressMessages=TRUE)
  expect_equal(unname(out2$statistics["Custom"]),
               2 * sum(abs(d[, "x"] - d[, "y"])))
})

test_that("discrete custom TS simple signatures work in power", {
  f <- function(delta=0) cbind(
    vals_x=c(0, 1, 0, 1),
    vals_y=c(0, 0, 1, 1),
    x=c(10, 12, 8, 11),
    y=c(9 + delta, 13, 7, 12)
  )

  TS1 <- function(z) c(Custom=sum(abs(z[, "x"] - z[, "y"])))
  p1 <- twosample_power(f, 0, TS=TS1, B=2, maxProcessor=1,
                        SuppressMessages=TRUE)
  expect_true(is.numeric(p1) || is.matrix(p1))

  TS2 <- function(z, TSextra)
    c(Custom=sum(abs(z[, "x"] - z[, "y"])) * TSextra$scale)
  p2 <- twosample_power(f, 0, TS=TS2, TSextra=list(scale=2), B=2,
                        maxProcessor=1, SuppressMessages=TRUE)
  expect_true(is.numeric(p2) || is.matrix(p2))
})
