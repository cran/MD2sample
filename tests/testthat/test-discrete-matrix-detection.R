test_that("standard discrete matrix names are recognized in any order", {
  d <- cbind(
    vals_x=c(1, 1, 2, 2),
    vals_y=c(1, 2, 1, 2),
    x=c(10, 12, 9, 11),
    y=c(8, 14, 10, 10)
  )

  expect_true(MD2sample:::is_discrete_data(d))
  expect_true(MD2sample:::is_discrete_data(d[, c("y", "vals_y", "x", "vals_x")]))

  z <- MD2sample:::prepare_discrete_data(d[, c("y", "vals_y", "x", "vals_x")])
  expect_equal(z$vals_x, d[, "vals_x"])
  expect_equal(z$vals_y, d[, "vals_y"])
  expect_equal(z$x, d[, "x"])
  expect_equal(z$y, d[, "y"])
})

test_that("unnamed and arbitrarily named discrete matrices are inferred", {
  d <- cbind(
    c(10, 12, 9, 11),
    c(1, 1, 2, 2),
    c(8, 14, 10, 10),
    c(1, 2, 1, 2)
  )

  expect_true(MD2sample:::is_discrete_data(d))

  colnames(d) <- c("a", "b", "c", "d")
  expect_true(MD2sample:::is_discrete_data(d))

  z <- MD2sample:::prepare_discrete_data(d)
  expect_equal(length(unique(z$vals_x)), length(unique(z$vals_y)))
  expect_true(all(z$x >= 0 & z$x == round(z$x)))
  expect_true(all(z$y >= 0 & z$y == round(z$y)))
})

test_that("invalid four-column matrices are not classified as discrete", {
  bad_counts <- cbind(
    c(1, 1, 2, 2),
    c(1, 2, 1, 2),
    c(10.5, 12, 9, 11),
    c(8, 14, 10, 10)
  )
  expect_false(MD2sample:::is_discrete_data(bad_counts))

  negative_counts <- cbind(
    c(1, 1, 2, 2),
    c(1, 2, 1, 2),
    c(10, 12, 9, -1),
    c(8, 14, 10, 10)
  )
  expect_false(MD2sample:::is_discrete_data(negative_counts))

  wrong_support <- cbind(
    c(1, 1, 2, 2),
    c(1, 2, 3, 4),
    c(10, 12, 9, 11),
    c(8, 14, 10, 10)
  )
  expect_false(MD2sample:::is_discrete_data(wrong_support))
})

test_that("discrete matrix detection preserves structured test output", {
  d <- cbind(
    c(1, 1, 2, 2),
    c(1, 2, 1, 2),
    c(10, 12, 9, 11),
    c(8, 14, 10, 10)
  )
  colnames(d) <- NULL
  z <- twosample_test(d, B=0, SuppressMessages=TRUE)
  expect_s3_class(z, "MD2sample_test")
  expect_true(is.data.frame(z$results))
  expect_true(all(is.na(z$p.values)))
})
