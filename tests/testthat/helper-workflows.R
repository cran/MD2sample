expect_valid_test_result <- function(x) {
  expect_type(x, "list")
  expect_named(x, c("statistics", "p.values"))
  expect_true(is.numeric(x$statistics))
  expect_true(is.numeric(x$p.values))
  expect_gt(length(x$statistics), 0L)
  expect_equal(length(x$statistics), length(x$p.values))
  expect_true(all(is.finite(x$statistics)))
  expect_true(all(is.finite(x$p.values)))
  expect_true(all(x$p.values >= 0 & x$p.values <= 1))
}

expect_valid_power <- function(x) {
  expect_true(is.matrix(x) || is.data.frame(x) || is.numeric(x))
  expect_gt(length(x), 0L)
  expect_true(all(is.finite(x)))
  expect_true(all(x >= 0 & x <= 1))
}
