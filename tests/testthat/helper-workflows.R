expect_valid_test_result <- function(x) {
  expect_s3_class(x, "MD2sample_test")
  expect_true(is.data.frame(x$results))
  expect_true(is.numeric(x$statistics))
  expect_true(is.numeric(x$p.values))
  expect_gt(length(x$statistics), 0L)
  expect_equal(length(x$statistics), length(x$p.values))
  expect_true(all(is.finite(x$statistics)))
  expect_true(all(is.finite(x$p.values) | is.na(x$p.values)))
  observed <- x$p.values[!is.na(x$p.values)]
  expect_true(all(observed >= 0 & observed <= 1))
}

expect_valid_power <- function(x) {
  if(inherits(x, "MD2sample_power")) x <- x$power
  expect_true(is.matrix(x) || is.data.frame(x) || is.numeric(x))
  expect_gt(length(x), 0L)
  observed <- x[!is.na(x)]
  expect_gt(length(observed), 0L)
  expect_true(all(is.finite(observed)))
  expect_true(all(observed >= 0 & observed <= 1))
}
