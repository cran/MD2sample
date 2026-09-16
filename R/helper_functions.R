# Internal validators and helpers --------------------------------------------

validate_integer_scalar <- function(x, name, min=0L) {
  if(length(x) != 1L || is.na(x) || !is.finite(x) || x < min ||
     abs(x-round(x)) > sqrt(.Machine$double.eps)) {
    stop(name, " must be a single integer >= ", min, ".", call.=FALSE)
  }
  invisible(as.integer(round(x)))
}

validate_probability <- function(x, name="alpha") {
  if(length(x) != 1L || is.na(x) || !is.finite(x) || x <= 0 || x >= 1)
    stop(name, " must be a single number strictly between 0 and 1.", call.=FALSE)
  invisible(x)
}

validate_logical_scalar <- function(x, name) {
  if(!is.logical(x) || length(x) != 1L || is.na(x))
    stop(name, " must be TRUE or FALSE.", call.=FALSE)
  invisible(x)
}

validate_ts_output <- function(x) {
  if(!is.numeric(x) || length(x) < 1L)
    stop("TS must return a non-empty numeric vector.", call.=FALSE)
  nm <- names(x)
  if(is.null(nm) || anyNA(nm) || any(!nzchar(nm)))
    stop("TS must return a named numeric vector.", call.=FALSE)
  if(anyDuplicated(nm)) stop("TS must return a vector with unique names.", call.=FALSE)
  invisible(x)
}

normalize_samplingmethod <- function(samplingmethod) {
  if(is.character(samplingmethod)) {
    if(length(samplingmethod) != 1L || !samplingmethod %in% c("Binomial", "independence"))
      stop("samplingmethod must be 'Binomial', 'independence', 1, or 2.", call.=FALSE)
    return(if(samplingmethod == "independence") 1L else 2L)
  }
  validate_integer_scalar(samplingmethod, "samplingmethod", 1L)
  if(!samplingmethod %in% c(1, 2))
    stop("samplingmethod must be 'Binomial', 'independence', 1, or 2.", call.=FALSE)
  as.integer(samplingmethod)
}


# Determine the likely ordering of a four-column discrete-data matrix.
# If the standard names are present, use them. Otherwise order columns by
# their number of distinct values; the two support/bin columns are expected
# to have fewer distinct values than the count columns.
.discrete_matrix_order <- function(x) {
  standard_names <- c("vals_x", "vals_y", "x", "y")
  if(!is.null(colnames(x)) && all(standard_names %in% colnames(x)))
    return(match(standard_names, colnames(x)))
  f <- function(z) length(unique(z))
  order(apply(x, 2, f), seq_len(ncol(x)))
}

# Recognize a discrete two-sample data matrix.
# Named matrices use the standard column names. Otherwise column roles are
# inferred by ordering the columns by their numbers of distinct values.
is_discrete_data <- function(x) {
  if(is.data.frame(x)) x <- as.matrix(x)
  if(!is.matrix(x) || !is.numeric(x) || ncol(x) != 4L) return(FALSE)
  if(any(!is.finite(x))) return(FALSE)

  standard_names <- c("vals_x", "vals_y", "x", "y")
  named <- !is.null(colnames(x)) && all(standard_names %in% colnames(x))
  z <- x[, .discrete_matrix_order(x), drop=FALSE]

  if(!named) {
    f <- function(v) length(unique(v))
    if(f(z[, 1]) != f(z[, 2])) return(FALSE)
  }

  counts <- z[, 3:4, drop=FALSE]
  if(any(counts < 0) || any(counts != round(counts))) return(FALSE)
  TRUE
}

# Standardize a recognized discrete matrix to the internal column order.
prepare_discrete_data <- function(x, as_list=TRUE) {
  if(is.data.frame(x)) x <- as.matrix(x)
  if(!is_discrete_data(x))
    stop(paste0(
      "Discrete matrix data must be a numeric four-column matrix with two ",
      "support/bin columns and two finite non-negative integer count columns."
    ), call.=FALSE)

  z <- x[, .discrete_matrix_order(x), drop=FALSE]
  colnames(z) <- c("vals_x", "vals_y", "x", "y")
  if(!as_list) return(z)
  list(vals_x=z[, 1], vals_y=z[, 2], x=z[, 3], y=z[, 4])
}

prepare_twosample_input <- function(x, y_missing, y=NULL, vals_x=NA, vals_y=NA,
                                    DoTransform=TRUE,
                                    Ranges=matrix(c(-Inf, Inf, -Inf, Inf), 2, 2),
                                    SuppressMessages=FALSE) {
  if(y_missing) {
    # A data frame is also a list in R, so test matrix/data-frame discrete
    # input before testing for the list form used by continuous data.
    if(is.data.frame(x) || is.matrix(x)) {
      ddisc <- prepare_discrete_data(x)
      x1 <- ddisc$x; y1 <- ddisc$y
      vals_x <- ddisc$vals_x; vals_y <- ddisc$vals_y
      Continuous <- FALSE
    } else if(is.list(x)) {
      if(!all(c("x", "y") %in% names(x)))
        stop("A continuous-data list must contain components named x and y.", call.=FALSE)
      x1 <- x$x; y1 <- x$y
      Continuous <- TRUE
    } else {
      stop(paste0(
        "When y is omitted, x must be either a continuous-data list with ",
        "components x and y or a four-column discrete matrix/data frame."
      ), call.=FALSE)
    }
  } else {
    one_missing <- any(is.na(vals_x)) != any(is.na(vals_y))
    if(one_missing) stop("vals_x and vals_y must either both be supplied or both be NA.", call.=FALSE)
    Continuous <- any(is.na(c(vals_x, vals_y)))
    x1 <- x; y1 <- y
  }

  if(Continuous) {
    if(is.data.frame(x1)) x1 <- as.matrix(x1)
    if(is.data.frame(y1)) y1 <- as.matrix(y1)
    if(!is.matrix(x1) || !is.numeric(x1) || !is.matrix(y1) || !is.numeric(y1))
      stop("For continuous data, x and y must be numeric matrices.", call.=FALSE)
    if(nrow(x1) < 1L || nrow(y1) < 1L || ncol(x1) < 1L || ncol(x1) != ncol(y1))
      stop("x and y must be non-empty matrices with the same number of columns.", call.=FALSE)
    if(any(!is.finite(x1)) || any(!is.finite(y1)))
      stop("x and y must contain only finite values.", call.=FALSE)
    if(nrow(y1) < nrow(x1)) { tmp <- y1; y1 <- x1; x1 <- tmp }
    rawdta <- list(x=x1, y=y1)
    dta <- rawdta
    if(DoTransform) {
      dta <- transform01(dta)
      Ranges <- matrix(c(0, 1, 0, 1), 2, 2)
    }
    if(!SuppressMessages) message("Data is assumed to be continuous")
    list(Continuous=TRUE, dta=dta, rawdta=rawdta, x=dta$x, y=dta$y,
         Dim=ncol(x1), n.x=nrow(x1), n.y=nrow(y1), DoTransform=DoTransform,
         Ranges=Ranges)
  } else {
    x1 <- as.numeric(x1); y1 <- as.numeric(y1)
    vals_x <- as.numeric(vals_x); vals_y <- as.numeric(vals_y)
    if(length(x1) < 1L || length(y1) < 1L || length(x1) != length(vals_x) || length(y1) != length(vals_y))
      stop("For discrete data, counts and support-value vectors must have matching positive lengths.", call.=FALSE)
    if(any(!is.finite(c(x1, y1, vals_x, vals_y))) || any(x1 < 0) || any(y1 < 0) ||
       any(abs(x1-round(x1)) > sqrt(.Machine$double.eps)) ||
       any(abs(y1-round(y1)) > sqrt(.Machine$double.eps)))
      stop("Discrete x and y must be non-negative integer counts with finite support values.", call.=FALSE)
    if(!SuppressMessages) message("Data is assumed to be discrete")
    dta <- list(x=x1, y=y1, vals_x=vals_x, vals_y=vals_y)
    list(Continuous=FALSE, dta=dta, rawdta=dta, x=NULL, y=NULL, Dim=2L,
         n.x=sum(x1), n.y=sum(y1), DoTransform=FALSE, Ranges=Ranges)
  }
}

makeTSextra <- function(dta, Continuous, DoTransform, samplingmethod,
                        TSextra=NULL, rnull=NULL, rawdta=dta) {
  if(is.null(TSextra)) TSextra <- list()
  if(!is.list(TSextra)) stop("TSextra must be a list when supplied.", call.=FALSE)
  system <- if(Continuous) list(
    knn=function(x) FNN::get.knn(x, 5)$nn.index,
    dist=function(dta) find_dist(dta),
    distances=find_dist(dta),
    DoTransform=DoTransform,
    ParametricBootstrap=!is.null(rnull)
  ) else list(
    dist=function(dta) NULL,
    organize=function(dta) dta[order(dta[,1], dta[,2]), , drop=FALSE],
    samplingmethod=samplingmethod,
    ParametricBootstrap=!is.null(rnull)
  )
  out <- utils::modifyList(TSextra, system)
  if(!is.null(rnull)) {
    if(!is.function(rnull)) stop("rnull must be a function.", call.=FALSE)
    out$rnull <- rnull
    out$rawdta <- rawdta
  }
  out
}
#' maketypeTS
#' 
#' find typeTS and TS
#' 
#' @param TS user-supplied test statistic, if not missing
#' @param Continuous is data continuous?
#' @return a list
#' @keywords internal
maketypeTS <- function(TS, Continuous) {
  useSingleProcessor <- FALSE
  
  if(missing(TS)) {
    return(list(
      TS = if(Continuous) TS_cont else TS_disc,
      typeTS = if(Continuous) 1L else 4L,
      CustomTS = FALSE,
      useSingleProcessor = FALSE
    ))
  }
  
  if(!is.function(TS))
    stop("TS must be a function.", call. = FALSE)
  
  nf <- length(formals(TS))
  b <- tryCatch(
    paste(deparse(body(TS)), collapse = " "),
    error = function(e) ""
  )
  
  if(grepl("\\.Call", b, fixed = FALSE))
    useSingleProcessor <- TRUE
  
  if(Continuous) {
    
    if(!nf %in% c(2L, 3L))
      stop(
        "For continuous data, TS must have 2 or 3 arguments: x, y, and optional TSextra.",
        call. = FALSE
      )
    
    typeTS <- if(nf == 2L) 2L else 3L
    
  } else if(nf == 1L) {
    
    # Simple discrete interface: TS(dta)
    TS0 <- TS
    
    TS1 <- function(x, y, vals_x, vals_y) {
      dta <- cbind(
        vals_x = vals_x,
        vals_y = vals_y,
        x = x,
        y = y
      )
      TS0(dta)
    }
    
    TS <- TS1
    typeTS <- 5L
    
  } else if(nf == 2L) {
    
    # Simple discrete interface: TS(dta, TSextra)
    TS0 <- TS
    
    TS2 <- function(x, y, vals_x, vals_y, TSextra) {
      dta <- cbind(
        vals_x = vals_x,
        vals_y = vals_y,
        x = x,
        y = y
      )
      TS0(dta, TSextra)
    }
    
    TS <- TS2
    typeTS <- 6L
    
  } else {
    
    # Preserve the previous discrete custom-statistic interface:
    # TS(x, y, vals_x, vals_y)
    # TS(x, y, vals_x, vals_y, TSextra)
    if(!nf %in% c(4L, 5L))
      stop(
        paste0(
          "For discrete data, TS must have 1 or 2 arguments ",
          "(a four-column data matrix and optional TSextra), ",
          "or the legacy 4 or 5 arguments."
        ),
        call. = FALSE
      )
    
    typeTS <- if(nf == 4L) 5L else 6L
  }
  
  list(
    TS = TS,
    typeTS = typeTS,
    CustomTS = TRUE,
    useSingleProcessor = useSingleProcessor
  )
}

makemaxProcessor <- function(maxProcessor, dta, TS, typeTS, TSextra, B,
                             SuppressMessages=FALSE, useSingleProcessor=FALSE) {
  if(useSingleProcessor) return(1L)
  if(!missing(maxProcessor)) {
    validate_integer_scalar(maxProcessor, "maxProcessor", 1L)
    return(as.integer(maxProcessor))
  }
  m <- parallel::detectCores(logical=FALSE)
  if(is.na(m) || m < 1L) m <- 1L
  ans <- max(1L, as.integer(m)-1L)
  if(ans > 1L) {
    tm <- timecheck(dta, TS, typeTS, TSextra)
    if(length(tm) > 1L) tm <- max(tm, na.rm=TRUE)
    if(!is.finite(tm) || 2*tm*B < 20 || B < 2*ans) {
      ans <- 1L
      if(!SuppressMessages) message("maxProcessor set to 1 because parallel overhead would dominate this short computation")
    } else if(!SuppressMessages) message("Using ", ans, " cores.")
  }
  ans
}
