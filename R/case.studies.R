#' Create case studies
#' 
#' This function creates the functions needed to run the various case studies.
#'
#' @param which name of the case study.
#' @param n =200, sample size for both data sets
#' @param nx =n, sample size for first data sets
#' @param ny =n, sample size for second data sets
#' @param nbins =-1, number of bins for chi-square test, 2D only
#' @param Ranges = matrix(c(-Inf, Inf, -Inf, Inf), 2, 2), ranges of variables
#' @param ReturnCaseNames =FALSE, should list of case studies be returned?
#' @return a list of functions and vectors
#' @export 
case.studies = function (which, n = 200, nx = n, ny = n, nbins = -1, Ranges = matrix(c(-Inf, 
    Inf, -Inf, Inf), 2, 2), ReturnCaseNames = FALSE) 
{
    cases = c("NormalD2", "tD2", "UniformMixtureD2", "FrankD2", 
        "ClaytonD2", "GumbelD2", "GalambosD2", "HuslerReissD2", 
        "ClaytonGumbelD2", "UniformFrankD2", "ParetoSimplexD2", 
        "KhoudrajiClaytonD2", "NormalUniformD2", "JoeD2", "DalitzCleoD2", 
        "DalitzPDGD2", "DalitzBabarD2", "NormalShiftM", "NormalStretchM", 
        "UniformRotateM", "UniformBetaM", "TruncExponentialM", 
        "DalitzCleoM", "DalitzPDGM", "DalitzBabarM", "FrankExponentialM", 
        "FrankLinearM", "FrankNormalM", "ClaytonExponentialM", 
        "ClaytonLinearM", "ClaytonNormalM", "GalambosExponentialM", 
        "GalambosLinearM", "GalambosNormalM", 
        "NormalD5", "tD5", "FrankD5", "ClaytonD5", "GumbelD5", "JoeD5",
        "UniformFrankD5", "FrankClaytonD5", "FrankJoeD5", "UniformExponentialM5",
        "FrankExponentialM5", "FrankLinearM5", "FrankNormalM5",
        "ClaytonExponentialM5", "ClaytonLinearM5", "ClaytonNormalM5")
    if (ReturnCaseNames) 
        return(cases)
    if (is.numeric(which)) 
        which = cases[which]
    uniforms = function(d=2) {
        x = matrix(stats::runif(d * nx), nx, d)
        y = matrix(stats::runif(d * ny), nx, d)
        out = list(x = x, y = y)
        if (nbins[1] > 0) 
            out = MD2sample::bincounterR(out, nbins = nbins, 
                Ranges = Ranges)
        out
    }
    if (which == "NormalD2") {
        return(list(f = function(a = 0) {
            S = diag(2)
            x = mvtnorm::rmvnorm(nx, sigma = S)
            S[1, 2] = a
            S[2, 1] = a
            y = mvtnorm::rmvnorm(ny, sigma = S)
            out = list(x = x, y = y)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(0, 0.6)))
    }
    if (which == "tD2") {
        return(list(f = function(a = 0, df = 5, d = 2) {
            S = diag(d)
            x = mvtnorm::rmvt(2 * nx, sigma = S, df = df)
            x = x[abs(x[, 1]) < 5, ]
            x = x[abs(x[, 2]) < 5, ]
            x = x[1:nx, ]
            S[1, d] = a
            S[d, 1] = a
            y = mvtnorm::rmvt(2 * ny, sigma = S, df = df)
            y = y[abs(y[, 1]) < 5, ]
            y = y[abs(y[, 2]) < 5, ]
            y = y[1:ny, ]
            out = list(x = x, y = y)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(0, 0.6)))
    }
    if (which == "UniformMixtureD2") {
        return(list(f = function(alpha = 0, d = 2) {
            if (alpha == 0) return(uniforms())
            x = matrix(stats::runif(d * nx), nx, d)
            m = round(alpha * ny)
            z1 = stats::runif(m)
            z2 = stats::rnorm(m, z1, 0.02)
            z = cbind(z2, 2 * z1 - z2)
            z[z[, 2] < 0, 2] = 0
            z[z[, 2] > 1, 2] = 1
            y = rbind(matrix(stats::runif(d * (ny - m)), ny - 
                m, d), z)
            out = list(x = x, y = y)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(0, 0.35)))
    }
    if (which == "FrankD2") {
        return(list(f = function(theta = 0, d = 2) {
            if (theta == 0) return(uniforms())
            x = matrix(stats::runif(nx * d), ncol = d)
            cop = copula::frankCopula(theta, d)
            out = list(x = x, y = copula::rCopula(ny, cop))
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(0, 3.15)))
    }
    if (which == "ClaytonD2") {
        return(list(f = function(theta = 0, d = 2) {
            if (theta == 0) return(uniforms())
            x = matrix(stats::runif(nx * d), ncol = d)
            cop = copula::claytonCopula(theta, d)
            out = list(x = x, y = copula::rCopula(ny, cop))
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(0, 3)))
    }
    if (which == "GumbelD2") {
        return(list(f = function(theta = 0, d = 2) {
            if (theta == 0) return(uniforms())
            x = matrix(stats::runif(nx * d), ncol = d)
            cop = copula::gumbelCopula(theta, d)
            out = list(x = x, y = copula::rCopula(ny, cop))
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(1, 1.5)))
    }
    if (which == "GalambosD2") {
        return(list(f = function(theta = 0, d = 2) {
            if (theta == 0) return(uniforms())
            x = matrix(stats::runif(nx * d), ncol = d)
            cop = copula::galambosCopula(theta)
            out = list(x = x, y = copula::rCopula(ny, cop))
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(0, 2.5)))
    }
    if (which == "HuslerReissD2") {
        return(list(f = function(theta = 0) {
            d = 2
            if (theta == 0) return(uniforms())
            x = matrix(stats::runif(nx * d), ncol = d)
            cop = copula::huslerReissCopula(theta)
            out = list(x = x, y = copula::rCopula(ny, cop))
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(0, 2)))
    }
    if (which == "ClaytonGumbelD2") {
        return(list(f = function(a = 1/2) {
            cc <- copula::claytonCopula(copula::iTau(copula::claytonCopula(), 
                tau = 0.95))
            gc <- copula::gumbelCopula(copula::iTau(copula::gumbelCopula(), 
                tau = 0.15))
            wts1 <- c(1/2, 1/2)
            mcg1 <- copula::mixCopula(list(cc, gc), w = wts1)
            x <- copula::rCopula(nx, copula = mcg1)
            wts2 <- c(a, 1 - a)
            mcg2 <- copula::mixCopula(list(cc, gc), w = wts2)
            y <- copula::rCopula(nx, copula = mcg2)
            out = list(x = x, y = y)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(0.5, 0.1)))
    }
    if (which == "UniformFrankD2") {
        return(list(f = function(a = 1/2) {
            un <- copula::indepCopula()
            fc <- copula::frankCopula(20, 2)
            wts <- c(1/2, 1/2)
            mcg <- copula::mixCopula(list(un, fc), w = wts)
            x <- copula::rCopula(nx, copula = mcg)
            wts <- c(a, 1 - a)
            mcg <- copula::mixCopula(list(un, fc), w = wts)
            y <- copula::rCopula(nx, copula = mcg)
            out = list(x = x, y = y)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(0.5, 0.1)))
    }
    if (which == "ParetoSimplexD2") {
        return(list(f = function(th = 1, d = 2) {
            incBeta <- function(x, a, b) stats::pbeta(x, a, b) * 
                beta(a, b)
            psi = function(t, th) t^(-1/th) * incBeta(pmin(1, 
                t), a = 1/th, b = d)/th
            R = stats::runif(nx)^(-1)
            E = matrix(stats::rexp(nx * d), nrow = nx, ncol = d)
            S = E/matrix(rowSums(E), nrow = nx, ncol = d)
            x <- psi(R * S, 1)
            R = stats::runif(ny)^(-1/th)
            E = matrix(stats::rexp(ny * d), nrow = ny, ncol = d)
            S = E/matrix(rowSums(E), nrow = ny, ncol = d)
            y = psi(R * S, th)
            out = list(x = x, y = y)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(1, 1.3)))
    }
    if (which == "KhoudrajiClaytonD2") {
        return(list(f = function(a) {
            kcx <- copula::khoudrajiCopula(copula2 = copula::claytonCopula(6), 
                shapes = c(0.2, 0.95))
            kcy <- copula::khoudrajiCopula(copula2 = copula::claytonCopula(6), 
                shapes = c(a, 0.95))
            out = list(x = copula::rCopula(nx, copula = kcx), 
                y = copula::rCopula(ny, copula = kcy))
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(0.2, 0.7)))
    }
    if (which == "NormalUniformD2") {
        return(list(f = function(a = 1/2) {
            nc1 <- copula::indepCopula()
            nc2 <- copula::normalCopula(copula::iTau(copula::normalCopula(), 
                tau = 0.5))
            wts <- c(1, 0)
            mcg <- copula::mixCopula(list(nc1, nc2), w = wts)
            x <- copula::rCopula(nx, copula = mcg)
            wts <- c(1 - a, a)
            mcg <- copula::mixCopula(list(nc1, nc2), w = wts)
            y <- copula::rCopula(nx, copula = mcg)
            out = list(x = x, y = y)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(0, 0.7)))
    }
    if (which == "JoeD2") {
        return(list(f = function(a = 1) {
            nc1 <- copula::indepCopula()
            nc2 = nc1
            if (a > 1) nc2 = copula::joeCopula(a)
            out = list(x = copula::rCopula(nx, copula = nc1), 
                y = copula::rCopula(ny, copula = nc2))
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(1, 2)))
    }
    if (startsWith(which, "Dalitz")) {
        par2 = 130
        addon = "Diagonal Stripe"
        if (endsWith(which, "M")) {
            par2 = 35
            addon = c("Left Stripe", "Bottom Stripe", "Right Stripe")
        }
        experiment = 1
        if (startsWith(which, "DalitzPDG")) 
            experiment = 2
        if (startsWith(which, "DalitzBabar")) 
            experiment = 3
        return(list(f = function(a = 0) {
            tmp = MD2sample::rDalitz(a, experiment, addon, , 
                nx = 2 * nx, ny = 2 * ny)
            xl = range(tmp[[3]][, 1])
            yl = range(tmp[[3]][, 2:3])
            Ranges = matrix(c(xl, yl), 2, 2)
            x = tmp[[1]]
            x = x[!is.na(x[, 1]), ]
            x = x[!is.na(x[, 2]), ]
            y = tmp[[2]]
            y = y[sample(1:nrow(y)), ]
            y = y[!is.na(y[, 1]), ]
            y = y[!is.na(y[, 2]), ]
            out = list(x = x[1:nx, ], y = y[1:ny, ])
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(0, par2)))
    }

    if (which == "NormalShiftM") {
        return(list(f = function(mu = 0) {
            x = mvtnorm::rmvnorm(nx, c(0, 0))
            y = mvtnorm::rmvnorm(ny, mean = c(0, mu))
            out = list(x = x, y = y)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(0, 0.5)))
    }
    if (which == "NormalStretchM") {
        return(list(f = function(s = 1) {
            x = mvtnorm::rmvnorm(nx, sigma = diag(c(1, 1)))
            y = mvtnorm::rmvnorm(ny, sigma = diag(c(1, s)))
            out = list(x = x, y = y)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(1, 2)))
    }
    if (which == "UniformRotateM") {
        return(list(f = function(alpha = 0) {
            x = matrix(stats::runif(2 * nx), ncol = 2)
            y = matrix(stats::runif(2 * ny), ncol = 2) %*% matrix(c(cos(alpha), 
                sin(alpha), -sin(alpha), cos(alpha)), 2, 2)
            out = list(x = x, y = y)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(0, 0.175)))
    }
    if (which == "UniformBetaM") {
        return(list(f = function(a = 1) {
            x = matrix(stats::runif(2 * nx), ncol = 2)
            y = matrix(stats::rbeta(2 * ny, a, a), ncol = 2)
            out = list(x = x, y = y)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(1, 1.7)))
    }
    if (which == "TruncExponentialM") {
        return(list(f = function(lambda = 1) {
            tmp = NULL
            repeat {
                tmp = c(tmp, stats::rexp(4 * nx, 1))
                tmp = tmp[tmp < 2]
                if (length(tmp) > 2 * nx) break
            }
            x = matrix(tmp[1:(2 * nx)], ncol = 2)
            tmp = NULL
            repeat {
                tmp = c(tmp, stats::rexp(4 * nx, lambda))
                tmp = tmp[tmp < 2]
                if (length(tmp) > 2 * ny) break
            }
            y = matrix(tmp[1:(2 * ny)], ncol = 2)
            out = list(x = x, y = y)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(1, 1.5)))
    }
    if (which == "FrankExponentialM") {
        return(list(f = function(lambda = 1) {
            cop = copula::frankCopula(3, 2)
            x = copula::rCopula(nx, cop)
            y = copula::rCopula(ny, cop)
            out = list(x = x, y = y)
            out = MD2sample::change.marginals(out, "Exponential", 
                lambda)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(1, 1.7)))
    }
    if (which == "FrankLinearM") {
        return(list(f = function(s = 0) {
            cop = copula::frankCopula(3, 2)
            x = copula::rCopula(nx, cop)
            y = copula::rCopula(ny, cop)
            out = list(x = x, y = y)
            out = MD2sample::change.marginals(out, "Linear", 
                s)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(0, 0.5)))
    }
    if (which == "FrankNormalM") {
        return(list(f = function(s = 1) {
            cop = copula::frankCopula(3, 2)
            x = copula::rCopula(nx, cop)
            y = copula::rCopula(ny, cop)
            out = list(x = x, y = y)
            out = MD2sample::change.marginals(out, "Normal", 
                s)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(1, 1.5)))
    }
    if (which == "ClaytonExponentialM") {
        return(list(f = function(lambda = 1) {
            cop = copula::claytonCopula(3, 2)
            x = copula::rCopula(nx, cop)
            y = copula::rCopula(ny, cop)
            out = list(x = x, y = y)
            out = MD2sample::change.marginals(out, "Exponential", 
                lambda)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(1, 1.7)))
    }
    if (which == "ClaytonLinearM") {
        return(list(f = function(s = 0) {
            cop = copula::claytonCopula(3, 2)
            x = copula::rCopula(nx, cop)
            y = copula::rCopula(ny, cop)
            out = list(x = x, y = y)
            out = MD2sample::change.marginals(out, "Linear", 
                s)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(0, 0.5)))
    }
    if (which == "ClaytonNormalM") {
        return(list(f = function(s = 1) {
            cop = copula::claytonCopula(3, 2)
            x = copula::rCopula(nx, cop)
            y = copula::rCopula(ny, cop)
            out = list(x = x, y = y)
            out = MD2sample::change.marginals(out, "Normal", 
                s)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(1, 1.5)))
    }
    if (which == "GalambosExponentialM") {
        return(list(f = function(lambda = 1) {
            cop = copula::galambosCopula(2.5)
            x = copula::rCopula(nx, cop)
            y = copula::rCopula(ny, cop)
            out = list(x = x, y = y)
            out = MD2sample::change.marginals(out, "Exponential", 
                lambda)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(1, 1.4)))
    }
    if (which == "GalambosLinearM") {
        return(list(f = function(s = 0) {
            cop = copula::galambosCopula(2.5)
            x = copula::rCopula(nx, cop)
            y = copula::rCopula(ny, cop)
            out = list(x = x, y = y)
            out = MD2sample::change.marginals(out, "Linear", 
                s)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(0, 0.65)))
    }
    if (which == "GalambosNormalM") {
        return(list(f = function(s = 1) {
            cop = copula::galambosCopula(2.5)
            x = copula::rCopula(nx, cop)
            y = copula::rCopula(ny, cop)
            out = list(x = x, y = y)
            out = MD2sample::change.marginals(out, "Normal", 
                s)
            if (nbins[1] > 0) out = MD2sample::bincounterR(out, 
                nbins = nbins, Ranges = Ranges)
            out
        }, nbins = c(5, 5), param_alt = c(1, 1.42)))
    }
    if (which == "NormalD5") {
        return(list(f = function(a = 0) {
            S = diag(5)
            x = mvtnorm::rmvnorm(nx, sigma = S)
            S = matrix(a, 5, 5)
            for (i in 1:5) S[i, i] = 1
            y = mvtnorm::rmvnorm(ny, sigma = S)
            list(x = x, y = y)
        }, param_alt = c(0, 0.3)))
    }
    if (which == "tD5") {
        return(list(f = function(a = 0, df = 5) {
            S = diag(5)
            x = mvtnorm::rmvt(nx, sigma = S, df = df)
            S = matrix(a, 5, 5)
            for (i in 1:5) S[i, i] = 1
            y = mvtnorm::rmvt(ny, sigma = S, df = df)
            list(x = x, y = y)
        }, param_alt = c(0, 0.3)))
    }
    if (which == "FrankD5") {
        return(list(f = function(theta = 0, d = 5) {
            x = matrix(stats::runif(nx * d), ncol = d)
            if (theta == 0) return(list(x = x, y = matrix(stats::runif(ny * 
                d), ncol = d)))
            cop = copula::frankCopula(theta, d)
            list(x = x, y = copula::rCopula(ny, cop))
        }, param_alt = c(0, 1.7)))
    }
    if (which == "ClaytonD5") {
        return(list(f = function(theta = 0, d = 5) {
            x = matrix(stats::runif(nx * d), ncol = d)
            if (theta == 0) return(list(x = x, y = matrix(stats::runif(ny * 
                d), ncol = d)))
            cop = copula::claytonCopula(theta, d)
            list(x = x, y = copula::rCopula(ny, cop))
        }, param_alt = c(0, 0.5)))
    }
    if (which == "GumbelD5") {
        return(list(f = function(theta = 1, d = 5) {
            x = matrix(stats::runif(nx * d), ncol = d)
            if (theta == 1) return(list(x = x, y = matrix(stats::runif(ny * 
                d), ncol = d)))
            cop = copula::gumbelCopula(theta, d)
            list(x = x, y = copula::rCopula(ny, cop))
        }, param_alt = c(1, 1.25)))
    }
    if (which == "JoeD5") {
        return(list(f = function(a = 1) {
            nc1 <- copula::indepCopula(dim = 5L)
            nc2 = nc1
            if (a > 1) nc2 = copula::joeCopula(a, dim = 5L)
            list(x = copula::rCopula(nx, copula = nc1), y = copula::rCopula(ny, 
                copula = nc2))
        }, param_alt = c(1, 2)))
    }    
    if (which == "UniformFrankD5") {
        return(list(f=function(a=0) { 
          nc1 <- copula::indepCopula(dim=5) # the first component
          nc2 <- copula::frankCopula(5,dim=5) # the second component
          wts <- c(1, 0) # the corresponding weights
          mcg <- copula::mixCopula(list(nc1, nc2), w = wts)
          x <- copula::rCopula(nx, copula = mcg)
          wts <- c(1-a, a) # the corresponding weights
          mcg <- copula::mixCopula(list(nc1, nc2), w = wts)
          y <- copula::rCopula(ny, copula = mcg)
          out=list(x=x, y=y)
          out
      }, param_alt = c(0, 0.3)))
    }
    if (which == "FrankClaytonD5") {
        return(list(f=function(a=0) { 
          nc1 <- copula::frankCopula(5, dim=5L) # the first component
          nc2 <- copula::claytonCopula(5,dim=5L) # the second component
          wts <- c(1, 0) # the corresponding weights
          mcg <- copula::mixCopula(list(nc1, nc2), w = wts)
          x <- copula::rCopula(nx, copula = mcg)
          wts <- c(1-a, a) # the corresponding weights
          mcg <- copula::mixCopula(list(nc1, nc2), w = wts)
          y <- copula::rCopula(ny, copula = mcg)
          out=list(x=x, y=y)
          out
      }, param_alt = c(0, 0.5)))
    }
    if (which == "FrankJoeD5") {
        return(list(f=function(a=0) { 
          nc1 <- copula::frankCopula(5, dim=5L) # the first component
          nc2 <- copula::joeCopula(3, dim=5L) # the second component
          wts <- c(1, 0) # the corresponding weights
          mcg <- copula::mixCopula(list(nc1, nc2), w = wts)
          x <- copula::rCopula(nx, copula = mcg)
          wts <- c(1-a, a) # the corresponding weights
          mcg <- copula::mixCopula(list(nc1, nc2), w = wts)
          y <- copula::rCopula(ny, copula = mcg)
          out=list(x=x, y=y)
          out
      }, param_alt = c(0, 1)))
    }
    if (which == "FrankExponentialM5") {
        return(list(f = function(s = 1) {
            cop = copula::frankCopula(3, 5)
            x = copula::rCopula(nx, cop)
            y = copula::rCopula(ny, cop)
            out = list(x = x, y = y)
            out = MD2sample::change.marginals(out, "Exponential", s)
            out
        }, param_alt = c(1, 1.4)))
    }
    if (which == "FrankLinearM5") {
        return(list(f = function(s = 0) {
            cop = copula::frankCopula(3, 5)
            x = copula::rCopula(nx, cop)
            y = copula::rCopula(ny, cop)
            out = list(x = x, y = y)
            out = MD2sample::change.marginals(out, "Linear", s)
            out
        }, param_alt = c(0, 0.4)))
    }
    if (which == "FrankNormalM5") {
        return(list(f = function(s = 1) {
            cop = copula::frankCopula(3, 5)
            x = copula::rCopula(nx, cop)
            y = copula::rCopula(ny, cop)
            out = list(x = x, y = y)
            out = MD2sample::change.marginals(out, "Normal", s)
            out
        }, param_alt = c(1, 1.25)))
    }
     if (which == "ClaytonExponentialM5") {
        return(list(f = function(s = 1) {
            cop = copula::frankCopula(3, 5)
            x = copula::rCopula(nx, cop)
            y = copula::rCopula(ny, cop)
            out = list(x = x, y = y)
            out = MD2sample::change.marginals(out, "Exponential", s)
            out
        }, param_alt = c(1, 1.4)))
    }
    if (which == "ClaytonLinearM5") {
        return(list(f = function(s = 0) {
            cop = copula::frankCopula(3, 5)
            x = copula::rCopula(nx, cop)
            y = copula::rCopula(ny, cop)
            out = list(x = x, y = y)
            out = MD2sample::change.marginals(out, "Linear", s)
            out
        }, param_alt = c(0, 0.4)))
    }
    if (which == "ClaytonNormalM5") {
        return(list(f = function(s = 1) {
            cop = copula::frankCopula(3, 5)
            x = copula::rCopula(nx, cop)
            y = copula::rCopula(ny, cop)
            out = list(x = x, y = y)
            out = MD2sample::change.marginals(out, "Normal", s)
            out
        }, param_alt = c(1, 1.25)))
    }   
    if (which == "UniformExponentialM5") {
      return(list(f = function(s = 1) {
        x = matrix(stats::runif(5 * nx), nx, 5)
        y = matrix(stats::runif(5 * nx), nx, 5)
        for(i in 1:5) {
          x[,i]=qexp(x[,i], 1)
          y[,i]=qexp(y[,i], s)
        }
        out = list(x = x, y = y)
      }, param_alt = c(1, 1.175)))
    }
}