#[derive(Default)]
pub struct BsOptions<'a> {
    df: Option<usize>,
    knots: Option<&'a [f64]>,
    degree: usize,
    intercept: bool,
    boundary_knots: Option<&'a [f64]>,
    warn_outside: bool,
}

// pub fn bs(x: &[f64], BsOptions { df, knots, degree, intercept,
// boundary_knots, warn_outside }: BsOptions) {     let ord = 1 + degree;
//     let outside = if let Some(boundary_knots) = boundary_knots {
//         let boundary_knots = boundary_knots;
//         let ol = x.iter().map(|&x| x <
// boundary_knots[0]).collect::<Vec<_>>();         let or = x.iter().map(|&x| x
// > boundary_knots[1]).collect::<Vec<_>>();         ol.iter().zip(or.iter()).
// map(|(&ol, &or)| ol || or).collect::<Vec<_>>()     } else {
//         vec![false; x.len()]
//     };
//     outside <- if(!missing(Boundary.knots)) {
//         Boundary.knots <- sort(Boundary.knots)
//         (ol <- x < Boundary.knots[1L]) | (or <- x > Boundary.knots[2L])
//     } else FALSE
//
//     if(mk.knots <- !is.null(df) && is.null(knots)) {
// 	nIknots <- df - ord + (1L - intercept) # ==  #{inner knots}
//         if(nIknots < 0L) {
//             nIknots <- 0L
//             warning(gettextf("'df' was too small; have used %d",
//                              ord - (1L - intercept)), domain = NA)
//         }
//         knots <-
//             if(nIknots > 0L) {
//                 knots <- seq.int(from = 0, to = 1,
//                                  length.out = nIknots + 2L)[-c(1L, nIknots +
// 2L)]                 quantile(x[!outside], knots, names=FALSE)
//             }
//     }
//     else if(!all(is.finite(knots))) stop("non-finite knots")
//     if(mk.knots && length(knots) && any(lrEq <- range(knots) %in%
// Boundary.knots)) {         if(lrEq[1L]) {
//             aE <- all(i <- knots == (piv <- Boundary.knots[1L]))
//             if(aE)
//                 warning("all interior knots match left boundary knot")
//             else
//                 knots[i] <- knots[i] + (min(knots[knots > piv]) - piv)/8
//         }
//         if(lrEq[2L]) {
//             aE2 <- all(i <- knots == (piv <- Boundary.knots[2L]))
//             if(aE2)
//                 warning("all interior knots match right boundary knot")
//             else
//                 knots[i] <- knots[i] - (piv - max(knots[knots < piv]))/8
//         }
//         if(!(lrEq[1L] && aE || lrEq[2L] && aE2)) # haven't warned yet
//             warning("shoving 'interior' knots matching boundary knots to
// inside")     }
//     Aknots <- sort(c(rep(Boundary.knots, ord), knots))
//     if(any(outside)) {
//         if(warn.outside) warning("some 'x' values beyond boundary knots may
// cause ill-conditioned bases")         derivs <- 0:degree
//         scalef <- gamma(1L:ord)# factorials
//         basis <- array(0, c(length(x), length(Aknots) - degree - 1L))
// 	e <- 1/4 # in theory anything in (0,1); was (implicitly) 0 in R <= 3.2.2
//         if(any(ol)) {
// 	    ## left pivot inside, i.e., a bit to the right of the boundary knot
// 	    k.pivot <- (1-e)*Boundary.knots[1L] + e*Aknots[ord+1]
//             xl <- cbind(1, outer(x[ol] - k.pivot, 1L:degree, `^`))
//             tt <- splineDesign(Aknots, rep(k.pivot, ord), ord, derivs)
//             basis[ol, ] <- xl %*% (tt/scalef)
//         }
//         if(any(or)) {
// 	    ## right pivot inside, i.e., a bit to the left of the boundary knot:
// 	    k.pivot <- (1-e)*Boundary.knots[2L] + e*Aknots[length(Aknots)-ord]
//             xr <- cbind(1, outer(x[or] - k.pivot, 1L:degree, `^`))
//             tt <- splineDesign(Aknots, rep(k.pivot, ord), ord, derivs)
//             basis[or, ] <- xr %*% (tt/scalef)
//         }
//         if(any(inside <- !outside))
//             basis[inside,  ] <- splineDesign(Aknots, x[inside], ord)
//     }
//     else basis <- splineDesign(Aknots, x, ord)
//     if(!intercept)
//         basis <- basis[, -1L , drop = FALSE]
//     n.col <- ncol(basis)
//     if(nas) {
//         nmat <- matrix(NA, length(nax), n.col)
//         nmat[!nax,  ] <- basis
//         basis <- nmat
//     }
//     dimnames(basis) <- list(nx, 1L:n.col)
//     a <- list(degree = degree, knots = if(is.null(knots)) numeric(0L) else
// knots,               Boundary.knots = Boundary.knots, intercept = intercept)
//     attributes(basis) <- c(attributes(basis), a)
//     class(basis) <- c("bs", "basis", "matrix")
//     basis
// }

// ns <- function(x, df = NULL, knots = NULL, intercept = FALSE,
//                Boundary.knots = range(x))
// {
//     nx <- names(x)
//     x <- as.vector(x)
//     nax <- is.na(x)
//     if(nas <- any(nax))
//         x <- x[!nax]
//     outside <- if(!missing(Boundary.knots)) {
//         Boundary.knots <- sort(Boundary.knots)
//         (ol <- x < Boundary.knots[1L]) | (or <- x > Boundary.knots[2L])
//     }
//     else {
// 	if(length(x) == 1L) ## && missing(Boundary.knots) : special treatment
// 	    Boundary.knots <- x*c(7,9)/8 # symmetrically around x
// 	FALSE # rep(FALSE, length = length(x))
//     }
//     if(mk.knots <- !is.null(df) && is.null(knots)) {
//         ## df = number(interior knots) + 1 + intercept
//         nIknots <- df - 1L - intercept
//         if(nIknots < 0L) {
//             nIknots <- 0L
//             warning(gettextf("'df' was too small; have used %d",
//                              1L + intercept), domain = NA)
//         }
//         knots <-
//             if(nIknots > 0L) {
//                 knots <- seq.int(from = 0, to = 1,
//                                  length.out = nIknots + 2L)[-c(1L, nIknots +
// 2L)]                 quantile(x[!outside], knots, names=FALSE)
//             }
//     } else {
//         if(!all(is.finite(knots))) stop("non-finite knots")
//         nIknots <- length(knots)
//     }
//     if(mk.knots && length(knots) && any(lrEq <- range(knots) %in%
// Boundary.knots)) {         if(lrEq[1L]) {
//             i <- knots == (piv <- Boundary.knots[1L])
//             if(all(i)) stop("all interior knots match left boundary knot")
//             knots[i] <- knots[i] + (min(knots[knots > piv]) - piv)/8
//         }
//         if(lrEq[2L]) {
//             i <- knots == (piv <- Boundary.knots[2L])
//             if(all(i)) stop("all interior knots match right boundary knot")
//             knots[i] <- knots[i] - (piv - max(knots[knots < piv]))/8
//         }
//         warning("shoving 'interior' knots matching boundary knots to inside")
//     }
//     Aknots <- sort(c(rep(Boundary.knots, 4L), knots))
//     if(any(outside)) {
//         basis <- array(0, c(length(x), nIknots + 4L))
//         if(any(ol)) {
//             k.pivot <- Boundary.knots[1L]
//             xl <- cbind(1, x[ol] - k.pivot)
//             tt <- splineDesign(Aknots, rep(k.pivot, 2L), 4, c(0, 1))
//             basis[ol,  ] <- xl %*% tt
//         }
//         if(any(or)) {
//             k.pivot <- Boundary.knots[2L]
//             xr <- cbind(1, x[or] - k.pivot)
//             tt <- splineDesign(Aknots, rep(k.pivot, 2L), 4, c(0, 1))
//             basis[or,  ] <- xr %*% tt
//         }
//         if(any(inside <- !outside))
//             basis[inside,  ] <- splineDesign(Aknots, x[inside], 4)
//     }
//     else basis <- splineDesign(Aknots, x, ord = 4L)
//     const <- splineDesign(Aknots, Boundary.knots, ord = 4L, derivs = c(2L,
// 2L))     if(!intercept) {
//         const <- const[, -1 , drop = FALSE]
//         basis <- basis[, -1 , drop = FALSE]
//     }
//     qr.const <- qr(t(const))
//     basis <- as.matrix((t(qr.qty(qr.const, t(basis))))[,  - (1L:2L), drop =
// FALSE])     n.col <- ncol(basis)
//     if(nas) {
//         nmat <- matrix(NA, length(nax), n.col)
//         nmat[!nax, ] <- basis
//         basis <- nmat
//     }
//     dimnames(basis) <- list(nx, 1L:n.col)
//     a <- list(degree = 3L, knots = if(is.null(knots)) numeric() else knots,
//               Boundary.knots = Boundary.knots, intercept = intercept)
//     attributes(basis) <- c(attributes(basis), a)
//     class(basis) <- c("ns", "basis", "matrix")
//     basis
// }

// pub fn splineDesign(
// splineDesign <-
//     ## Creates the "design matrix" for a collection of B-splines.
//     function(knots, x, ord = 4L, derivs = 0L, outer.ok = FALSE,
//              sparse = FALSE)
// {
//     if((nk <- length(knots <- as.numeric(knots))) <= 0)
//         stop("must have at least 'ord' knots")
//     if(is.unsorted(knots)) knots <- sort.int(knots)
//     x <- as.numeric(x)
//     nx <- length(x)
//     ## derivs is re-cycled to length(x) in C
//     if(length(derivs) > nx)
// 	stop("length of 'derivs' is larger than length of 'x'")
//     if(length(derivs) < 1L) stop("empty 'derivs'")
//     ord <- as.integer(ord)
//     if(ord > nk || ord < 1)
// 	stop("'ord' must be positive integer, at most the number of knots")
//
//     ## The x test w/ sorted knots assumes ord <= nk+1-ord, or nk >= 2*ord-1L:
//     if(!outer.ok && nk < 2*ord-1)
//         stop(gettextf("need at least %s (=%d) knots",
//                       "2*ord -1", 2*ord -1),
//              domain = NA)
//
//     degree <- ord - 1L
// ### FIXME: the 'outer.ok && need.outer' handling would more efficiently
// happen ###        in the underlying C code - with some programming effort
// though..     if(need.outer <- any(x < knots[ord] | knots[nk - degree] < x)) {
//         if(outer.ok) { ## x[] is allowed to be 'anywhere'
// 	    in.x <- knots[1L] <= x & x <= knots[nk]
// 	    if((x.out <- !all(in.x))) {
// 		x <- x[in.x]
// 		nnx <- length(x)
// 	    }
// 	    ## extend knots set "temporarily": the boundary knots must be repeated
// >= 'ord' times.             ## NB: If these are already repeated originally,
// then, on the *right* only, we need             ##    to make sure not to add
// more than needed             dkn <- diff(knots)[(nk-1L):1] # >= 0, since they
// are sorted 	    knots <- knots[c(rep.int(1L, degree),
//                              seq_len(nk),
//                              rep.int(nk, max(0L, ord - match(TRUE, dkn >
// 0))))] 	} else
// 	    stop(gettextf("the 'x' data must be in the range %g to %g unless you set
// '%s'", 			  knots[ord], knots[nk - degree], "outer.ok = TRUE"),
// 		 domain = NA)
//     }
//     temp <- .Call(C_spline_basis, knots, ord, x, derivs)
//     ncoef <- nk - ord
//
//     ii <- if(need.outer && x.out) { # only assign non-zero for x[]'s "inside"
// knots         rep.int((1L:nx)[in.x], rep.int(ord, nnx))
//     } else rep.int(1L:nx, rep.int(ord, nx))
//     jj <- c(outer(1L:ord, attr(temp, "Offsets"), `+`))
//     ## stopifnot(length(ii) == length(jj))
//
//     if(sparse) {
// 	if(is.null(tryCatch(loadNamespace("Matrix"), error = function(e)NULL)))
// 	    stop(gettextf("%s needs package 'Matrix' correctly installed",
//                           "splineDesign(*, sparse=TRUE)"),
//                  domain = NA)
// 	if(need.outer) { ## shift column numbers and drop those "outside"
// 	    jj <- jj - degree - 1L
// 	    ok <- 0 <= jj & jj < ncoef
// 	    methods::as(methods::new("dgTMatrix", i = ii[ok] - 1L, j = jj[ok],
// 				     x = as.double(temp[ok]), # vector, not matrix
// 				     Dim = c(nx, ncoef)), "CsparseMatrix")
// 	}
// 	else
// 	    methods::as(methods::new("dgTMatrix", i = ii - 1L, j = jj - 1L,
// 				     x = as.double(temp), # vector
// 				     Dim = c(nx, ncoef)), "CsparseMatrix")
//     } else { ## traditional (dense) matrix
// 	design <- matrix(double(nx * ncoef), nx, ncoef)
// 	if(need.outer) { ## shift column numbers and drop those "outside"
// 	    jj <- jj - degree
// 	    ok <- 1 <= jj & jj <= ncoef
// 	    design[cbind(ii, jj)[ok , , drop=FALSE]] <- temp[ok]
// 	}
// 	else
// 	    design[cbind(ii, jj)] <- temp
// 	design
//     }
// }
