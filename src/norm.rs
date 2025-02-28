use std::f64::consts::PI;

// https://github.com/wch/r-source/blob/c0993c9ab73c4964b3d160e608631622fe01f0fb/src/nmath/dpq.h#L52
#[inline(always)]
fn r_dt_qiv(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if log_p {
        if lower_tail {
            p.exp()
        } else {
            -p.exp_m1()
        }
    } else {
        r_d_lval(p, lower_tail)
    }
}

// https://github.com/wch/r-source/blob/c0993c9ab73c4964b3d160e608631622fe01f0fb/src/nmath/dpq.h#L33
#[inline(always)]
fn r_d_lval(p: f64, lower_tail: bool) -> f64 {
    if lower_tail {
        p
    } else {
        (0.5 - p) + 0.5
    }
}

// https://github.com/wch/r-source/blob/c0993c9ab73c4964b3d160e608631622fe01f0fb/src/nmath/dpq.h#L56
#[inline(always)]
fn r_dt_civ(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if log_p {
        if lower_tail {
            -p.exp_m1()
        } else {
            p.exp()
        }
    } else {
        r_d_cval(p, lower_tail)
    }
}

// https://github.com/wch/r-source/blob/c0993c9ab73c4964b3d160e608631622fe01f0fb/src/nmath/dpq.h#L34
#[inline(always)]
fn r_d_cval(p: f64, lower_tail: bool) -> f64 {
    if lower_tail {
        (0.5 - p) + 0.5
    } else {
        p
    }
}
pub fn qnorm(p: f64) -> f64 {
    qnorm5(p, 0.0, 1.0, true, false)
}

// lower_tail is normally true, log_p is normally false
// taken from R's qnorm5, https://github.com/wch/r-source/blob/c0993c9ab73c4964b3d160e608631622fe01f0fb/src/nmath/qnorm.c
pub fn qnorm5(p: f64, mu: f64, sigma: f64, lower_tail: bool, log_p: bool) -> f64 {
    let mut r: f64;
    let mut val: f64;

    if p.is_nan() || mu.is_nan() || sigma.is_nan() {
        return f64::NAN;
    }
    let left = f64::NEG_INFINITY;
    let right = f64::INFINITY;
    if log_p {
        if p > 0.0 {
            return f64::NAN;
        }
        if p == 0.0 {
            return if lower_tail { right } else { left };
        }
        if p == f64::NEG_INFINITY {
            return if lower_tail { left } else { right };
        }
    } else {
        /* !log_p */
        if !(0.0..=1.0).contains(&p) {
            return f64::NAN;
        }
        if (p == 0.0) {
            return if lower_tail { left } else { right };
        }
        if (p == 1.0) {
            return if lower_tail { right } else { left };
        }
    }
    if sigma < 0.0 {
        return f64::NAN;
    }
    if sigma == 0.0 {
        return mu;
    }

    let p_ = r_dt_qiv(p, lower_tail, log_p);
    let q = p_ - 0.5;

    if (q.abs() <= 0.425) {
        r = 0.180625 - q * q;
        val = q
            * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r
                + 67265.770927008700853)
                * r
                + 45921.953931549871457)
                * r
                + 13731.693765509461125)
                * r
                + 1971.5909503065514427)
                * r
                + 133.14166789178437745)
                * r
                + 3.387132872796366608)
            / (((((((r * 5226.495278852854561 + 28729.085735721942674) * r
                + 39307.89580009271061)
                * r
                + 21213.794301586595867)
                * r
                + 5394.1960214247511077)
                * r
                + 687.1870074920579083)
                * r
                + 42.313330701600911252)
                * r
                + 1.0);
    } else {
        r = (-if log_p && ((lower_tail && q <= 0.0) || (!lower_tail && q > 0.0)) {
            p
        } else {
            if q > 0.0 {
                r_dt_civ(p, lower_tail, log_p)
            } else {
                p_
            }
            .ln()
        })
        .sqrt();

        if (r <= 5.) {
            r += -1.6;
            val = (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r
                + 0.24178072517745061177)
                * r
                + 1.27045825245236838258)
                * r
                + 3.64784832476320460504)
                * r
                + 5.7694972214606914055)
                * r
                + 4.6303378461565452959)
                * r
                + 1.42343711074968357734)
                / (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r
                    + 0.0151986665636164571966)
                    * r
                    + 0.14810397642748007459)
                    * r
                    + 0.68976733498510000455)
                    * r
                    + 1.6763848301838038494)
                    * r
                    + 2.05319162663775882187)
                    * r
                    + 1.0);
        } else {
            r += -5.;
            val = (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r
                + 0.0012426609473880784386)
                * r
                + 0.026532189526576123093)
                * r
                + 0.29656057182850489123)
                * r
                + 1.7848265399172913358)
                * r
                + 5.4637849111641143699)
                * r
                + 6.6579046435011037772)
                / (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r
                    + 1.8463183175100546818e-5)
                    * r
                    + 7.868691311456132591e-4)
                    * r
                    + 0.0148753612908506148525)
                    * r
                    + 0.13692988092273580531)
                    * r
                    + 0.59983220655588793769)
                    * r
                    + 1.0);
        }

        if (q < 0.0) {
            val = -val;
        }
    }

    mu + sigma * val
}

const M_1_SQRT_2PI: f64 = 0.398942280401432677939946059934;
const M_LN_SQRT_2PI: f64 = 0.918938533204672741780329736406;
const M_LN2: f64 = 0.693147180559945309417232121458;

#[inline(always)]
fn ldexp(x: f64, exp: i32) -> f64 {
    x * (2.0f64).powi(exp)
}

pub fn dnorm(x: f64) -> f64 {
    dnorm4(x, 0.0, 1.0, false)
}

// https://github.com/wch/r-source/blob/c0993c9ab73c4964b3d160e608631622fe01f0fb/src/nmath/dnorm.c
pub fn dnorm4(mut x: f64, mu: f64, sigma: f64, log: bool) -> f64 {
    if x.is_nan() || mu.is_nan() || sigma.is_nan() {
        return x + mu + sigma;
    }
    if sigma < 0.0 {
        return f64::NAN;
    }
    if !sigma.is_finite() {
        return r_d_0(log);
    }
    if !x.is_finite() && mu == x {
        return f64::NAN;
    }
    if sigma == 0.0 {
        return if x == mu { f64::INFINITY } else { r_d_0(log) };
    }
    x = (x - mu) / sigma;

    if !x.is_finite() {
        return r_d_0(log);
    }

    x = x.abs();
    if (x >= (2.0 * f64::MAX).sqrt()) {
        return r_d_0(log);
    }
    if (log) {
        return -(M_LN_SQRT_2PI * x * x + sigma.sqrt().ln());
    }
    if (x < 5.0) {
        return M_1_SQRT_2PI * (-0.5 * x * x).exp() / sigma;
    }

    if (x > (-2.0 * M_LN2 * (f64::MIN_EXP as f64 + 1.0 - f64::MANTISSA_DIGITS as f64)).sqrt()) {
        return 0.0;
    }

    let x1 = ldexp(ldexp(x, 16).round(), -16);
    let x2 = x - x1;
    M_1_SQRT_2PI / sigma * ((-0.5 * x1 * x1).exp() * ((-0.5 * x2 - x1) * x2).exp())
}

// https://github.com/wch/r-source/blob/da3c6a4782457665bb312b255c69e3582b78948f/src/nmath/dpq.h#L25
#[inline(always)]
fn r_d_0(log_p: bool) -> f64 {
    if log_p {
        f64::NEG_INFINITY
    } else {
        0.0
    }
}

// https://github.com/wch/r-source/blob/da3c6a4782457665bb312b255c69e3582b78948f/src/nmath/dpq.h#L26
#[inline(always)]
fn r_d_1(log_p: bool) -> f64 {
    if log_p {
        0.0
    } else {
        1.0
    }
}

// https://github.com/wch/r-source/blob/da3c6a4782457665bb312b255c69e3582b78948f/src/nmath/dpq.h#L27
#[inline(always)]
fn r_dt_0(lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail {
        r_d_0(log_p)
    } else {
        r_d_1(log_p)
    }
}

// https://github.com/wch/r-source/blob/da3c6a4782457665bb312b255c69e3582b78948f/src/nmath/dpq.h#L28
#[inline(always)]
fn r_dt_1(lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail {
        r_d_1(log_p)
    } else {
        r_d_0(log_p)
    }
}

#[inline(always)]
fn d_2(x: f64) -> f64 {
    ldexp(x, -1)
}

#[inline(always)]
fn do_del(x: f64, cum: &mut f64, ccum: &mut f64, temp: f64, lower: bool, upper: bool, log_p: bool) {
    let xsq = ldexp(ldexp(x, 4).trunc(), -4);
    let del = (x - xsq) * (x + xsq);
    if log_p {
        *cum = (-xsq * d_2(xsq)) - d_2(del) + temp.ln();
        if (lower && x > 0.) || (upper && x <= 0.) {
            *ccum = (-(-xsq * d_2(xsq)).exp() * (-d_2(del)).exp() * temp).ln_1p();
        }
    } else {
        *cum = (-xsq * d_2(xsq)).exp() * (-d_2(del)).exp() * temp;
        *ccum = 1.0 - *cum;
    }
}

const M_SQRT_32: f64 = 5.656854249492380195206754896838;

pub fn pnorm(x: f64) -> f64 {
    pnorm5(x, 0.0, 1.0, true, false)
}

// https://github.com/wch/r-source/blob/c0993c9ab73c4964b3d160e608631622fe01f0fb/src/nmath/pnorm.c
pub fn pnorm5(mut x: f64, mu: f64, sigma: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x.is_nan() || mu.is_nan() || sigma.is_nan() {
        return x + mu + sigma;
    }
    if !x.is_finite() && mu == x {
        return f64::NAN;
    }
    if sigma <= 0.0 {
        if sigma < 0.0 {
            return f64::NAN;
        }
        return if x < mu {
            r_dt_0(lower_tail, log_p)
        } else {
            r_dt_1(lower_tail, log_p)
        };
    }
    let p = (x - mu) / sigma;
    if !p.is_finite() {
        return if x < mu {
            r_dt_0(lower_tail, log_p)
        } else {
            r_dt_1(lower_tail, log_p)
        };
    }
    x = p;

    let (p, cp) = pnorm_both(x, p, if lower_tail { 0 } else { 1 }, log_p);

    if lower_tail {
        p
    } else {
        cp
    }
}

fn pnorm_both(x: f64, p: f64, i_tail: i32, log_p: bool) -> (f64, f64) {
    let mut cum = p;
    let mut ccum = 0.0;
    let a: [f64; 5] = [
        2.2352520354606839287,
        161.02823106855587881,
        1067.6894854603709582,
        18154.981253343561249,
        0.065682337918207449113,
    ];
    let b: [f64; 4] = [
        47.20258190468824187,
        976.09855173777669322,
        10260.932208618978205,
        45507.789335026729956,
    ];
    let c: [f64; 9] = [
        0.39894151208813466764,
        8.8831497943883759412,
        93.506656132177855979,
        597.27027639480026226,
        2494.5375852903726711,
        6848.1904505362823326,
        11602.651437647350124,
        9842.7148383839780218,
        1.0765576773720192317e-8,
    ];
    let d: [f64; 8] = [
        22.266688044328115691,
        235.38790178262499861,
        1519.377599407554805,
        6485.558298266760755,
        18615.571640885098091,
        34900.952721145977266,
        38912.003286093271411,
        19685.429676859990727,
    ];
    let p: [f64; 6] = [
        0.21589853405795699,
        0.1274011611602473639,
        0.022235277870649807,
        0.001421619193227893466,
        2.9112874951168792e-5,
        0.02307344176494017303,
    ];
    let q: [f64; 5] = [
        1.28426009614491121,
        0.468238212480865118,
        0.0659881378689285515,
        0.00378239633202758244,
        7.29751555083966205e-5,
    ];

    let (mut xden, mut xnum, mut temp, eps, xsq, y);
    // double xden, xnum, temp, del, eps, xsq, y;
    let (mut lower, mut upper);

    if x.is_nan() {
        cum = x;
        ccum = x;
        return (cum, ccum);
    }

    eps = f64::EPSILON * 0.5;

    lower = i_tail != 1;
    upper = i_tail != 0;

    y = x.abs();
    if (y <= 0.67448975) {
        if (y > eps) {
            xsq = x * x;
            xnum = a[4] * xsq;
            xden = xsq;
            for i in 0..3 {
                xnum = (xnum + a[i]) * xsq;
                xden = (xden + b[i]) * xsq;
            }
        } else {
            xnum = 0.0;
            xden = 0.0;
        }

        temp = x * (xnum + a[3]) / (xden + b[3]);
        if lower {
            cum = 0.5 + temp
        };
        if upper {
            ccum = 0.5 - temp
        };
        if (log_p) {
            if lower {
                cum = cum.ln();
            }
            if upper {
                ccum = ccum.ln();
            }
        }
    } else if (y <= M_SQRT_32) {
        xnum = c[8] * y;
        xden = y;
        for i in 0..7 {
            xnum = (xnum + c[i]) * y;
            xden = (xden + d[i]) * y;
        }
        temp = (xnum + c[7]) / (xden + d[7]);

        do_del(y, &mut cum, &mut ccum, temp, lower, upper, log_p);
        if (x > 0.) {
            temp = cum;
            if lower {
                cum = ccum;
            }
            ccum = temp;
        }
    } else if (log_p && y < 1e170)
        || (lower && -38.4674 < x && x < 8.2924)
        || (upper && -8.2924 < x && x < 38.4674)
    {
        xsq = 1.0 / (x * x); /* (1./x)*(1./x) might be better */
        xnum = p[5] * xsq;
        xden = xsq;
        for i in 0..4 {
            xnum = (xnum + p[i]) * xsq;
            xden = (xden + q[i]) * xsq;
        }
        temp = xsq * (xnum + p[4]) / (xden + q[4]);
        temp = (M_1_SQRT_2PI - temp) / y;

        do_del(y, &mut cum, &mut ccum, temp, lower, upper, log_p);
        if (x > 0.) {
            temp = cum;
            if lower {
                cum = ccum;
            }
            ccum = temp;
        }
    } else if (x > 0.0) {
        cum = r_d_1(log_p);
        ccum = r_d_0(log_p);
    } else {
        cum = r_d_0(log_p);
        ccum = r_d_1(log_p);
    }
    (cum, ccum)
}

fn fmod(x: f64, y: f64) -> f64 {
    x - y * (x / y).floor()
}

// https://github.com/wch/r-source/blob/da3c6a4782457665bb312b255c69e3582b78948f/src/nmath/cospi.c#L73
fn tanpi(mut x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if !x.is_finite() {
        return f64::NAN;
    }

    x = fmod(x, 1.0);
    if (x <= -0.5) {
        x += 1.0;
    } else if (x > 0.5) {
        x -= 1.0;
    }

    if x == 0.0 {
        0.0
    } else if x == 0.5 {
        f64::NAN
    } else if x == 0.25 {
        1.0
    } else if x == -0.25 {
        -1.0
    } else {
        (PI * x).tan()
    }
}

pub fn qcauchy(p: f64) -> f64 {
    qcauchy_inner(p, 0.0, 1.0, true, false)
}

// https://github.com/wch/r-source/blob/da3c6a4782457665bb312b255c69e3582b78948f/src/nmath/qcauchy.c
pub fn qcauchy_inner(
    mut p: f64,
    location: f64,
    scale: f64,
    mut lower_tail: bool,
    log_p: bool,
) -> f64 {
    if p.is_nan() || location.is_nan() || scale.is_nan() {
        return p + location + scale;
    }
    if (log_p && p > 0.0) || (!log_p && !(0.0..=1.0).contains(&p)) {
        return f64::NAN;
    }
    if scale <= 0.0 || !scale.is_finite() {
        if (scale == 0.0) {
            return location;
        }
        return f64::NAN;
    }

    if (log_p) {
        if (p > -1.0) {
            if (p == 0.0) {
                return location + (if lower_tail { scale } else { -scale }) * f64::INFINITY;
            }
            lower_tail = !lower_tail;
            p = -p.exp_m1();
        } else {
            p = p.exp();
        }
    } else if (p > 0.5) {
        if (p == 1.0) {
            return location + (if lower_tail { scale } else { -scale }) * f64::INFINITY;
        }
        p = 1.0 - p;
        lower_tail = !lower_tail;
    }

    if (p == 0.5) {
        return location;
    }
    if (p == 0.) {
        return location + (if lower_tail { scale } else { -scale }) * f64::NEG_INFINITY;
    }
    location + (if lower_tail { -scale } else { scale }) / tanpi(p)
}

pub fn dcauchy(x: f64) -> f64 {
    dcauchy_inner(x, 0.0, 1.0, false)
}

pub fn dcauchy_inner(x: f64, location: f64, scale: f64, give_log: bool) -> f64 {
    if x.is_nan() || location.is_nan() || scale.is_nan() {
        return x + location + scale;
    }
    if scale <= 0.0 {
        return f64::NAN;
    }

    let y = (x - location) / scale;
    if give_log {
        -(PI * scale * (1.0 + y * y)).ln()
    } else {
        1.0 / (PI * scale * (1.0 + y * y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_log::test;

    #[test]
    fn test_qnorm() {
        assert_eq!(qnorm(0.0), f64::NEG_INFINITY);
        assert_eq!(qnorm(0.3), -0.5244005127080406669648);
        assert_eq!(qnorm(0.5), 0.0);
        assert_eq!(qnorm(0.7), 0.5244005127080406669648);
        assert_eq!(qnorm(0.8), 0.8416212335729144067287);
        assert_eq!(qnorm(1.0), f64::INFINITY);
    }

    #[test]
    fn test_dnorm() {
        assert_eq!(dnorm(0.0), 0.3989422804014327028632);
        assert_eq!(dnorm(1.0), 0.2419707245191433653275);
        assert_eq!(dnorm(5.0), 1.486719514734297677895e-6);
    }

    #[test]
    fn test_pnorm() {
        assert_eq!(pnorm(-5.0), 2.866515718791939118538e-07);
        assert_eq!(pnorm(-1.0), 0.1586552539314570464679);
        assert_eq!(pnorm(0.0), 0.5);
        assert_eq!(pnorm(1.0), 0.8413447460685429257765);
        assert_eq!(pnorm(5.0), 0.9999997133484280764648);
    }

    #[test]
    fn test_qcauchy() {
        assert_eq!(qcauchy(0.0), f64::NEG_INFINITY);
        assert_eq!(qcauchy(0.3), -0.7265425280053610102016);
        assert_eq!(qcauchy(0.5), 0.0);
        assert_eq!(qcauchy(0.7), 0.726542528005360788157);
        assert_eq!(qcauchy(1.0), f64::INFINITY);
    }

    #[test]
    fn test_dcauchy() {
        assert_eq!(dcauchy(-5.0), 0.01224268793014579408129);
        assert_eq!(dcauchy(-1.0), 0.1591549430918953456082);
        assert_eq!(dcauchy(0.0), 0.3183098861837906912164);
        assert_eq!(dcauchy(1.0), 0.1591549430918953456082);
        assert_eq!(dcauchy(5.0), 0.01224268793014579408129);
    }
}
