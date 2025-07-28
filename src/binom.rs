use std::f64::consts::PI;

// https://github.com/wch/r-source/blob/7e572477ba67f17036e88fe5e2e30f54369da30b/src/nmath/dbinom.c
const M_LN_2PI: f64 = 1.837877066409345483560659472811;
const M_LN_SQRT_2PI: f64 = 0.918938533204672741780329736406;

pub fn dbinom(x: f64, n: f64, p: f64, give_log: bool) -> f64 {
    if !(0.0..=1.0).contains(&p) || n.is_nan() || x.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 || !x.is_finite() {
        return 0.0;
    }

    let n = n.round();
    let x = x.round();

    dbinom_raw(x, n, p, 1.0 - p, give_log)
}

fn dbinom_raw(x: f64, n: f64, p: f64, q: f64, give_log: bool) -> f64 {
    if p == 0.0 {
        return if x == 0.0 { 1.0 } else { 0.0 };
    }
    if q == 0.0 {
        return if x == n { 1.0 } else { 0.0 };
    }

    // The smaller of p and q is the most accurate
    if x == 0.0 {
        if n == 0.0 {
            return 1.0;
        }
        if p > q {
            return if give_log { n * q.ln() } else { q.powf(n) };
        } else {
            return if give_log {
                n * (-p).ln_1p()
            } else {
                pow1p(-p, n)
            };
        }
    }
    if x == n {
        if p > q {
            return if give_log {
                n * (-q).ln_1p()
            } else {
                pow1p(-q, n)
            };
        } else {
            return if give_log { n * p.ln() } else { p.powf(n) };
        }
    }
    if x < 0.0 || x > n {
        return 0.0;
    }

    let lc = stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0(x, n * p) - bd0(n - x, n * q);
    let lf = M_LN_2PI + x.ln() + (-x / n).ln_1p();

    (lc - 0.5 * lf).exp()
}

fn pow1p(x: f64, y: f64) -> f64 {
    if y.is_nan() {
        return if x == 0.0 { 1.0 } else { y };
    }
    if y >= 0.0 && y == y.trunc() && y <= 4.0 {
        match y as i32 {
            0 => return 1.0,
            1 => return x + 1.0,
            2 => return x * (x + 2.0) + 1.0,
            3 => return x * (x * (x + 3.0) + 3.0) + 1.0,
            4 => return x * (x * (x * (x + 4.0) + 6.0) + 4.0) + 1.0,
            _ => {},
        }
    }
    let xp1 = x + 1.0;
    let x_ = xp1 - 1.0;
    if x_ == x || x.abs() > 0.5 || x.is_nan() {
        xp1.powf(y)
    } else {
        (y * x.ln_1p()).exp()
    }
}

// https://github.com/wch/r-source/blob/7e572477ba67f17036e88fe5e2e30f54369da30b/src/nmath/stirlerr.c
const S0: f64 = 0.083333333333333333333; // 1/12
const S1: f64 = 0.00277777777777777777778; // 1/360
const S2: f64 = 0.00079365079365079365079365; // 1/1260
const S3: f64 = 0.000595238095238095238095238; // 1/1680
const S4: f64 = 0.0008417508417508417508417508; // 1/1188
const S5: f64 = 0.0019175269175269175269175262; // 691/360360
const S6: f64 = 0.0064102564102564102564102561; // 1/156
const S7: f64 = 0.029550653594771241830065352; // 3617/122400
const S8: f64 = 0.17964437236883057316493850; // 43867/244188
const S9: f64 = 1.3924322169059011164274315; // 174611/125400
const S10: f64 = 13.402864044168391994478957; // 77683/5796
const S11: f64 = 156.84828462600201730636509; // 236364091/1506960
const S12: f64 = 2193.1033333333333333333333; // 657931/300
const S13: f64 = 36108.771253724989357173269; // 3392780147/93960
const S14: f64 = 691472.26885131306710839498; // 1723168255201/2492028
const S15: f64 = 15238221.539407416192283370; // 7709321041217/505920
const S16: f64 = 382900751.39141414141414141; // 151628697551/396
const SFERR_HALVES: [f64; 31] = [
    0.0,                           /* n=0 - wrong, place holder only */
    0.1534264097200273452913848,   /* 0.5 */
    0.0810614667953272582196702,   /* 1.0 */
    0.0548141210519176538961390,   /* 1.5 */
    0.0413406959554092940938221,   /* 2.0 */
    0.03316287351993628748511048,  /* 2.5 */
    0.02767792568499833914878929,  /* 3.0 */
    0.02374616365629749597132920,  /* 3.5 */
    0.02079067210376509311152277,  /* 4.0 */
    0.01848845053267318523077934,  /* 4.5 */
    0.01664469118982119216319487,  /* 5.0 */
    0.01513497322191737887351255,  /* 5.5 */
    0.01387612882307074799874573,  /* 6.0 */
    0.01281046524292022692424986,  /* 6.5 */
    0.01189670994589177009505572,  /* 7.0 */
    0.01110455975820691732662991,  /* 7.5 */
    0.010411265261972096497478567, /* 8.0 */
    0.009799416126158803298389475, /* 8.5 */
    0.009255462182712732917728637, /* 9.0 */
    0.008768700134139385462952823, /* 9.5 */
    0.008330563433362871256469318, /* 10.0 */
    0.007934114564314020547248100, /* 10.5 */
    0.007573675487951840794972024, /* 11.0 */
    0.007244554301320383179543912, /* 11.5 */
    0.006942840107209529865664152, /* 12.0 */
    0.006665247032707682442354394, /* 12.5 */
    0.006408994188004207068439631, /* 13.0 */
    0.006171712263039457647532867, /* 13.5 */
    0.005951370112758847735624416, /* 14.0 */
    0.005746216513010115682023589, /* 14.5 */
    0.005554733551962801371038690, /* 15.0 */
];
fn stirlerr(n: f64) -> f64 {
    let mut nn: f64;
    if (n <= 23.5) {
        nn = n + n;
        if n <= 15.0 && nn == nn.trunc() {
            return SFERR_HALVES[nn.trunc() as usize];
        }
        // else:
        if n <= 5.25 {
            if n >= 1.0 {
                let l_n = n.ln();
                return libm::lgamma(n) + n * (1.0 - l_n) + ldexp(l_n - M_LN_2PI, -1);
            } else {
                return lgamma1p(n) - (n + 0.5) * n.ln() + n - M_LN_SQRT_2PI;
            }
        }
        nn = n * n;
        if (n > 12.8) {
            return (S0 - (S1 - (S2 - (S3 - (S4 - (S5 - S6 / nn) / nn) / nn) / nn) / nn) / nn) / n;
        }
        if (n > 12.3) {
            return (S0
                - (S1 - (S2 - (S3 - (S4 - (S5 - (S6 - S7 / nn) / nn) / nn) / nn) / nn) / nn) / nn)
                / n;
        }
        if (n > 8.9) {
            return (S0
                - (S1
                    - (S2 - (S3 - (S4 - (S5 - (S6 - (S7 - S8 / nn) / nn) / nn) / nn) / nn) / nn)
                        / nn)
                    / nn)
                / n;
        }
        if (n > 7.3) {
            return (S0
                - (S1
                    - (S2
                        - (S3
                            - (S4
                                - (S5
                                    - (S6 - (S7 - (S8 - (S9 - S10 / nn) / nn) / nn) / nn)
                                        / nn)
                                    / nn)
                                / nn)
                            / nn)
                        / nn)
                    / nn)
                / n;
        }
        if (n > 6.6) {
            return (S0
                - (S1
                    - (S2
                        - (S3
                            - (S4
                                - (S5
                                    - (S6
                                        - (S7
                                            - (S8
                                                - (S9
                                                    - (S10 - (S11 - S12 / nn) / nn) / nn)
                                                    / nn)
                                                / nn)
                                            / nn)
                                        / nn)
                                    / nn)
                                / nn)
                            / nn)
                        / nn)
                    / nn)
                / n;
        }
        if (n > 6.1) {
            return (S0
                - (S1
                    - (S2
                        - (S3
                            - (S4
                                - (S5
                                    - (S6
                                        - (S7
                                            - (S8
                                                - (S9
                                                    - (S10
                                                        - (S11
                                                            - (S12
                                                                - (S13 - S14 / nn)
                                                                    / nn)
                                                                / nn)
                                                            / nn)
                                                        / nn)
                                                    / nn)
                                                / nn)
                                            / nn)
                                        / nn)
                                    / nn)
                                / nn)
                            / nn)
                        / nn)
                    / nn)
                / n;
        }
        (S0 - (S1
            - (S2
                - (S3
                    - (S4
                        - (S5
                            - (S6
                                - (S7
                                    - (S8
                                        - (S9
                                            - (S10
                                                - (S11
                                                    - (S12
                                                        - (S13
                                                            - (S14
                                                                - (S15 - S16 / nn)
                                                                    / nn)
                                                                / nn)
                                                            / nn)
                                                        / nn)
                                                    / nn)
                                                / nn)
                                            / nn)
                                        / nn)
                                    / nn)
                                / nn)
                            / nn)
                        / nn)
                    / nn)
                / nn)
            / nn)
            / n
    } else {
        nn = n * n;
        if (n > 15.7e6) {
            return S0 / n;
        }
        if (n > 6180.0) {
            return (S0 - S1 / nn) / n;
        }
        if (n > 205.0) {
            return (S0 - (S1 - S2 / nn) / nn) / n;
        }
        if (n > 86.0) {
            return (S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / n;
        }
        if (n > 27.0) {
            return (S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n;
        }
        (S0 - (S1 - (S2 - (S3 - (S4 - S5 / nn) / nn) / nn) / nn) / nn) / n
    }
}

#[inline(always)]
fn ldexp(x: f64, exp: i32) -> f64 {
    x * (2.0f64).powi(exp)
}

//https://github.com/wch/r-source/blob/7e572477ba67f17036e88fe5e2e30f54369da30b/src/nmath/pgamma.c
const EULERS_CONST: f64 = 0.5772156649015328606065120900824024;
const N: usize = 40;
const COEFFS: [f64; 40] = [
    0.3224670334241132182362075833230126e-0, /* = (zeta(2)-1)/2 */
    0.6735230105319809513324605383715000e-1, /* = (zeta(3)-1)/3 */
    0.2058080842778454787900092413529198e-1,
    0.7385551028673985266273097291406834e-2,
    0.2890510330741523285752988298486755e-2,
    0.1192753911703260977113935692828109e-2,
    0.5096695247430424223356548135815582e-3,
    0.2231547584535793797614188036013401e-3,
    0.9945751278180853371459589003190170e-4,
    0.4492623673813314170020750240635786e-4,
    0.2050721277567069155316650397830591e-4,
    0.9439488275268395903987425104415055e-5,
    0.4374866789907487804181793223952411e-5,
    0.2039215753801366236781900709670839e-5,
    0.9551412130407419832857179772951265e-6,
    0.4492469198764566043294290331193655e-6,
    0.2120718480555466586923135901077628e-6,
    0.1004322482396809960872083050053344e-6,
    0.4769810169363980565760193417246730e-7,
    0.2271109460894316491031998116062124e-7,
    0.1083865921489695409107491757968159e-7,
    0.5183475041970046655121248647057669e-8,
    0.2483674543802478317185008663991718e-8,
    0.1192140140586091207442548202774640e-8,
    0.5731367241678862013330194857961011e-9,
    0.2759522885124233145178149692816341e-9,
    0.1330476437424448948149715720858008e-9,
    0.6422964563838100022082448087644648e-10,
    0.3104424774732227276239215783404066e-10,
    0.1502138408075414217093301048780668e-10,
    0.7275974480239079662504549924814047e-11,
    0.3527742476575915083615072228655483e-11,
    0.1711991790559617908601084114443031e-11,
    0.8315385841420284819798357793954418e-12,
    0.4042200525289440065536008957032895e-12,
    0.1966475631096616490411045679010286e-12,
    0.9573630387838555763782200936508615e-13,
    0.4664076026428374224576492565974577e-13,
    0.2273736960065972320633279596737272e-13,
    0.1109139947083452201658320007192334e-13, /* = (zeta(40+1)-1)/(40+1) */
];
const C: f64 = 0.2273736845824652515226821577978691e-12; /* zeta(N+2)-1 */
const TOL_LOGCF: f64 = 1e-14;

fn lgamma1p(a: f64) -> f64 {
    if (a.abs() >= 0.5) {
        return lgammafn(a + 1.0);
    }

    let mut lgam = C * logcf(-a / 2.0, (N + 2) as f64, 1.0, TOL_LOGCF);
    for c in COEFFS {
        lgam = c - a * lgam;
    }

    (a * lgam - EULERS_CONST) * a - log1pmx(a)
}

// https://github.com/wch/r-source/blob/7e572477ba67f17036e88fe5e2e30f54369da30b/src/nmath/pgamma.c
fn log1pmx(x: f64) -> f64 {
    const MIN_LOG1_VALUE: f64 = -0.79149064;

    if !(MIN_LOG1_VALUE..=1.0).contains(&x) {
        x.ln_1p() - x
    } else {
        /* -.791 <=  x <= 1  -- expand in  [x/(2+x)]^2 =: y :
         * log(1+x) - x =  x/(2+x) * [ 2 * y * S(y) - x],  with
         * ---------------------------------------------
         * S(y) = 1/3 + y/5 + y^2/7 + ... = \sum_{k=0}^\infty  y^k / (2k + 3)
         */
        let r = x / (2.0 + x);
        let y = r * r;
        if (x.abs() < 1e-2) {
            r * ((((2.0 / 9.0 * y + 2.0 / 7.0) * y + 2.0 / 5.0) * y + 2.0 / 3.0) * y - x)
        } else {
            r * (2.0 * y * logcf(y, 3.0, 2.0, TOL_LOGCF) - x)
        }
    }
}

const fn sqr(x: f64) -> f64 {
    x * x
}
const SCALE_FACTOR: f64 = sqr(sqr(sqr(4294967296.0)));

fn logcf(x: f64, i: f64, d: f64, eps: f64) -> f64 {
    let mut c1 = 2.0 * d;
    let mut c2 = i + d;
    let mut c4 = c2 + d;
    let mut a1 = c2;
    let mut b1 = i * (c2 - i * x);
    let mut b2 = d * d * x;
    let mut a2 = c4 * c2 - b2;

    let mut b2 = c4 * b1 - i * b2;

    while ((a2 * b1 - a1 * b2).abs() > (eps * b1 * b2).abs()) {
        let mut c3 = c2 * c2 * x;
        c2 += d;
        c4 += d;
        a1 = c4 * a2 - c3 * a1;
        b1 = c4 * b2 - c3 * b1;

        c3 = c1 * c1 * x;
        c1 += d;
        c4 += d;
        a2 = c4 * a1 - c3 * a2;
        b2 = c4 * b1 - c3 * b2;

        if ((b2).abs() > SCALE_FACTOR) {
            a1 /= SCALE_FACTOR;
            b1 /= SCALE_FACTOR;
            a2 /= SCALE_FACTOR;
            b2 /= SCALE_FACTOR;
        } else if ((b2).abs() < 1.0 / SCALE_FACTOR) {
            a1 *= SCALE_FACTOR;
            b1 *= SCALE_FACTOR;
            a2 *= SCALE_FACTOR;
            b2 *= SCALE_FACTOR;
        }
    }

    a2 / b2
}

// https://github.com/wch/r-source/blob/7e572477ba67f17036e88fe5e2e30f54369da30b/src/nmath/lgamma.c
const M_LN_SQRT_PID2: f64 = 0.225791352644727432363097614947;
fn lgammafn(x: f64) -> f64 {
    let xmax = d1mach(2) / (d1mach(2).ln());
    let dxrel = (d1mach(4).sqrt());

    if (x.is_nan()) {
        return x;
    }

    if (x <= 0.0 && x == x.trunc()) {
        return f64::INFINITY;
    }

    let y = x.abs();

    if (y < 1e-306) {
        return -(y.ln());
    }
    if (y <= 10.0) {
        return ((gammafn(x)).abs()).ln();
    }

    if (y > xmax) {
        return f64::INFINITY;
    }

    if (x > 0.0) {
        /* i.e. y = x > 10 */
        if (x > 1e17) {
            return (x * (x.ln() - 1.));
        } else if (x > 4934720.) {
            return (M_LN_SQRT_2PI + (x - 0.5) * x.ln() - x);
        } else {
            return M_LN_SQRT_2PI + (x - 0.5) * x.ln() - x + lgammacor(x);
        }
    }
    /* else: x < -10; y = -x > 10 */
    let sinpiy = (sinpi(y).abs());

    let ans = M_LN_SQRT_PID2 + (x - 0.5) * y.ln() - x - sinpiy.ln() - lgammacor(y);

    ans
}

const M_LOG10_2: f64 = 0.301029995663981195213738894724;

// https://github.com/wch/r-source/blob/7e572477ba67f17036e88fe5e2e30f54369da30b/src/nmath/d1mach.c
fn d1mach(i: usize) -> f64 {
    match (i) {
        1 => f64::MIN,
        2 => f64::MAX,
        3 => 0.5 * f64::EPSILON,
        4 => f64::EPSILON,
        5 => M_LOG10_2,
        _ => 0.0,
    }
}

// https://github.com/wch/r-source/blob/7e572477ba67f17036e88fe5e2e30f54369da30b/src/nmath/cospi.c
fn sinpi(mut x: f64) -> f64 {
    if (x.is_nan()) {
        return x;
    }
    if (!x.is_finite()) {
        return f64::NAN;
    }

    x %= 2.0;
    // map (-2,2) --> (-1,1] :
    if (x <= -1.0) {
        x += 2.;
    } else if (x > 1.0) {
        x -= 2.;
    }
    if (x == 0.0 || x == 1.0) {
        return 0.0;
    }
    if (x == 0.5) {
        return 1.0;
    }
    if (x == -0.5) {
        return -1.0;
    }
    (PI * x).sin()
}
// https://github.com/wch/r-source/blob/7e572477ba67f17036e88fe5e2e30f54369da30b/src/nmath/lgammacor.c
const ALGMCS: [f64; 15] = [
    0.1666389480451863247205729650822e+0,
    -0.1384948176067563840732986059135e-4,
    0.9810825646924729426157171547487e-8,
    -0.1809129475572494194263306266719e-10,
    0.6221098041892605227126015543416e-13,
    -0.3399615005417721944303330599666e-15,
    0.2683181998482698748957538846666e-17,
    -0.2868042435334643284144622399999e-19,
    0.3962837061046434803679306666666e-21,
    -0.6831888753985766870111999999999e-23,
    0.1429227355942498147573333333333e-24,
    -0.3547598158101070547199999999999e-26,
    0.1025680058010470912000000000000e-27,
    -0.3401102254316748799999999999999e-29,
    0.1276642195630062933333333333333e-30,
];
const NALGM: usize = 5;
const XBIG: f64 = 94906265.62425156;
fn lgammacor(x: f64) -> f64 {
    if (x < 10.0)
    // possibly consider stirlerr()
    {
        return f64::NAN;
    } else if (x < XBIG) {
        let tmp = 10.0 / x;
        return chebyshev_eval(tmp * tmp * 2.0 - 1.0, &ALGMCS, NALGM) / x;
    }
    1.0 / (x * 12.0)
}

// https://github.com/wch/r-source/blob/7e572477ba67f17036e88fe5e2e30f54369da30b/src/nmath/chebyshev.c
fn chebyshev_init(dos: &[f64], nos: usize, eta: f64) -> usize {
    if (nos < 1) {
        return 0;
    }

    let mut err = 0.0;
    let mut i = 0; /* just to avoid compiler warnings */
    for ii in 1..=nos {
        i = nos - ii;
        err += (dos[i]).abs();
        if (err > eta) {
            return i;
        }
    }
    i
}
fn chebyshev_eval(x: f64, a: &[f64], n: usize) -> f64 {
    if (n < 1 || n > 1000) {
        return f64::NAN;
    };

    if (x < -1.1 || x > 1.1) {
        return f64::NAN;
    };

    let twox = x * 2.0;
    let mut b1 = 0.0;
    let mut b2 = 0.0;
    let mut b0 = 0.0;
    for i in 1..=n {
        b2 = b1;
        b1 = b0;
        b0 = twox * b1 - b2 + a[n - i];
    }
    return (b0 - b2) * 0.5;
}

// https://github.com/wch/r-source/blob/7e572477ba67f17036e88fe5e2e30f54369da30b/src/nmath/gamma.c
const GAMCS: [f64; 42] = [
    0.8571195590989331421920062399942e-2,
    0.4415381324841006757191315771652e-2,
    0.5685043681599363378632664588789e-1,
    -0.4219835396418560501012500186624e-2,
    0.1326808181212460220584006796352e-2,
    -0.1893024529798880432523947023886e-3,
    0.3606925327441245256578082217225e-4,
    -0.6056761904460864218485548290365e-5,
    0.1055829546302283344731823509093e-5,
    -0.1811967365542384048291855891166e-6,
    0.3117724964715322277790254593169e-7,
    -0.5354219639019687140874081024347e-8,
    0.9193275519859588946887786825940e-9,
    -0.1577941280288339761767423273953e-9,
    0.2707980622934954543266540433089e-10,
    -0.4646818653825730144081661058933e-11,
    0.7973350192007419656460767175359e-12,
    -0.1368078209830916025799499172309e-12,
    0.2347319486563800657233471771688e-13,
    -0.4027432614949066932766570534699e-14,
    0.6910051747372100912138336975257e-15,
    -0.1185584500221992907052387126192e-15,
    0.2034148542496373955201026051932e-16,
    -0.3490054341717405849274012949108e-17,
    0.5987993856485305567135051066026e-18,
    -0.1027378057872228074490069778431e-18,
    0.1762702816060529824942759660748e-19,
    -0.3024320653735306260958772112042e-20,
    0.5188914660218397839717833550506e-21,
    -0.8902770842456576692449251601066e-22,
    0.1527474068493342602274596891306e-22,
    -0.2620731256187362900257328332799e-23,
    0.4496464047830538670331046570666e-24,
    -0.7714712731336877911703901525333e-25,
    0.1323635453126044036486572714666e-25,
    -0.2270999412942928816702313813333e-26,
    0.3896418998003991449320816639999e-27,
    -0.6685198115125953327792127999999e-28,
    0.1146998663140024384347613866666e-28,
    -0.1967938586345134677295103999999e-29,
    0.3376448816585338090334890666666e-30,
    -0.5793070335782135784625493333333e-31,
];
fn gammafn(x: f64) -> f64 {
    let ngam = chebyshev_init(&GAMCS, 42, f64::EPSILON / 20.0);
    let xmin = -170.5674972726612;
    let xmax = 171.61447887182298;
    let xsml = ((f64::MIN).ln().max(-(f64::MAX.ln())) + 0.01).exp();
    let dxrel = f64::EPSILON.sqrt();

    if (x.is_nan()) {
        return x;
    }

    /* If the argument is exactly zero or a negative integer
     * then return NaN. */
    if (x == 0.0 || (x < 0.0 && x == x.round())) {
        return f64::NAN;
    }

    let i: usize;
    let mut y = x.abs();
    let mut value: f64;

    if (y <= 10.0) {
        let mut n = x.trunc() as isize;
        if (x < 0.0) {
            n -= 1;
        }
        y = x - n as f64;
        n -= 1;
        value = chebyshev_eval(y * 2.0 - 1.0, &GAMCS, ngam) + 0.9375;
        if (n == 0) {
            return value;
        }

        if (n < 0) {
            if (y < xsml) {
                if (x > 0.0) {
                    return f64::INFINITY;
                } else {
                    return f64::NEG_INFINITY;
                }
            }

            n = -n;

            for i in 0..(n as usize) {
                value /= (x + i as f64);
            }
            return value;
        } else {
            for i in 0..(n as usize) {
                value *= (y + i as f64);
            }
            return value;
        }
    } else {
        if (x > xmax) {
            return f64::INFINITY;
        }

        if (x < xmin) {
            return 0.0;
        }

        if (y <= 50.0 && y == y.trunc()) {
            value = 1.0;
            for i in 2..(y.trunc() as usize) {
                value *= i as f64;
            }
        } else {
            /* normal case */
            value = ((y - 0.5) * y.ln() - y
                + M_LN_SQRT_2PI
                + (if (2.0 * y == (2.0 * y).trunc()) {
                    stirlerr(y)
                } else {
                    lgammacor(y)
                }))
            .exp();
        }

        if (x > 0.0) {
            return value;
        }

        let sinpiy = sinpi(y);
        if (sinpiy == 0.0) {
            return f64::INFINITY;
        }

        return -PI / (y * sinpiy * value);
    }
}
// https://github.com/wch/r-source/blob/7e572477ba67f17036e88fe5e2e30f54369da30b/src/nmath/bd0.c
fn bd0(x: f64, np: f64) -> f64 {
    if (!x.is_finite() || !np.is_finite() || np == 0.0) {
        return f64::NAN;
    }

    if ((x - np).abs() < 0.1 * (x + np)) {
        let d = x - np;
        let mut v = d / (x + np);
        if ((d != 0.0) && (v == 0.0)) {
            // v has underflown to 0 (as  x+np = inf)
            let x_ = ldexp(x, -2);
            let n_ = ldexp(np, -2);
            v = (x_ - n_) / (x_ + n_);
        }
        let mut s = ldexp(d, -1) * v; // was d * v
        if ((ldexp(s, 1)).abs() < f64::MIN) {
            return ldexp(s, 1);
        }
        let mut ej = x * v; // as 2*x*v could overflow:  v > 1/2  <==> ej = 2xv > x
        v *= v; // "v = v^2"
        for j in 1..1000 {
            ej *= v; // = x v^(2j+1)
            let s_ = s;
            s += ej / ((j << 1) as f64 + 1.0);
            if (s == s_) {
                /* last term was effectively 0 */
                return ldexp(s, 1); // 2*s ; as we dropped '2 *' above
            }
        }
    }
    let lg_x_n = {
        if (x / np).is_finite() {
            (x / np).ln()
        } else {
            x.ln() - np.ln()
        }
    };
    if x > np {
        x * (lg_x_n - 1.0) + np
    } else {
        x * lg_x_n + np - x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dbinom() {
        assert_eq!(dbinom(1.0, 1.0, 1.0, false), 1.0);
        assert_eq!(dbinom(2.0, 1.0, 0.5, false), 0.0);
        assert_eq!(dbinom(2.0, 1.0, 0.0, false), 0.0);
    }
}
