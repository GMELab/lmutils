#[derive(Debug, Clone)]
pub struct Coef {
    label: String,
    coef: f64,
    std_err: f64,
    t: f64,
    p: f64,
}

impl Coef {
    pub fn new(label: impl ToString, coef: f64, std_err: f64, t: f64, p: f64) -> Self {
        Coef {
            label: label.to_string(),
            coef,
            std_err,
            t,
            p,
        }
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn coef(&self) -> f64 {
        self.coef
    }

    pub fn std_err(&self) -> f64 {
        self.std_err
    }

    pub fn t(&self) -> f64 {
        self.t
    }

    pub fn p(&self) -> f64 {
        self.p
    }
}
