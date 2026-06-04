// SPDX-License-Identifier: MIT
use mango_option_sys as sys;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    Call,
    Put,
}

impl OptionType {
    pub(crate) fn to_c(self) -> sys::MangoOptionType {
        match self {
            OptionType::Call => sys::MANGO_CALL,
            OptionType::Put => sys::MANGO_PUT,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TenorPoint {
    pub tenor: f64,
    pub log_discount: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Dividend {
    pub calendar_time: f64,
    pub amount: f64,
}

/// Risk-free rate: a constant, or a yield curve given as tenor points
/// (the first point must be `tenor = 0, log_discount = 0`).
#[derive(Debug, Clone)]
pub enum Rate {
    Const(f64),
    Curve(Vec<TenorPoint>),
}

#[derive(Debug, Clone)]
pub struct OptionSpec {
    pub spot: f64,
    pub strike: f64,
    pub maturity: f64,
    pub dividend_yield: f64,
    pub rate: Rate,
    pub discrete_dividends: Vec<Dividend>,
    pub option_type: OptionType,
}
