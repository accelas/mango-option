// SPDX-License-Identifier: MIT
use crate::error::Error;
use crate::pricing::{blank_error, div_ptr_or_null, dividend_array, ptr_or_null, tenor_array};
use crate::types::{OptionSpec, Rate};
use mango_option_sys as sys;

pub struct IvQuery {
    pub spec: OptionSpec,
    pub market_price: f64,
}

#[derive(Default)]
pub struct IvConfig {
    pub max_iter: Option<u32>,
    pub brent_tol_abs: Option<f64>,
}

#[derive(Debug)]
pub struct IvSuccess {
    pub implied_vol: f64,
    pub iterations: usize,
    pub final_error: f64,
    pub vega: Option<f64>,
}

pub fn solve_iv(query: &IvQuery, config: &IvConfig) -> Result<IvSuccess, Error> {
    let tenors = tenor_array(&query.spec.rate);
    let divs = dividend_array(&query.spec.discrete_dividends);
    let rate_const = match query.spec.rate {
        Rate::Const(r) => r,
        Rate::Curve(_) => 0.0,
    };
    let c = sys::MangoIvQuery {
        spot: query.spec.spot,
        strike: query.spec.strike,
        maturity: query.spec.maturity,
        dividend_yield: query.spec.dividend_yield,
        market_price: query.market_price,
        rate_const,
        tenor_points: ptr_or_null(&tenors),
        n_tenor_points: tenors.len() as u64,
        dividends: div_ptr_or_null(&divs),
        n_dividends: divs.len() as u64,
        option_type: query.spec.option_type.to_c(),
    };
    let cfg = sys::MangoIvConfig {
        brent_tol_abs: config.brent_tol_abs.unwrap_or(0.0),
        max_iter: config.max_iter.map(|m| i32::try_from(m).unwrap_or(i32::MAX)).unwrap_or(0),
    };
    let mut out = sys::MangoIvSuccess {
        implied_vol: 0.0,
        iterations: 0,
        final_error: 0.0,
        vega: 0.0,
        has_vega: 0,
    };
    let mut err = blank_error();
    let status = unsafe { sys::mango_solve_iv(&c, &cfg, &mut out, &mut err) };
    if status == sys::MANGO_OK {
        Ok(IvSuccess {
            implied_vol: out.implied_vol,
            iterations: out.iterations as usize,
            final_error: out.final_error,
            vega: if out.has_vega != 0 { Some(out.vega) } else { None },
        })
    } else {
        Err(Error::from_c(status, &err))
    }
}
