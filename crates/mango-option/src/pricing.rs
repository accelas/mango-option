// SPDX-License-Identifier: MIT
use crate::error::{Error, ErrorKind};
use crate::types::{Dividend, OptionSpec, Rate, TenorPoint};
use mango_option_sys as sys;

pub struct PricingParams {
    pub spec: OptionSpec,
    pub volatility: f64,
}

/// Owns a C++ `AmericanOptionResult`. `!Send + !Sync` (raw pointer): a single
/// result must not be used concurrently because `value_at` drives the C++
/// object's lazy, mutable spline/operator caches.
pub struct PriceResult {
    handle: *mut sys::MangoAmericanResult,
}

impl core::fmt::Debug for PriceResult {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "PriceResult({:p})", self.handle)
    }
}

impl PriceResult {
    pub fn value(&self) -> f64 {
        unsafe { sys::mango_american_value(self.handle) }
    }
    pub fn delta(&self) -> f64 {
        unsafe { sys::mango_american_delta(self.handle) }
    }
    pub fn gamma(&self) -> f64 {
        unsafe { sys::mango_american_gamma(self.handle) }
    }
    pub fn theta(&self) -> f64 {
        unsafe { sys::mango_american_theta(self.handle) }
    }
    pub fn value_at(&self, spot: f64) -> Result<f64, Error> {
        let mut out = 0.0;
        let mut err = blank_error();
        let status = unsafe {
            sys::mango_american_value_at(self.handle, spot, &mut out, &mut err)
        };
        if status == sys::MANGO_OK { Ok(out) } else { Err(Error::from_c(status, &err)) }
    }
}

impl Drop for PriceResult {
    fn drop(&mut self) {
        unsafe { sys::mango_american_result_free(self.handle) }
    }
}

pub fn price_american(params: &PricingParams) -> Result<PriceResult, Error> {
    if matches!(&params.spec.rate, Rate::Curve(v) if v.is_empty()) {
        return Err(Error { kind: ErrorKind::Validation,
                           message: "yield curve has no tenor points".to_string() });
    }
    // Keep arrays alive for the duration of the call.
    let tenors = tenor_array(&params.spec.rate);
    let divs = dividend_array(&params.spec.discrete_dividends);
    let rate_const = match params.spec.rate {
        Rate::Const(r) => r,
        Rate::Curve(_) => 0.0,
    };
    let c = sys::MangoPricingParams {
        spot: params.spec.spot,
        strike: params.spec.strike,
        maturity: params.spec.maturity,
        dividend_yield: params.spec.dividend_yield,
        volatility: params.volatility,
        rate_const,
        tenor_points: ptr_or_null(&tenors),
        n_tenor_points: tenors.len() as u64,
        dividends: div_ptr_or_null(&divs),
        n_dividends: divs.len() as u64,
        option_type: params.spec.option_type.to_c(),
    };
    let mut handle: *mut sys::MangoAmericanResult = core::ptr::null_mut();
    let mut err = blank_error();
    let status = unsafe { sys::mango_price_american(&c, &mut handle, &mut err) };
    if status == sys::MANGO_OK {
        Ok(PriceResult { handle })
    } else {
        Err(Error::from_c(status, &err))
    }
}

// --- shared marshalling helpers (also used by iv.rs) ---

pub(crate) fn blank_error() -> sys::MangoError {
    sys::MangoError { code: 0, message: [0; 256] }
}

pub(crate) fn tenor_array(rate: &Rate) -> Vec<sys::MangoTenorPoint> {
    match rate {
        Rate::Const(_) => Vec::new(),
        Rate::Curve(points) => points
            .iter()
            .map(|p: &TenorPoint| sys::MangoTenorPoint {
                tenor: p.tenor,
                log_discount: p.log_discount,
            })
            .collect(),
    }
}

pub(crate) fn dividend_array(divs: &[Dividend]) -> Vec<sys::MangoDividend> {
    divs.iter()
        .map(|d| sys::MangoDividend { calendar_time: d.calendar_time, amount: d.amount })
        .collect()
}

pub(crate) fn ptr_or_null(v: &[sys::MangoTenorPoint]) -> *const sys::MangoTenorPoint {
    if v.is_empty() { core::ptr::null() } else { v.as_ptr() }
}

pub(crate) fn div_ptr_or_null(v: &[sys::MangoDividend]) -> *const sys::MangoDividend {
    if v.is_empty() { core::ptr::null() } else { v.as_ptr() }
}
