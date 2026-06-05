// SPDX-License-Identifier: MIT
use crate::error::Error;
use crate::interp::{make_price_table_handle, FactoryConfig, InterpIvSolver, InterpSolverConfig};
use crate::pricing::{blank_error, pricing_params_to_c, PricingParams};
use crate::types::OptionType;
use mango_option_sys as sys;

/// A reusable, immutable B-spline price surface. Thread-safe to query.
pub struct PriceTable {
    handle: *mut sys::MangoPriceTable,
}
// SAFETY: the C++ AnyPriceTable wraps an immutable surface; all query methods
// are const and documented thread-safe.
unsafe impl Send for PriceTable {}
unsafe impl Sync for PriceTable {}

impl core::fmt::Debug for PriceTable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "PriceTable({:p})", self.handle)
    }
}

impl PriceTable {
    pub fn new(cfg: &FactoryConfig) -> Result<Self, Error> {
        Ok(PriceTable { handle: make_price_table_handle(cfg)? })
    }

    /// Opt-in bounds check; out-of-domain params => Err. `price`/`vega` do NOT
    /// call this and will extrapolate.
    pub fn validate(&self, params: &PricingParams) -> Result<(), Error> {
        let (c, _keep) = pricing_params_to_c(params)?;
        let mut err = blank_error();
        let status = unsafe { sys::mango_price_table_validate(self.handle, &c, &mut err) };
        let _ = &_keep;
        if status == sys::MANGO_OK { Ok(()) } else { Err(Error::from_c(status, &err)) }
    }

    /// Interpolated price. Extrapolates out of the surface domain; returns NaN
    /// only on an internal failure. Call `validate` first if you need bounds.
    pub fn price(&self, params: &PricingParams) -> f64 {
        match pricing_params_to_c(params) {
            Ok((c, _keep)) => {
                let v = unsafe { sys::mango_price_table_price(self.handle, &c) };
                let _ = &_keep;
                v
            }
            Err(_) => f64::NAN,
        }
    }
    pub fn vega(&self, params: &PricingParams) -> f64 {
        match pricing_params_to_c(params) {
            Ok((c, _keep)) => {
                let v = unsafe { sys::mango_price_table_vega(self.handle, &c) };
                let _ = &_keep;
                v
            }
            Err(_) => f64::NAN,
        }
    }

    pub fn delta(&self, params: &PricingParams) -> Result<f64, Error> {
        self.greek(params, sys::mango_price_table_delta)
    }
    pub fn gamma(&self, params: &PricingParams) -> Result<f64, Error> {
        self.greek(params, sys::mango_price_table_gamma)
    }
    pub fn theta(&self, params: &PricingParams) -> Result<f64, Error> {
        self.greek(params, sys::mango_price_table_theta)
    }
    pub fn rho(&self, params: &PricingParams) -> Result<f64, Error> {
        self.greek(params, sys::mango_price_table_rho)
    }

    fn greek(
        &self,
        params: &PricingParams,
        f: unsafe extern "C" fn(*const sys::MangoPriceTable, *const sys::MangoPricingParams,
                                *mut f64, *mut sys::MangoError) -> sys::MangoStatus,
    ) -> Result<f64, Error> {
        let (c, _keep) = pricing_params_to_c(params)?;
        let mut out = 0.0;
        let mut err = blank_error();
        let status = unsafe { f(self.handle, &c, &mut out, &mut err) };
        let _ = &_keep;
        if status == sys::MANGO_OK { Ok(out) } else { Err(Error::from_c(status, &err)) }
    }

    pub fn option_type(&self) -> OptionType {
        let t = unsafe { sys::mango_price_table_option_type(self.handle) };
        if t == sys::MANGO_CALL { OptionType::Call } else { OptionType::Put }
    }
    pub fn dividend_yield(&self) -> f64 {
        unsafe { sys::mango_price_table_dividend_yield(self.handle) }
    }

    /// Derive an IV solver from this already-built surface (no rebuild).
    pub fn iv_solver(&self, cfg: Option<&InterpSolverConfig>) -> Result<InterpIvSolver, Error> {
        let c = cfg.map(|c| c.to_c());
        let mut handle: *mut sys::MangoInterpIvSolver = core::ptr::null_mut();
        let mut err = blank_error();
        let status = unsafe {
            sys::mango_price_table_make_iv_solver(
                self.handle,
                c.as_ref().map_or(core::ptr::null(), |c| c as *const _),
                &mut handle,
                &mut err,
            )
        };
        if status == sys::MANGO_OK {
            Ok(InterpIvSolver::from_raw(handle))
        } else {
            Err(Error::from_c(status, &err))
        }
    }
}

impl Drop for PriceTable {
    fn drop(&mut self) {
        unsafe { sys::mango_price_table_free(self.handle) }
    }
}
