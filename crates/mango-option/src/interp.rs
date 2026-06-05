// SPDX-License-Identifier: MIT
use crate::error::{Error, ErrorKind};
use crate::iv::{IvQuery, IvSuccess};
use crate::pricing::{blank_error, div_ptr_or_null, dividend_array, ptr_or_null, tenor_array};
use crate::types::{Dividend, OptionType, Rate};
use mango_option_sys as sys;

/// IV grid axes (S/K moneyness, NOT log). Default mirrors the C++ IVGrid.
#[derive(Debug, Clone)]
pub struct IvGrid {
    pub moneyness: Vec<f64>,
    pub vol: Vec<f64>,
    pub rate: Vec<f64>,
}
impl Default for IvGrid {
    fn default() -> Self {
        IvGrid {
            moneyness: vec![0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
            vol: vec![0.05, 0.10, 0.20, 0.30, 0.50],
            rate: vec![0.01, 0.03, 0.05, 0.10],
        }
    }
}

/// Newton config for the interpolated solver. Default mirrors C++ defaults.
#[derive(Debug, Clone, Copy)]
pub struct InterpSolverConfig {
    pub max_iter: usize,
    pub tolerance: f64,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub vega_threshold: f64,
}
impl Default for InterpSolverConfig {
    fn default() -> Self {
        InterpSolverConfig {
            max_iter: 50,
            tolerance: 1e-6,
            sigma_min: 0.01,
            sigma_max: 3.0,
            vega_threshold: 1e-4,
        }
    }
}
impl InterpSolverConfig {
    pub(crate) fn to_c(self) -> sys::MangoInterpSolverConfig {
        sys::MangoInterpSolverConfig {
            max_iter: self.max_iter as u64,
            tolerance: self.tolerance,
            sigma_min: self.sigma_min,
            sigma_max: self.sigma_max,
            vega_threshold: self.vega_threshold,
        }
    }
}

/// Adaptive grid refinement params. Default mirrors C++ defaults.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveGridParams {
    pub target_iv_error: f64,
    pub max_iter: usize,
    pub max_points_per_dim: usize,
    pub min_moneyness_points: usize,
    pub validation_samples: usize,
    pub refinement_factor: f64,
    pub lhs_seed: u64,
    pub vega_floor: f64,
    pub max_failure_rate: f64,
}
impl Default for AdaptiveGridParams {
    fn default() -> Self {
        AdaptiveGridParams {
            target_iv_error: 2e-5,
            max_iter: 5,
            max_points_per_dim: 160,
            min_moneyness_points: 60,
            validation_samples: 64,
            refinement_factor: 1.3,
            lhs_seed: 42,
            vega_floor: 1e-4,
            max_failure_rate: 0.5,
        }
    }
}
impl AdaptiveGridParams {
    fn to_c(&self) -> sys::MangoAdaptiveGridParams {
        sys::MangoAdaptiveGridParams {
            target_iv_error: self.target_iv_error,
            max_iter: self.max_iter as u64,
            max_points_per_dim: self.max_points_per_dim as u64,
            min_moneyness_points: self.min_moneyness_points as u64,
            validation_samples: self.validation_samples as u64,
            refinement_factor: self.refinement_factor,
            lhs_seed: self.lhs_seed,
            vega_floor: self.vega_floor,
            max_failure_rate: self.max_failure_rate,
        }
    }
}

/// Multi reference-strike config for the discrete-dividend path.
#[derive(Debug, Clone)]
pub struct MultiKRef {
    pub k_refs: Vec<f64>,
    pub k_ref_count: i32,
    pub k_ref_span: f64,
}
impl Default for MultiKRef {
    fn default() -> Self {
        MultiKRef { k_refs: Vec::new(), k_ref_count: 11, k_ref_span: 0.3 }
    }
}

#[derive(Debug, Clone)]
pub struct DiscreteDividendConfig {
    pub maturity: f64,
    pub dividends: Vec<Dividend>,
    pub kref_config: MultiKRef,
}

#[derive(Debug, Clone)]
pub struct FactoryConfig {
    pub option_type: OptionType,
    pub spot: f64,
    pub dividend_yield: f64,
    pub grid: IvGrid,
    pub maturity_grid: Vec<f64>,
    pub solver: InterpSolverConfig,
    pub adaptive: Option<AdaptiveGridParams>,
    pub discrete_dividends: Option<DiscreteDividendConfig>,
}

/// Result of a batch solve: per-query results plus the failure count.
#[derive(Debug)]
pub struct BatchResult {
    pub results: Vec<Result<IvSuccess, Error>>,
    pub failed: usize,
}

/// Interpolated IV solver over a pre-built B-spline surface. Thread-safe to
/// query concurrently (immutable surface).
pub struct InterpIvSolver {
    handle: *mut sys::MangoInterpIvSolver,
}
// SAFETY: the C++ AnyInterpIVSolver wraps an immutable surface; solve()/
// solve_batch() are const and documented thread-safe.
unsafe impl Send for InterpIvSolver {}
unsafe impl Sync for InterpIvSolver {}

impl core::fmt::Debug for InterpIvSolver {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "InterpIvSolver({:p})", self.handle)
    }
}

impl InterpIvSolver {
    pub fn new(cfg: &FactoryConfig) -> Result<Self, Error> {
        Ok(InterpIvSolver { handle: make_factory_handle(cfg)? })
    }

    /// Wrap a raw solver handle (e.g. one derived from a `PriceTable`). The
    /// returned `InterpIvSolver` takes ownership and frees it on drop.
    pub(crate) fn from_raw(handle: *mut sys::MangoInterpIvSolver) -> Self {
        InterpIvSolver { handle }
    }

    pub fn solve(&self, query: &IvQuery) -> Result<IvSuccess, Error> {
        let (c, _keep) = iv_query_to_c(query)?;
        let mut out = blank_iv_success();
        let mut err = blank_error();
        let status =
            unsafe { sys::mango_interp_iv_solve(self.handle, &c, &mut out, &mut err) };
        if status == sys::MANGO_OK {
            Ok(iv_success_from_c(&out))
        } else {
            Err(Error::from_c(status, &err))
        }
    }

    pub fn solve_batch(&self, queries: &[IvQuery]) -> BatchResult {
        // Per-query failures are isolated to their slot (a bad query does not
        // fail the whole batch). Rust-side conversion errors (e.g. an empty
        // yield curve, which would otherwise silently become rate 0 at the C
        // boundary) are recorded per-slot here; C-side build/solve failures are
        // recorded per-slot by the shim.
        let mut results: Vec<Option<Result<IvSuccess, Error>>> =
            (0..queries.len()).map(|_| None).collect();
        let mut c_queries = Vec::with_capacity(queries.len());
        let mut keepalive = Vec::with_capacity(queries.len());
        let mut valid_idx = Vec::with_capacity(queries.len()); // original index per c_query
        for (i, q) in queries.iter().enumerate() {
            match iv_query_to_c(q) {
                Ok((c, keep)) => {
                    c_queries.push(c);
                    keepalive.push(keep);
                    valid_idx.push(i);
                }
                Err(e) => results[i] = Some(Err(e)),
            }
        }
        let n = c_queries.len() as u64;
        let mut slots: Vec<sys::MangoIvBatchSlot> = (0..c_queries.len())
            .map(|_| sys::MangoIvBatchSlot { status: 0, success: blank_iv_success() })
            .collect();
        let mut failed: u64 = 0;
        let mut err = blank_error();
        let status = unsafe {
            sys::mango_interp_iv_solve_batch(
                self.handle,
                if c_queries.is_empty() { core::ptr::null() } else { c_queries.as_ptr() },
                n,
                if slots.is_empty() { core::ptr::null_mut() } else { slots.as_mut_ptr() },
                &mut failed,
                &mut err,
            )
        };
        // keepalive must outlive the FFI call above.
        let _ = &keepalive;
        if status != sys::MANGO_OK {
            // Whole-call failure (e.g. null handle): mark every still-pending
            // (successfully-converted) slot with the error.
            let e = Error::from_c(status, &err);
            for &i in &valid_idx {
                results[i] = Some(Err(e.clone()));
            }
        } else {
            for (k, &i) in valid_idx.iter().enumerate() {
                let s = &slots[k];
                results[i] = Some(if s.status == sys::MANGO_OK {
                    Ok(iv_success_from_c(&s.success))
                } else {
                    Err(Error { kind: ErrorKind::from_status(s.status),
                                message: batch_error_message(s.status) })
                });
            }
        }
        let results: Vec<Result<IvSuccess, Error>> =
            results.into_iter().map(|r| r.expect("every slot resolved")).collect();
        let failed = results.iter().filter(|r| r.is_err()).count();
        BatchResult { results, failed }
    }
}

impl Drop for InterpIvSolver {
    fn drop(&mut self) {
        unsafe { sys::mango_interp_iv_solver_free(self.handle) }
    }
}

// --- shared helpers used by interp.rs and table.rs ---

pub(crate) fn blank_iv_success() -> sys::MangoIvSuccess {
    sys::MangoIvSuccess {
        implied_vol: 0.0, iterations: 0, final_error: 0.0,
        vega: 0.0, has_vega: 0, used_rate_approximation: 0,
    }
}

pub(crate) fn iv_success_from_c(out: &sys::MangoIvSuccess) -> IvSuccess {
    IvSuccess {
        implied_vol: out.implied_vol,
        iterations: out.iterations as usize,
        final_error: out.final_error,
        vega: if out.has_vega != 0 { Some(out.vega) } else { None },
        used_rate_approximation: out.used_rate_approximation != 0,
    }
}

fn batch_error_message(status: i32) -> String {
    format!("interpolated IV solve failed ({})", ErrorKind::from_status(status))
}

// Backing arrays that must outlive an FFI call referencing their pointers.
pub(crate) struct IvQueryKeepalive {
    _tenors: Vec<sys::MangoTenorPoint>,
    _divs: Vec<sys::MangoDividend>,
}

pub(crate) fn iv_query_to_c(
    query: &IvQuery,
) -> Result<(sys::MangoIvQuery, IvQueryKeepalive), Error> {
    if matches!(&query.spec.rate, Rate::Curve(v) if v.is_empty()) {
        return Err(Error { kind: ErrorKind::Validation,
                           message: "yield curve has no tenor points".to_string() });
    }
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
    Ok((c, IvQueryKeepalive { _tenors: tenors, _divs: divs }))
}

// Build the C factory config, keeping every Vec alive across the FFI call,
// then invoke `f` with a pointer to the config and the error out-param.
pub(crate) fn with_c_config<T>(
    cfg: &FactoryConfig,
    f: impl FnOnce(*const sys::MangoIvFactoryConfig, *mut sys::MangoError) -> (i32, T),
) -> Result<T, Error> {
    let moneyness = cfg.grid.moneyness.clone();
    let vol = cfg.grid.vol.clone();
    let rate = cfg.grid.rate.clone();
    let maturity = cfg.maturity_grid.clone();

    let adaptive_c = cfg.adaptive.as_ref().map(|a| a.to_c());
    // Discrete-dividend sub-config: keep its Vecs alive too.
    let dd = cfg.discrete_dividends.as_ref().map(|d| {
        let divs = dividend_array(&d.dividends);
        let krefs = d.kref_config.k_refs.clone();
        let c = sys::MangoDiscreteDividendConfig {
            maturity: d.maturity,
            dividends: div_ptr_or_null(&divs),
            n_dividends: divs.len() as u64,
            kref_config: sys::MangoMultiKRef {
                K_refs: if krefs.is_empty() { core::ptr::null() } else { krefs.as_ptr() },
                n_K_refs: krefs.len() as u64,
                K_ref_count: d.kref_config.k_ref_count,
                K_ref_span: d.kref_config.k_ref_span,
            },
        };
        (c, divs, krefs)
    });

    let f64_ptr = |v: &[f64]| if v.is_empty() { core::ptr::null() } else { v.as_ptr() };

    let c = sys::MangoIvFactoryConfig {
        option_type: cfg.option_type.to_c(),
        spot: cfg.spot,
        dividend_yield: cfg.dividend_yield,
        moneyness: f64_ptr(&moneyness),
        n_moneyness: moneyness.len() as u64,
        vol: f64_ptr(&vol),
        n_vol: vol.len() as u64,
        rate: f64_ptr(&rate),
        n_rate: rate.len() as u64,
        maturity_grid: f64_ptr(&maturity),
        n_maturity: maturity.len() as u64,
        solver_config: cfg.solver.to_c(),
        adaptive: adaptive_c.as_ref().map_or(core::ptr::null(), |a| a as *const _),
        discrete_dividends: dd.as_ref().map_or(core::ptr::null(), |(c, _, _)| c as *const _),
    };

    let mut err = blank_error();
    let (status, value) = f(&c, &mut err);
    // keepalive: moneyness/vol/rate/maturity/adaptive_c/dd live until here.
    let _ = (&moneyness, &vol, &rate, &maturity, &adaptive_c, &dd);
    if status == sys::MANGO_OK {
        Ok(value)
    } else {
        Err(Error::from_c(status, &err))
    }
}

pub(crate) fn make_factory_handle(
    cfg: &FactoryConfig,
) -> Result<*mut sys::MangoInterpIvSolver, Error> {
    let mut handle: *mut sys::MangoInterpIvSolver = core::ptr::null_mut();
    with_c_config(cfg, |c, err| unsafe {
        (sys::mango_make_interp_iv_solver(c, &mut handle, err), handle)
    })
}

pub(crate) fn make_price_table_handle(
    cfg: &FactoryConfig,
) -> Result<*mut sys::MangoPriceTable, Error> {
    let mut handle: *mut sys::MangoPriceTable = core::ptr::null_mut();
    with_c_config(cfg, |c, err| unsafe {
        (sys::mango_make_price_table(c, &mut handle, err), handle)
    })
}
