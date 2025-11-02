//! Core types for mango-iv option pricing
//!
//! These types are `#[repr(C)]` compatible with the existing C codebase.

/// Option type: call or put
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum OptionType {
    Call = 0,
    Put = 1,
}

/// Exercise style: European or American
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ExerciseType {
    European = 0,
    American = 1,
}

/// Option parameters for GPU pricing kernels
///
/// This struct contains all parameters needed for option pricing on GPU.
/// It uses `#[repr(C)]` for stable layout but is NOT intended for C FFI
/// in Phase 1. FFI bridge will be added in Phase 6 (Week 15-16).
///
/// Differences from C's `OptionData`:
/// - Includes `spot_price` (needed for GPU kernel dispatch)
/// - Uses `dividend_yield` instead of discrete dividends
/// - Includes `exercise_type` for kernel routing
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OptionParams {
    pub spot_price: f64,
    pub strike: f64,
    pub time_to_maturity: f64,
    pub risk_free_rate: f64,
    pub dividend_yield: f64,
    pub volatility: f64,
    pub option_type: OptionType,
    pub exercise_type: ExerciseType,
}

impl OptionParams {
    /// Create American put option parameters
    pub fn american_put(
        spot: f64,
        strike: f64,
        maturity: f64,
        rate: f64,
        volatility: f64,
    ) -> Self {
        Self {
            spot_price: spot,
            strike,
            time_to_maturity: maturity,
            risk_free_rate: rate,
            dividend_yield: 0.0,
            volatility,
            option_type: OptionType::Put,
            exercise_type: ExerciseType::American,
        }
    }

    /// Create American call option parameters
    pub fn american_call(
        spot: f64,
        strike: f64,
        maturity: f64,
        rate: f64,
        volatility: f64,
    ) -> Self {
        Self {
            spot_price: spot,
            strike,
            time_to_maturity: maturity,
            risk_free_rate: rate,
            dividend_yield: 0.0,
            volatility,
            option_type: OptionType::Call,
            exercise_type: ExerciseType::American,
        }
    }
}

/// Grid parameters for PDE solver
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GridParams {
    pub n_points: usize,
    pub x_min: f64,
    pub x_max: f64,
    pub dx: f64,
}

impl GridParams {
    /// Create uniform grid
    pub fn uniform(x_min: f64, x_max: f64, n_points: usize) -> Self {
        let dx = (x_max - x_min) / (n_points as f64 - 1.0);
        Self {
            n_points,
            x_min,
            x_max,
            dx,
        }
    }
}

/// Time domain parameters
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct TimeParams {
    pub t_start: f64,
    pub t_end: f64,
    pub dt: f64,
    pub n_steps: usize,
}

impl TimeParams {
    /// Create time domain with uniform steps
    pub fn uniform(t_start: f64, t_end: f64, n_steps: usize) -> Self {
        let dt = (t_end - t_start) / n_steps as f64;
        Self {
            t_start,
            t_end,
            dt,
            n_steps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_option_type_repr() {
        // Verify C-compatible memory layout
        assert_eq!(std::mem::size_of::<OptionType>(), 4);
        assert_eq!(OptionType::Call as i32, 0);
        assert_eq!(OptionType::Put as i32, 1);
    }

    #[test]
    fn test_exercise_type_repr() {
        assert_eq!(std::mem::size_of::<ExerciseType>(), 4);
        assert_eq!(ExerciseType::European as i32, 0);
        assert_eq!(ExerciseType::American as i32, 1);
    }

    #[test]
    fn test_option_params_layout() {
        // Verify struct size and alignment
        assert_eq!(std::mem::size_of::<OptionParams>(), 56);
        assert_eq!(std::mem::align_of::<OptionParams>(), 8);
    }

    #[test]
    fn test_american_put_constructor() {
        let params = OptionParams::american_put(100.0, 100.0, 1.0, 0.05, 0.20);
        assert_eq!(params.spot_price, 100.0);
        assert_eq!(params.strike, 100.0);
        assert_eq!(params.time_to_maturity, 1.0);
        assert_eq!(params.risk_free_rate, 0.05);
        assert_eq!(params.volatility, 0.20);
        assert_eq!(params.dividend_yield, 0.0);
        assert_eq!(params.option_type, OptionType::Put);
        assert_eq!(params.exercise_type, ExerciseType::American);
    }

    #[test]
    fn test_american_call_constructor() {
        let params = OptionParams::american_call(100.0, 105.0, 0.5, 0.03, 0.25);
        assert_eq!(params.option_type, OptionType::Call);
        assert_eq!(params.strike, 105.0);
    }

    #[test]
    fn test_grid_params() {
        let grid = GridParams::uniform(-1.0, 1.0, 101);
        assert_eq!(grid.n_points, 101);
        assert_eq!(grid.x_min, -1.0);
        assert_eq!(grid.x_max, 1.0);
        assert!((grid.dx - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_time_params() {
        let time = TimeParams::uniform(0.0, 1.0, 1000);
        assert_eq!(time.n_steps, 1000);
        assert_eq!(time.t_start, 0.0);
        assert_eq!(time.t_end, 1.0);
        assert!((time.dt - 0.001).abs() < 1e-10);
    }
}
