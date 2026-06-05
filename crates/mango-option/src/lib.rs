// SPDX-License-Identifier: MIT
//! Safe Rust bindings for the mango-option American option pricer and FDM
//! implied-vol solver. Callers write no `unsafe`.
mod error;
mod interp;
mod iv;
mod pricing;
mod types;

pub use error::{Error, ErrorKind};
pub use interp::{
    AdaptiveGridParams, BatchResult, DiscreteDividendConfig, FactoryConfig, InterpIvSolver,
    InterpSolverConfig, IvGrid, MultiKRef,
};
pub use iv::{solve_iv, IvConfig, IvQuery, IvSuccess};
pub use pricing::{price_american, PriceResult, PricingParams};
pub use types::{Dividend, OptionSpec, OptionType, Rate, TenorPoint};
