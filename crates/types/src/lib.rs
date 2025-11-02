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
}
