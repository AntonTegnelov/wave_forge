// Basic test module for wfc-gpu crate

#[cfg(test)]
mod unit_tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

// Test modules for specific components
#[cfg(test)]
pub mod entropy_strategies;
#[cfg(test)]
pub mod propagation_strategies;

// Place for other unit tests to be added
// For integration tests, use the tests directory at the crate root
