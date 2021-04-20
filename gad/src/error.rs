// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

use backtrace::Backtrace;
use std::fmt::Debug;
use thiserror::Error;

/// Default error type for the crate.
#[derive(Error, Debug, Clone)]
pub enum Error {
    #[error("Incompatible dimensions for {name}: {dimensions}\n{trace}")]
    Dimensions {
        name: String,
        dimensions: String,
        trace: String,
    },
    #[error(
        "Incorrect reduction of dimensions for {name}: {dimensions} {reduced_dimensions}\n{trace}"
    )]
    ReducedDimensions {
        name: String,
        dimensions: String,
        reduced_dimensions: String,
        trace: String,
    },
    #[error("Incompatible lengths for {name}: {lengths}\n{trace}")]
    Lengths {
        name: String,
        lengths: String,
        trace: String,
    },
    #[error("Unexpected empty input for {name}\n{trace}")]
    Empty { name: String, trace: String },
    #[error("Trying to obtain `id` of a constant value.")]
    MissingId { name: String, trace: String },
    #[error("No gradient of the expected type could be found in gradient store.")]
    MissingGradient { name: String, trace: String },
    #[error("Trying to obtain a node from an incorrect `id`.")]
    MissingNode { name: String, trace: String },
}

/// Default result type for the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Computes the name of the current function.
// https://stackoverflow.com/questions/38088067/equivalent-of-func-or-function-in-rust
#[macro_export]
macro_rules! func_name {
    () => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        let name = type_name_of(f);
        &name[..name.len() - 3]
    }};
}

impl Error {
    fn backtrace() -> String {
        if std::env::var("RUST_BACKTRACE").is_ok() {
            format!("{:?}", Backtrace::new())
        } else {
            String::new()
        }
    }

    /// Report incompatible dimensions.
    pub fn dimensions<D>(name: &str, dims: D) -> Self
    where
        D: Debug,
    {
        Error::Dimensions {
            name: name.to_string(),
            dimensions: format!("{:?}", dims),
            trace: Self::backtrace(),
        }
    }

    /// Report incorrect reduced dimensions.
    pub fn reduced_dimensions<D>(name: &str, dims: D, rdims: D) -> Self
    where
        D: Debug,
    {
        Error::ReducedDimensions {
            name: name.to_string(),
            dimensions: format!("{:?}", dims),
            reduced_dimensions: format!("{:?}", rdims),
            trace: Self::backtrace(),
        }
    }

    /// Report incompatible lengths.
    pub fn lengths<L>(name: &str, lengths: L) -> Self
    where
        L: Debug,
    {
        Error::Lengths {
            name: name.to_string(),
            lengths: format!("{:?}", lengths),
            trace: Self::backtrace(),
        }
    }

    /// Report an empty input.
    pub fn empty(name: &str) -> Self {
        Error::Empty {
            name: name.to_string(),
            trace: Self::backtrace(),
        }
    }

    /// Report a missing id.
    pub fn missing_id(name: &str) -> Self {
        Error::MissingId {
            name: name.to_string(),
            trace: Self::backtrace(),
        }
    }

    /// Report a missing gradient.
    pub fn missing_gradient(name: &str) -> Self {
        Error::MissingGradient {
            name: name.to_string(),
            trace: Self::backtrace(),
        }
    }

    /// Report a missing node.
    pub fn missing_node(name: &str) -> Self {
        Error::MissingNode {
            name: name.to_string(),
            trace: Self::backtrace(),
        }
    }
}

/// Check that all the given dimensions are equal.
pub fn check_equal_dimensions<D>(name: &str, dims: &[&D]) -> Result<D>
where
    D: PartialEq + Debug + Clone,
{
    let mut it = dims.iter();
    if let Some(first) = it.next() {
        if it.all(|x| x == first) {
            Ok((**first).clone())
        } else {
            Err(Error::dimensions(name, dims))
        }
    } else {
        Err(Error::empty(name))
    }
}

/// Check that all the given lengths are equal.
pub fn check_equal_lengths(name: &str, lengths: &[usize]) -> Result<usize> {
    let mut it = lengths.iter();
    if let Some(first) = it.next() {
        if it.all(|x| x == first) {
            Ok(*first)
        } else {
            Err(Error::lengths(name, lengths))
        }
    } else {
        Err(Error::empty(name))
    }
}

#[cfg(feature = "arrayfire")]
pub mod af {
    use super::*;
    use arrayfire as af;

    /// Check that all the dimensions can be reduced as given.
    pub fn check_reduced_dimensions(
        name: &str,
        dims: af::Dim4,
        rdims: af::Dim4,
    ) -> Result<af::Dim4> {
        for i in 0..4 {
            if rdims[i] == dims[i] {
                continue;
            }
            if rdims[i] != 1 {
                return Err(Error::reduced_dimensions(name, dims, rdims));
            }
        }
        Ok(rdims)
    }
}
