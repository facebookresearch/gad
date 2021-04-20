// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{core, graph, net, store, Check, Eval, Graph1, GraphN};
use arrayfire as af;

/// Generic trait for an algebra implementing all known operations over `af::Array<T>` (and `T`) for a
/// given float type `T`.
pub trait AfAlgebra<T>:
    net::HasGradientReader<GradientReader = <Self as AfAlgebra<T>>::GradientReader>
    + core::CoreAlgebra<af::Array<T>, Value = <Self as AfAlgebra<T>>::Value>
    + core::CoreAlgebra<T, Value = <Self as AfAlgebra<T>>::Scalar>
    + crate::matrix::MatrixAlgebra<<Self as AfAlgebra<T>>::Value>
    + crate::array::ArrayAlgebra<
        <Self as AfAlgebra<T>>::Value,
        Scalar = <Self as AfAlgebra<T>>::Scalar,
    > + crate::analytic::AnalyticAlgebra<<Self as AfAlgebra<T>>::Value>
    + crate::analytic::AnalyticAlgebra<<Self as AfAlgebra<T>>::Scalar>
    + crate::arith::ArithAlgebra<<Self as AfAlgebra<T>>::Value>
    + crate::arith::ArithAlgebra<<Self as AfAlgebra<T>>::Scalar>
    + crate::const_arith::ConstArithAlgebra<<Self as AfAlgebra<T>>::Value, T>
    + crate::const_arith::ConstArithAlgebra<<Self as AfAlgebra<T>>::Scalar, T>
    + crate::const_arith::ConstArithAlgebra<<Self as AfAlgebra<T>>::Value, i16>
    + crate::const_arith::ConstArithAlgebra<<Self as AfAlgebra<T>>::Scalar, i16>
    + crate::compare::CompareAlgebra<<Self as AfAlgebra<T>>::Value>
    + crate::compare::CompareAlgebra<<Self as AfAlgebra<T>>::Scalar>
    + crate::array_compare::ArrayCompareAlgebra<<Self as AfAlgebra<T>>::Value>
where
    T: Float,
{
    type Scalar;
    type Value: net::HasGradientId;
    type GradientReader: store::GradientReader<
        <<Self as AfAlgebra<T>>::Value as net::HasGradientId>::GradientId,
        af::Array<T>,
    >;
}

impl<T: Float> AfAlgebra<T> for Eval {
    type Scalar = T;
    type Value = af::Array<T>;
    type GradientReader = store::EmptyGradientMap;
}

impl<T: Float> AfAlgebra<T> for Check {
    type Scalar = ();
    type Value = af::Dim4;
    type GradientReader = store::EmptyGradientMap;
}

impl<T: Float> AfAlgebra<T> for Graph1 {
    type Scalar = graph::Value<T>;
    type Value = graph::Value<af::Array<T>>;
    type GradientReader = store::GenericGradientMap1;
}

impl<T: Float> AfAlgebra<T> for GraphN {
    type Scalar = graph::Value<T>;
    type Value = graph::Value<af::Array<T>>;
    type GradientReader = store::GenericGradientMapN;
}

/// All supported float types.
pub trait Float:
    crate::Number
    + Default
    + PartialOrd
    + num::Float
    + From<i16>
    + num::pow::Pow<i16, Output = Self>
    + num::pow::Pow<Self, Output = Self>
    + af::HasAfEnum<
        InType = Self,
        AggregateOutType = Self,
        ProductOutType = Self,
        UnaryOutType = Self,
        AbsOutType = Self,
    > + af::ImplicitPromote<Self, Output = Self>
    + af::ConstGenerator<OutType = Self>
    + af::Convertable<OutType = Self>
    + af::FloatingPoint
    + for<'a> std::ops::Div<&'a af::Array<Self>, Output = af::Array<Self>>
{
}

impl Float for f32 {}
impl Float for f64 {}

/// An AfAlgebra for all supported floats.
pub trait FullAlgebra:
    AfAlgebra<f32, GradientReader = <Self as FullAlgebra>::GradientReader>
    + AfAlgebra<f64, GradientReader = <Self as FullAlgebra>::GradientReader>
{
    type GradientReader;
}

impl FullAlgebra for Eval {
    type GradientReader = store::EmptyGradientMap;
}

impl FullAlgebra for Check {
    type GradientReader = store::EmptyGradientMap;
}

impl FullAlgebra for Graph1 {
    type GradientReader = store::GenericGradientMap1;
}

impl FullAlgebra for GraphN {
    type GradientReader = store::GenericGradientMapN;
}

/// Convenient functions used for testing.
pub mod testing {
    use super::*;
    use crate::array::ArrayAlgebra;

    /// Estimate gradient along the given direction.
    #[allow(clippy::suspicious_operation_groupings)]
    pub fn estimate_gradient<T, F>(
        input: &af::Array<T>,
        direction: &af::Array<T>,
        epsilon: T,
        f: F,
    ) -> af::Array<T>
    where
        T: Float + std::fmt::Display,
        F: Fn(&af::Array<T>) -> af::Array<T>,
    {
        let mut v = vec![T::zero(); input.elements()];
        input.host(&mut v);

        let mut gradient = vec![T::zero(); input.elements()];
        for i in 0..input.elements() {
            let x = v[i];

            v[i] = x + epsilon;
            let out = f(&af::Array::new(&v, input.dims()));
            let y2 = Eval::default().dot(&out, direction).unwrap();

            v[i] = x - epsilon;
            let out = f(&af::Array::new(&v, input.dims()));
            let y1 = Eval::default().dot(&out, direction).unwrap();

            gradient[i] = (y2 - y1) / (epsilon + epsilon);
            v[i] = x;
        }

        af::Array::new(&gradient, input.dims())
    }

    /// Assert that the two arrays are close for the L-infinity norm.
    pub fn assert_almost_all_equal<T>(v1: &af::Array<T>, v2: &af::Array<T>, precision: T)
    where
        T: af::HasAfEnum<AbsOutType = T, InType = T, BaseType = T>
            + af::ImplicitPromote<T, Output = T>
            + af::Fromf64
            + std::cmp::PartialOrd,
    {
        assert_eq!(v1.dims(), v2.dims());
        let d = af::max_all(&af::abs(&(v1 - v2))).0;
        assert!(d < precision);
    }
}
