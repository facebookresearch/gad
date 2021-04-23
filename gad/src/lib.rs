// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Generic Automatic Differentiation (GAD)
//!
//! This library provides automatic differentiation by backward propagation (aka.
//! "autograd") in Rust. It was designed to allow first-class user extensions (e.g. with
//! new array types or new operators) and to support multiples modes of execution with
//! minimal overhead.
//!
//! The following modes of execution are currently supported for all library-defined operators:
//! * first-order differentiation,
//! * higher-order differentiation,
//! * forward-only evaluation, and
//! * dimension checking.
//!
//! ## Design Principles
//!
//! The core of this library consists of a tape-based implementation of [automatic
//! differentiation in reverse
//! mode](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation).
//!
//! We have chosen to prioritize idiomatic Rust in order to make this
//! library as re-usable as possible:
//!
//! * The core differentiation algorithm does not use unsafe Rust features or interior
//! mutability (e.g. `RefCell`). All differentiable expressions explicitly mutate a tape
//! when they are constructed. (Below, the tape variable is noted `graph` or `g`.)
//!
//! * Fallible operations never panic and always return a `Result` type. For instance, the
//! sum of two arrays `x` and `y` may be written `g.add(&x, &y)?`.
//!
//! * All structures and values implement `Send` and `Sync` to support concurrent programming.
//!
//! * Generic programming is encouraged so that user formulas can be interpreted in
//! different modes of execution (forward evaluation, dimension checking, etc) with
//! minimal overhead. (See the section below for a code example.)
//!
//! While this library is primarily motivated by machine learning applications, it is
//! meant to cover other use cases of automatic differentiation in reverse mode. In the
//! sections below, we show how a user may define new operators and add new modes of
//! execution while retaining automatic differentiability.
//!
//! ## Limitations
//!
//! * Currently, the usual syntax of operators `+`, `-`, `*`, etc is not available for
//! differentiable values. All operations are method calls of the form `g.op(x1, .. xN)`
//! (or typically `g.op(x1, .. xN)?` for fallible operations).
//!
//! * Because of a current [limitation](https://github.com/rust-lang/rust/issues/49434) of
//! the Rust borrow checker, expressions cannot be nested: `g.add(&x, &g.mul(&y, &z)?)?` must
//! be written `let v = g.mul(&y, &z)?; g.add(&x, &v)?`.
//!
//! We believe that this state of affairs could be improved in the future using Rust
//! macros. Alternatively, future extensions of the library could define a new category of
//! differentiable values that contain an implicit `RefCell` reference to a common tape
//! and provide (implicitly fallible, thread unsafe) operator traits for these values.
//!
//! ## Quick Start
//!
//! To compute gradients, we first build an expression using operations provided by a
//! fresh tape `g` of type `Graph1`. Successive algebraic operations modify the internal
//! state of `g` to track all relevant computations and enables future backward
//! propagation passes.
//!
//! We then call `g.evaluate_gradients(..)` to run a backward propagation algorithm from the
//! desired starting point and using an initial gradient value `direction`.
//!
//! Unless a one-time optimized variant `g.evaluate_gradients_once(..)` is used, backward
//! propagation with `g.evaluate_gradients(..)` does not modify `g`. This allows
//! successive (or concurrent) backward propagations to be run from different starting
//! points or with different gradient values.
//!
//! ```
//! # use gad::prelude::*;
//! # fn main() -> Result<()> {
//! // A new tape supporting first-order differentials (aka gradients)
//! let mut g = Graph1::new();
//! // Compute forward values.
//! let a = g.variable(1f32);
//! let b = g.variable(2f32);
//! let c = g.mul(&a, &b)?;
//! // Compute the derivatives of `c` relative to `a` and `b`
//! let gradients = g.evaluate_gradients(c.gid()?, 1f32)?;
//! // Read the `dc/da` component.
//! assert_eq!(*gradients.get(a.gid()?).unwrap(), 2.0);
//! # Ok(())
//! # }
//! ```
//!
//! Because `Graph1`, the type of `g`, provides algebraic operations as methods, below we
//! refer to such a type as an "algebra". GAD uses particular Rust traits to represent the
//! set of operations supported by a given algebra.
//!
//! ## Computations with Arrayfire
//!
//! The default array operations of the library are currently based on
//! [Arrayfire](https://crates.io/crates/arrayfire), a portable array library supporting GPUs
//! and JIT-compilation.
//! ```
//! # #[cfg(feature = "arrayfire")] {
//! # use gad::prelude::*;
//! use arrayfire as af;
//! # fn main() -> Result<()> {
//! // A new tape supporting first-order differentials (aka gradients)
//! let mut g = Graph1::new();
//! // Compute forward values using Arrayfire arrays
//! let dims = af::Dim4::new(&[4, 3, 1, 1]);
//! let a = g.variable(af::randu::<f32>(dims));
//! let b = g.variable(af::randu::<f32>(dims));
//! let c = g.mul(&a, &b)?;
//! // Compute gradient of c
//! let direction = af::constant(1f32, dims);
//! let gradients = g.evaluate_gradients_once(c.gid()?, direction)?;
//! # Ok(())
//! # }
//! # }
//! ```
//!
//! ## Using Generics for Forward Evaluation and Fast Dimension Checking
//!
//! The algebra `Graph1` used in the examples above is one choice amongst several
//! "default" algebras offered by the library:
//!
//! * We also provide a special algebra `Eval` for forward evaluation, that is, running
//! only primitive operations and dimension checks (no tape, no gradients);
//!
//! * Similarly, using the algebra `Check` will check dimensions without
//! evaluating or allocating any array data;
//!
//! * Finally, differentiation is obtained using `Graph1` for first-order differentials,
//! and `GraphN` for higher-order differentials.
//!
//! Users are encouraged to program formulas in a generic way so that any of the default
//! algebras can be chosen.
//!
//! The following example illustrates such a programming style in the case of array
//! operations:
//! ```
//! # #[cfg(feature = "arrayfire")] {
//! # use gad::prelude::*;
//! use arrayfire as af;
//!
//! fn get_value<A>(g: &mut A) -> Result<<A as AfAlgebra<f32>>::Value>
//! where A : AfAlgebra<f32>
//! {
//!     let dims = af::Dim4::new(&[4, 3, 1, 1]);
//!     let a = g.variable(af::randu::<f32>(dims));
//!     let b = g.variable(af::randu::<f32>(dims));
//!     g.mul(&a, &b)
//! }
//!
//! # fn main() -> Result<()> {
//! // Direct evaluation. The result type is a primitive (non-differentiable) value.
//! let mut g = Eval::default();
//! let c : af::Array<f32> = get_value(&mut g)?;
//!
//! // Fast dimension-checking. The result type is a dimension.
//! let mut g = Check::default();
//! let d : af::Dim4 = get_value(&mut g)?;
//! assert_eq!(c.dims(), d);
//! # Ok(())
//! # }
//! # }
//! ```
//!
//! ## Higher-Order Differentials
//!
//! Higher-order differentials are computed using the algebra `GraphN`. In this case, gradients
//! are values whose computations is also tracked.
//!
//! ```
//! # use gad::prelude::*;
//! # fn main() -> Result<()> {
//! // A new tape supporting differentials of any order.
//! let mut g = GraphN::new();
//!
//! // Compute forward values using floats.
//! let x = g.variable(1.0f32);
//! let y = g.variable(0.4f32);
//! // z = x * y^2
//! let z = {
//!     let h = g.mul(&x, &y)?;
//!     g.mul(&h, &y)?
//! };
//! // Use short names for gradient ids.
//! let (x, y, z) = (x.gid()?, y.gid()?, z.gid()?);
//!
//! // Compute gradient.
//! let dz = g.constant(1f32);
//! let dz_d = g.compute_gradients(z, dz)?;
//! let dz_dx = dz_d.get(x).unwrap();
//!
//! // Compute some 2nd-order differentials.
//! let ddz = g.constant(1f32);
//! let ddz_dxd = g.compute_gradients(dz_dx.gid()?, ddz)?;
//! let ddz_dxdy = ddz_dxd.get(y).unwrap().data();
//! assert_eq!(*ddz_dxdy, 0.8); // 2y
//!
//! // Compute some 3rd-order differentials.
//! let dddz = g.constant(1f32);
//! let dddz_dxdyd = g.compute_gradients(ddz_dxd.get(y).unwrap().gid()?, dddz)?;
//! let dddz_dxdydy = dddz_dxdyd.get(y).unwrap().data();
//! assert_eq!(*dddz_dxdydy, 2.0);
//! # Ok(())
//! # }
//! ```
//!
//! ## Extending Automatic Differentiation
//!
//! ### Operations and algebras
//!
//! The default algebras `Eval`, `Check`, `Graph1`, `GraphN` are meant to provide
//! interchangeable sets of operations in each of the default modes of operation
//! (respectively, evaluation, dimension-checking, first-order differentiation, and
//! higher-order differentiation).
//!
//! Default operations are grouped into several traits named `*Algebra` and implemented by
//! each of the four default algebras above.
//!
//! * The special trait `CoreAlgebra<Data>` defines the mapping from underlying data (e.g.
//! array) to differentiable values. In particular, the method `fn variable(&mut self, data:
//! &Data) -> Self::Value` creates differentiable variables `x` whose gradient value can be
//! referred to later by an id written `x.gid()?` (assuming the algebra is `Graph1` or
//! `GraphN`).
//!
//! * Other traits are parameterized over one or several value types. E.g.
//!   `ArithAlgebra<Value>` provides pointwise negation, multiplication, subtraction, etc
//!   over `Value`.
//!
//! The motivation for using several `*Algebra` traits is twofold:
//!
//! * Users may define their own operations (see next paragraph).
//!
//! * Certain operations are more broadly applicable than others.
//!
//! The following example illustrates gradient computations over integers:
//! ```
//! # use gad::prelude::*;
//! # fn main() -> Result<()> {
//! let mut g = Graph1::new();
//! let a = g.variable(1i32);
//! let b = g.variable(2i32);
//! let c = g.sub(&a, &b)?;
//! assert_eq!(*c.data(), -1);
//! let gradients = g.evaluate_gradients_once(c.gid()?, 1)?;
//! assert_eq!(*gradients.get(a.gid()?).unwrap(), 1);
//! assert_eq!(*gradients.get(b.gid()?).unwrap(), -1);
//! # Ok(())
//! # }
//! ```
//!
//! ### User-defined operations
//!
//! Users may define new differentiable operations by defining their own `*Algebra` trait
//! and providing implementations to the default algebras `Eval`, `Check`, `Graph1`,
//! `GraphN`.
//!
//! In the following example, we define a new operation `square` over integers and
//! af-arrays and add support for first-order differentials:
//! ```
//! # use gad::prelude::*;
//! # #[cfg(feature = "arrayfire")]
//! use arrayfire as af;
//!
//! pub trait UserAlgebra<Value> {
//!     fn square(&mut self, v: &Value) -> Result<Value>;
//! }
//!
//! impl UserAlgebra<i32> for Eval
//! {
//!     fn square(&mut self, v: &i32) -> Result<i32> { Ok(v * v) }
//! }
//!
//! # #[cfg(feature = "arrayfire")] {
//! impl<T> UserAlgebra<af::Array<T>> for Eval
//! where
//!     T: af::HasAfEnum + af::ImplicitPromote<T, Output = T>
//! {
//!     fn square(&mut self, v: &af::Array<T>) -> Result<af::Array<T>> { Ok(v * v) }
//! }
//! # }
//!
//! impl<D> UserAlgebra<Value<D>> for Graph1
//! where
//!     Eval: CoreAlgebra<D, Value = D>
//!         + UserAlgebra<D>
//!         + ArithAlgebra<D>
//!         + LinkedAlgebra<Value<D>, D>,
//!     D: HasDims + Clone + 'static + Send + Sync,
//!     D::Dims: PartialEq + std::fmt::Debug + Clone + 'static + Send + Sync,
//! {
//!     fn square(&mut self, v: &Value<D>) -> Result<Value<D>> {
//!         let result = self.eval().square(v.data())?;
//!         let value = self.make_node(result, vec![v.input()], {
//!             let v = v.clone();
//!             move |graph, store, gradient| {
//!                 if let Some(id) = v.id() {
//!                     let c = graph.link(&v);
//!                     let grad1 = graph.mul(&gradient, c)?;
//!                     let grad2 = graph.mul(c, &gradient)?;
//!                     let grad = graph.add(&grad1, &grad2)?;
//!                     store.add_gradient(graph, id, &grad)?;
//!                 }
//!                 Ok(())
//!             }
//!         });
//!         Ok(value)
//!     }
//! }
//!
//! fn main() -> Result<()> {
//!   let mut g = Graph1::new();
//!   let a = g.variable(3i32);
//!   let b = g.square(&a)?;
//!   assert_eq!(*b.data(), 9);
//!   let gradients = g.evaluate_gradients_once(b.gid()?, 1)?;
//!   assert_eq!(*gradients.get(a.gid()?).unwrap(), 6);
//!   Ok(())
//! }
//! ```
//!
//! The implementation for `GraphN` would be identical to `Graph1`. We have omitted
//! dimension-checking for simplicity. We refer readers to the test files of the library
//! for a more complete example.
//!
//! ### User-defined algebras
//!
//! Users may define new "evaluation" algebras (similar to `Eval`) by implementing a
//! subset of operation traits that includes `CoreAlgebra<Data, Value=Data>` for each
//! supported `Data` types.
//!
//! An evaluation-only algebra can be turned into algebras supporting differentiation
//! (similar to `Graph1` and `GraphN`) using the `Graph` construction provided by the
//! library.
//!
//! The following example illustrates how to define a new evaluation algebra `SymEval`
//! then deduce its counterpart `SymGraph1`:
//! ```
//! # use gad::prelude::*;
//! # use std::sync::Arc;
//! /// A custom algebra for forward-only symbolic evaluation.
//! #[derive(Clone, Default)]
//! struct SymEval;
//!
//! /// Symbolic expressions of type T.
//! #[derive(Debug, PartialEq)]
//! enum Exp_<T> {
//! #    One,
//! #    Zero,
//!     Num(T),
//!     Neg(Exp<T>),
//!     Add(Exp<T>, Exp<T>),
//!     Mul(Exp<T>, Exp<T>),
//! }
//!
//! type Exp<T> = Arc<Exp_<T>>;
//!
//! impl<T> CoreAlgebra<Exp<T>> for SymEval {
//!     type Value = Exp<T>;
//!     fn variable(&mut self, data: Exp<T>) -> Self::Value {
//!         data
//!     }
//!     fn constant(&mut self, data: Exp<T>) -> Self::Value {
//!         data
//!     }
//!     fn add(&mut self, v1: &Self::Value, v2: &Self::Value) -> Result<Self::Value> {
//!         Ok(Arc::new(Exp_::Add(v1.clone(), v2.clone())))
//!     }
//! }
//!
//! impl<T> ArithAlgebra<Exp<T>> for SymEval {
//! #    fn zeros(&mut self, _v: &Exp<T>) -> Exp<T> {
//! #        Arc::new(Exp_::Zero)
//! #    }
//! #    fn ones(&mut self, _v: &Exp<T>) -> Exp<T> {
//! #        Arc::new(Exp_::One)
//! #    }
//!     fn neg(&mut self, v: &Exp<T>) -> Exp<T> {
//!         Arc::new(Exp_::Neg(v.clone()))
//!     }
//!     fn sub(&mut self, v1: &Exp<T>, v2: &Exp<T>) -> Result<Exp<T>> {
//!         let v2 = self.neg(v2);
//!         Ok(Arc::new(Exp_::Add(v1.clone(), v2)))
//!     }
//!     fn mul(&mut self, v1: &Exp<T>, v2: &Exp<T>) -> Result<Exp<T>> {
//!         Ok(Arc::new(Exp_::Mul(v1.clone(), v2.clone())))
//!     }
//! }
//!
//! // No dimension checks.
//! impl<T> HasDims for Exp_<T> {
//!     type Dims = ();
//!     fn dims(&self) {}
//! }
//!
//! impl<T: std::fmt::Display> std::fmt::Display for Exp_<T> {
//!     // ...
//! #   fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//! #       use Exp_::*;
//! #       match self {
//! #           Zero => write!(f, "0"),
//! #           One => write!(f, "1"),
//! #           Num(x) => write!(f, "{}", x),
//! #           Neg(e) => write!(f, "(-{})", *e),
//! #           Add(e1, e2) => write!(f, "({}+{})", *e1, *e2),
//! #           Mul(e1, e2) => write!(f, "{}{}", *e1, *e2),
//! #       }
//! #   }
//! }
//!
//! /// Apply `graph` module to Derive an algebra supporting gradients.
//! type SymGraph1 = Graph<Config1<SymEval>>;
//!
//! fn main() -> Result<()> {
//!     let mut g = SymGraph1::new();
//!     let a = g.variable(Arc::new(Exp_::Num("a")));
//!     let b = g.variable(Arc::new(Exp_::Num("b")));
//!     let c = g.mul(&a, &b)?;
//!     let d = g.mul(&a, &c)?;
//!     assert_eq!(format!("{}", d.data()), "aab");
//!     let gradients = g.evaluate_gradients_once(d.gid()?, Arc::new(Exp_::Num("1")))?;
//!     assert_eq!(format!("{}", gradients.get(a.gid()?).unwrap()), "(1ab+a1b)");
//!     assert_eq!(format!("{}", gradients.get(b.gid()?).unwrap()), "aa1");
//!     Ok(())
//! }
//! ```

/// Convenient prelude.
/// For testing and external use only.
pub mod prelude {
    pub use crate::{
        analytic::AnalyticAlgebra,
        arith::ArithAlgebra,
        array::ArrayAlgebra,
        array_compare::ArrayCompareAlgebra,
        compare::CompareAlgebra,
        const_arith::ConstArithAlgebra,
        core::{CoreAlgebra, HasDims},
        error::{check_equal_dimensions, Error, Result},
        func_name,
        graph::{Config1, ConfigN, Graph, Value},
        linked::LinkedAlgebra,
        matrix::{MatProp, MatrixAlgebra},
        net::{
            CheckNet as _, ConstantData, EvalNet as _, HasGradientId, HasGradientReader, InputData,
            Net, WeightData, WeightOps,
        },
        net_ext::{DiffNet as _, SingleOutputNet as _},
        store::{GradientId, GradientReader, GradientStore},
        Check, Eval, Graph1, GraphN, Number,
    };
    pub use thiserror::Error as _;

    #[cfg(feature = "arrayfire")]
    pub use crate::arrayfire::{testing, AfAlgebra, Float, FullAlgebra};
}

/// Error and result types.
#[macro_use]
pub mod error;

/// Provide algebras supporting higher-order, tape-based auto-differentiation.
pub mod graph;

/// Core operations.
pub mod core;

/// Pointwise analytic functions (cos, sin, log, exp, pow, sqrt, ..)
pub mod analytic;

/// Pointwise arithmetic operations.
pub mod arith;

/// Pointwise arithmetic operations with a constant value.
pub mod const_arith;

/// Pointwise comparison operations.
pub mod compare;

/// Operation to propagate gradients in the case of high-order differentials.
pub mod linked;

/// Gradient storage for the `graph` module.
pub mod store;

/// Array operations.
pub mod array;

/// Array operations with comparisons.
pub mod array_compare;

/// Operations on matrix.
pub mod matrix;

/// Neural networks.
pub mod net;

/// Network extensions.
pub mod net_ext;

/// Additional definitions for Arrayfire.
#[cfg(feature = "arrayfire")]
pub mod arrayfire;

/// The default algebra that only checks dimensions.
#[derive(Clone, Default)]
pub struct Check;

/// The default algebra that only computes forward values.
#[derive(Clone, Default)]
pub struct Eval {
    check: Check,
}

/// The default algebra that allows computing first-order differentials (aka gradients).
pub type Graph1 = graph::Graph<graph::Config1<Eval>>;

/// The default algebra that allows computing higher-order differentials.
pub type GraphN = graph::Graph<graph::ConfigN<Eval>>;

impl Eval {
    /// Access the underlying default "Check" algebra.
    #[inline]
    pub fn check(&mut self) -> &mut Check {
        &mut self.check
    }
}

mod private {
    /// "Sealed" trait used to avoid conflicting trait implementations.
    pub trait Reserved {}

    impl Reserved for i8 {}
    impl Reserved for i16 {}
    impl Reserved for i32 {}
    impl Reserved for i64 {}
    impl Reserved for f32 {}
    impl Reserved for f64 {}
    impl Reserved for num::complex::Complex<f32> {}
    impl Reserved for num::complex::Complex<f64> {}
    impl Reserved for num::Rational32 {}
    impl Reserved for num::Rational64 {}
    impl Reserved for half::f16 {}
    impl Reserved for half::bf16 {}
}

/// Supported numbers for default algebras.
pub trait Number:
    private::Reserved
    + num::Num
    + std::ops::Neg<Output = Self>
    + std::fmt::Debug
    + serde::Serialize
    + serde::de::DeserializeOwned
    + 'static
    + Clone
    + Copy
    + Send
    + Sync
{
}
impl Number for i8 {}
impl Number for i16 {}
impl Number for i32 {}
impl Number for i64 {}
impl Number for f32 {}
impl Number for f64 {}
impl Number for num::complex::Complex<f32> {}
impl Number for num::complex::Complex<f64> {}
impl Number for num::Rational32 {}
impl Number for num::Rational64 {}
impl Number for half::f16 {}
impl Number for half::bf16 {}

#[cfg(test)]
mod testing {
    use super::*;
    trait Test: Sync + Send + Clone + Default {}
    impl Test for Graph1 {}
    impl Test for GraphN {}
}
