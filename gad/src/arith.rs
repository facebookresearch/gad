// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    core::{CoreAlgebra, HasDims},
    error::Result,
    graph::{Config1, ConfigN, Graph, Value},
    linked::LinkedAlgebra,
    store::GradientStore,
    Check, Eval, Number,
};

/// Element-wise arithmetic operations.
pub trait ArithAlgebra<Value> {
    /// Element-wise subtraction `v0 - v1`
    fn sub(&mut self, v0: &Value, v1: &Value) -> Result<Value>;

    /// Element-wise multiplication `v0 * v1`
    fn mul(&mut self, v0: &Value, v1: &Value) -> Result<Value>;

    /// Element-wise zero (same dimensions as v).
    fn zeros(&mut self, v: &Value) -> Value;

    /// Element-wise one (same dimensions as v).
    fn ones(&mut self, v: &Value) -> Value;

    /// Element-wise negation `-v`
    fn neg(&mut self, v: &Value) -> Value {
        let z = self.zeros(v);
        self.sub(&z, v).expect("subtracting zero should not fail")
    }
}

#[cfg(feature = "arrayfire")]
mod af_arith {
    use super::*;
    use crate::error::check_equal_dimensions;
    use arrayfire as af;

    impl<T> ArithAlgebra<af::Array<T>> for Eval
    where
        Self: CoreAlgebra<af::Array<T>, Value = af::Array<T>>,
        T: af::HasAfEnum
            + af::ImplicitPromote<T, Output = T>
            + af::ConstGenerator<OutType = T>
            + num::Zero
            + num::One,
    {
        #[inline]
        fn zeros(&mut self, v: &af::Array<T>) -> af::Array<T> {
            af::constant(T::zero(), v.dims())
        }

        #[inline]
        fn ones(&mut self, v: &af::Array<T>) -> af::Array<T> {
            af::constant(T::one(), v.dims())
        }

        #[inline]
        fn neg(&mut self, v: &af::Array<T>) -> af::Array<T> {
            af::constant(T::zero(), v.dims()) - v
        }

        #[inline]
        fn sub(&mut self, v0: &af::Array<T>, v1: &af::Array<T>) -> Result<af::Array<T>> {
            self.check().sub(&v0.dims(), &v1.dims())?;
            Ok(v0 - v1)
        }

        #[inline]
        fn mul(&mut self, v0: &af::Array<T>, v1: &af::Array<T>) -> Result<af::Array<T>> {
            self.check().mul(&v0.dims(), &v1.dims())?;
            Ok(v0 * v1)
        }
    }

    impl ArithAlgebra<af::Dim4> for Check {
        #[inline]
        fn zeros(&mut self, v: &af::Dim4) -> af::Dim4 {
            *v
        }

        #[inline]
        fn ones(&mut self, v: &af::Dim4) -> af::Dim4 {
            *v
        }

        #[inline]
        fn neg(&mut self, v: &af::Dim4) -> af::Dim4 {
            *v
        }

        #[inline]
        fn sub(&mut self, v0: &af::Dim4, v1: &af::Dim4) -> Result<af::Dim4> {
            check_equal_dimensions(func_name!(), &[v0, v1])
        }

        #[inline]
        fn mul(&mut self, v0: &af::Dim4, v1: &af::Dim4) -> Result<af::Dim4> {
            check_equal_dimensions(func_name!(), &[v0, v1])
        }
    }
}

impl<T: Number> ArithAlgebra<T> for Eval {
    #[inline]
    fn zeros(&mut self, _v: &T) -> T {
        T::zero()
    }

    #[inline]
    fn ones(&mut self, _v: &T) -> T {
        T::one()
    }

    #[inline]
    fn neg(&mut self, v: &T) -> T {
        -(*v)
    }

    #[inline]
    fn sub(&mut self, v0: &T, v1: &T) -> Result<T> {
        Ok(*(v0) - *(v1))
    }

    #[inline]
    fn mul(&mut self, v0: &T, v1: &T) -> Result<T> {
        Ok((*v0) * (*v1))
    }
}

impl ArithAlgebra<()> for Check {
    #[inline]
    fn zeros(&mut self, _v: &()) {}

    #[inline]
    fn ones(&mut self, _v: &()) {}

    #[inline]
    fn neg(&mut self, _v: &()) {}

    #[inline]
    fn sub(&mut self, _v0: &(), _v1: &()) -> Result<()> {
        Ok(())
    }

    #[inline]
    fn mul(&mut self, _v0: &(), _v1: &()) -> Result<()> {
        Ok(())
    }
}

macro_rules! impl_graph {
    ($config:ident) => {
        impl<D, E, Dims> ArithAlgebra<Value<D>> for Graph<$config<E>>
        where
            E: Default
                + Clone
                + CoreAlgebra<D, Value = D>
                + ArithAlgebra<D>
                + LinkedAlgebra<Value<D>, D>,
            D: HasDims<Dims = Dims> + Clone + 'static + Send + Sync,
            Dims: PartialEq + std::fmt::Debug + Clone + 'static + Send + Sync,
        {
            fn zeros(&mut self, v: &Value<D>) -> Value<D> {
                let result = self.eval().zeros(v.data());
                self.constant(result)
            }

            fn ones(&mut self, v: &Value<D>) -> Value<D> {
                let result = self.eval().ones(v.data());
                self.constant(result)
            }

            fn neg(&mut self, v: &Value<D>) -> Value<D> {
                let result = self.eval().neg(v.data());
                self.make_node(result, vec![v.input()], {
                    let id = v.id();
                    move |graph, store, gradient| {
                        if let Some(id) = id {
                            let n = graph.neg(&gradient);
                            store.add_gradient(graph, id, &n)?;
                        }
                        Ok(())
                    }
                })
            }

            fn sub(&mut self, v0: &Value<D>, v1: &Value<D>) -> Result<Value<D>> {
                let result = self.eval().sub(v0.data(), v1.data())?;
                let value = self.make_node(result, vec![v0.input(), v1.input()], {
                    let id0 = v0.id();
                    let id1 = v1.id();
                    move |graph, store, gradient| {
                        if let Some(id) = id0 {
                            store.add_gradient(graph, id, &gradient)?;
                        }
                        if let Some(id) = id1 {
                            let n = graph.neg(&gradient);
                            store.add_gradient(graph, id, &n)?;
                        }
                        Ok(())
                    }
                });
                Ok(value)
            }

            fn mul(&mut self, v0: &Value<D>, v1: &Value<D>) -> Result<Value<D>> {
                let result = self.eval().mul(v0.data(), v1.data())?;
                let value = self.make_node(result, vec![v0.input(), v1.input()], {
                    let v0 = v0.clone();
                    let v1 = v1.clone();
                    move |graph, store, gradient| {
                        if let Some(id) = v0.id() {
                            let c1 = graph.link(&v1);
                            let grad = graph.mul(&gradient, c1)?;
                            store.add_gradient(graph, id, &grad)?;
                        }
                        if let Some(id) = v1.id() {
                            let c0 = graph.link(&v0);
                            let grad = graph.mul(c0, &gradient)?;
                            store.add_gradient(graph, id, &grad)?;
                        }
                        Ok(())
                    }
                });
                Ok(value)
            }
        }
    };
}

impl_graph!(Config1);
impl_graph!(ConfigN);
