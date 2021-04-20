// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    arith::ArithAlgebra,
    core::{CoreAlgebra, HasDims},
    graph::{Config1, ConfigN, Graph, Value},
    linked::LinkedAlgebra,
    store::GradientStore,
    Check, Eval, Number,
};

/// Element-wise arithmetic operations with a constant value.
pub trait ConstArithAlgebra<Value, Const> {
    /// Return a value with the same shape as `v` but filled with constant `c`.
    fn setc(&mut self, v: &Value, c: Const) -> Value;

    /// Element-wise addition of a constant `v + c`.
    fn addc(&mut self, v: &Value, c: Const) -> Value;

    /// Element-wise multiplication by a constant `v * c`.
    fn mulc(&mut self, v: &Value, c: Const) -> Value;

    /// Element-wise exponentiation by a constant `v ^ c`.
    fn powc(&mut self, v: &Value, c: Const) -> Value;
}

#[cfg(feature = "arrayfire")]
mod af_arith {
    use super::*;
    use arrayfire as af;

    impl<T, C> ConstArithAlgebra<af::Array<T>, C> for Eval
    where
        Self: CoreAlgebra<af::Array<T>, Value = af::Array<T>>,
        T: af::HasAfEnum
            + af::ImplicitPromote<T, Output = T>
            + af::ConstGenerator<OutType = T>
            + From<C>,
    {
        fn setc(&mut self, v: &af::Array<T>, c: C) -> af::Array<T> {
            af::constant(T::from(c), v.dims())
        }

        fn addc(&mut self, v: &af::Array<T>, c: C) -> af::Array<T> {
            v + af::constant(T::from(c), v.dims())
        }

        fn mulc(&mut self, v: &af::Array<T>, c: C) -> af::Array<T> {
            v * af::constant(T::from(c), v.dims())
        }

        fn powc(&mut self, v: &af::Array<T>, c: C) -> af::Array<T> {
            af::pow(v, &af::constant(T::from(c), v.dims()), false)
        }
    }

    impl<C> ConstArithAlgebra<af::Dim4, C> for Check {
        #[inline]
        fn setc(&mut self, v: &af::Dim4, _c: C) -> af::Dim4 {
            *v
        }

        #[inline]
        fn addc(&mut self, v: &af::Dim4, _c: C) -> af::Dim4 {
            *v
        }

        #[inline]
        fn mulc(&mut self, v: &af::Dim4, _c: C) -> af::Dim4 {
            *v
        }

        #[inline]
        fn powc(&mut self, v: &af::Dim4, _c: C) -> af::Dim4 {
            *v
        }
    }
}

impl<T, C> ConstArithAlgebra<T, C> for Eval
where
    T: Number + From<C> + num::pow::Pow<C, Output = T>,
{
    #[inline]
    fn setc(&mut self, _v: &T, c: C) -> T {
        c.into()
    }

    #[inline]
    fn addc(&mut self, v: &T, c: C) -> T {
        v.add(c.into())
    }

    #[inline]
    fn mulc(&mut self, v: &T, c: C) -> T {
        v.mul(c.into())
    }

    #[inline]
    fn powc(&mut self, v: &T, c: C) -> T {
        v.pow(c)
    }
}

impl<C> ConstArithAlgebra<(), C> for Check {
    #[inline]
    fn setc(&mut self, _v: &(), _c: C) {}

    #[inline]
    fn addc(&mut self, _v: &(), _c: C) {}

    #[inline]
    fn mulc(&mut self, _v: &(), _c: C) {}

    #[inline]
    fn powc(&mut self, _v: &(), _c: C) {}
}

macro_rules! impl_graph {
    ($config:ident) => {
        impl<D, E, Dims, C> ConstArithAlgebra<Value<D>, C> for Graph<$config<E>>
        where
            E: Default
                + Clone
                + CoreAlgebra<D, Value = D>
                + ArithAlgebra<D>
                + ConstArithAlgebra<D, C>
                + LinkedAlgebra<Value<D>, D>,
            C: std::ops::Sub<C, Output = C> + num::One + Clone + 'static + Send + Sync,
            D: HasDims<Dims = Dims> + Clone + 'static + Send + Sync,
            Dims: PartialEq + std::fmt::Debug + Clone + 'static + Send + Sync,
        {
            fn setc(&mut self, v: &Value<D>, c: C) -> Value<D> {
                let result = self.eval().addc(v.data(), c);
                self.constant(result)
            }

            fn addc(&mut self, v: &Value<D>, c: C) -> Value<D> {
                let result = self.eval().addc(v.data(), c);
                self.make_node(result, vec![v.input()], {
                    let id = v.id();
                    move |graph, store, gradient| {
                        if let Some(id) = id {
                            store.add_gradient(graph, id, &gradient)?;
                        }
                        Ok(())
                    }
                })
            }

            fn mulc(&mut self, v: &Value<D>, c: C) -> Value<D> {
                let result = self.eval().mulc(v.data(), c.clone());
                self.make_node(result, vec![v.input()], {
                    let id = v.id();
                    move |graph, store, gradient| {
                        if let Some(id) = id {
                            let grad = graph.mulc(&gradient, c.clone());
                            store.add_gradient(graph, id, &grad)?;
                        }
                        Ok(())
                    }
                })
            }

            fn powc(&mut self, v: &Value<D>, c: C) -> Value<D> {
                let result = self.eval().powc(v.data(), c.clone());
                self.make_node(result, vec![v.input()], {
                    let v = v.clone();
                    move |graph, store, gradient| {
                        if let Some(id) = v.id() {
                            let v = graph.link(&v);
                            let e = graph.powc(&v, c.clone() - C::one());
                            let f = graph.mulc(&e, c.clone());
                            let grad = graph.mul(&f, &gradient)?;
                            store.add_gradient(graph, id, &grad)?;
                        }
                        Ok(())
                    }
                })
            }
        }
    };
}

impl_graph!(Config1);
impl_graph!(ConfigN);
