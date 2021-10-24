// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    arith::ArithAlgebra,
    const_arith::ConstArithAlgebra,
    core::{CoreAlgebra, HasDims},
    error::Result,
    graph::{Config1, ConfigN, Graph, Value},
    linked::LinkedAlgebra,
    store::GradientStore,
    Check, Eval, Number,
};

/// Element-wise analytic functions.
pub trait AnalyticAlgebra<Value> {
    /// Element-wise natural logarithm `exp(x)`.
    fn exp(&mut self, v: &Value) -> Value;

    /// Element-wise natural logarithm `log(x)`.
    fn log(&mut self, v: &Value) -> Value;

    /// Element-wise natural logarithm shifted by one `log(1 + x)`.
    fn log1p(&mut self, v: &Value) -> Value;

    /// Element-wise sinus `sin(x)`.
    fn sin(&mut self, v: &Value) -> Value;

    /// Element-wise cosinus `cos(x)`.
    fn cos(&mut self, v: &Value) -> Value;

    /// Element-wise hyperbolic tangent `tanh(x)`.
    fn tanh(&mut self, v: &Value) -> Value;

    /// Element-wise sigmoid `1 / (1 + exp(-x))`.
    fn sigmoid(&mut self, v: &Value) -> Value;

    /// Element-wise reciprocal `1/x`.
    fn reciprocal(&mut self, v: &Value) -> Value;

    /// Element-wise square root `sqrt(x)`.
    fn sqrt(&mut self, v: &Value) -> Value;

    /// Element-wise division `x / y`.
    fn div(&mut self, v0: &Value, v1: &Value) -> Result<Value>;

    /// Element-wise power `x ^ p`.
    fn pow(&mut self, v: &Value, p: &Value) -> Result<Value>
    where
        Self: ArithAlgebra<Value>,
    {
        let l = self.log(v);
        let e = self.mul(p, &l)?;
        Ok(self.exp(&e))
    }
}

#[cfg(feature = "arrayfire")]
mod af_arith {
    use super::*;
    use crate::error::check_equal_dimensions;
    use arrayfire as af;

    impl<T> AnalyticAlgebra<af::Array<T>> for Eval
    where
        Self: CoreAlgebra<af::Array<T>, Value = af::Array<T>>,
        T: af::HasAfEnum<UnaryOutType = T, AbsOutType = T>
            + af::ImplicitPromote<T, Output = T>
            + af::ConstGenerator<OutType = T>
            + num::Zero
            + num::One
            + for<'a> std::ops::Div<&'a af::Array<T>, Output = af::Array<T>>,
    {
        #[inline]
        fn exp(&mut self, v: &af::Array<T>) -> af::Array<T> {
            af::exp(v)
        }

        #[inline]
        fn log(&mut self, v: &af::Array<T>) -> af::Array<T> {
            af::log(v)
        }

        #[inline]
        fn log1p(&mut self, v: &af::Array<T>) -> af::Array<T> {
            af::log1p(v)
        }

        #[inline]
        fn sin(&mut self, v: &af::Array<T>) -> af::Array<T> {
            af::sin(v)
        }

        #[inline]
        fn cos(&mut self, v: &af::Array<T>) -> af::Array<T> {
            af::cos(v)
        }

        #[inline]
        fn tanh(&mut self, v: &af::Array<T>) -> af::Array<T> {
            af::tanh(v)
        }

        #[inline]
        fn sigmoid(&mut self, v: &af::Array<T>) -> af::Array<T> {
            af::sigmoid(v)
        }

        #[inline]
        fn reciprocal(&mut self, v: &af::Array<T>) -> af::Array<T> {
            T::one() / v
        }

        #[inline]
        fn sqrt(&mut self, v: &af::Array<T>) -> af::Array<T> {
            af::sqrt(v)
        }

        fn div(&mut self, v0: &af::Array<T>, v1: &af::Array<T>) -> Result<af::Array<T>> {
            self.check.div(&v0.dims(), &v1.dims())?;
            Ok(af::div(v0, v1, false))
        }

        fn pow(&mut self, v0: &af::Array<T>, v1: &af::Array<T>) -> Result<af::Array<T>> {
            self.check.pow(&v0.dims(), &v1.dims())?;
            Ok(af::pow(v0, v1, false))
        }
    }

    impl AnalyticAlgebra<af::Dim4> for Check {
        #[inline]
        fn exp(&mut self, v: &af::Dim4) -> af::Dim4 {
            *v
        }

        #[inline]
        fn log(&mut self, v: &af::Dim4) -> af::Dim4 {
            *v
        }

        #[inline]
        fn log1p(&mut self, v: &af::Dim4) -> af::Dim4 {
            *v
        }

        #[inline]
        fn sin(&mut self, v: &af::Dim4) -> af::Dim4 {
            *v
        }

        #[inline]
        fn cos(&mut self, v: &af::Dim4) -> af::Dim4 {
            *v
        }

        #[inline]
        fn tanh(&mut self, v: &af::Dim4) -> af::Dim4 {
            *v
        }

        #[inline]
        fn sigmoid(&mut self, v: &af::Dim4) -> af::Dim4 {
            *v
        }

        #[inline]
        fn reciprocal(&mut self, v: &af::Dim4) -> af::Dim4 {
            *v
        }

        #[inline]
        fn sqrt(&mut self, v: &af::Dim4) -> af::Dim4 {
            *v
        }

        #[inline]
        fn div(&mut self, v0: &af::Dim4, v1: &af::Dim4) -> Result<af::Dim4> {
            check_equal_dimensions(func_name!(), &[v0, v1])
        }

        #[inline]
        fn pow(&mut self, v0: &af::Dim4, v1: &af::Dim4) -> Result<af::Dim4> {
            check_equal_dimensions(func_name!(), &[v0, v1])
        }
    }
}

impl<T> AnalyticAlgebra<T> for Eval
where
    T: Number + num::Float,
{
    #[inline]
    fn exp(&mut self, v: &T) -> T {
        v.exp()
    }

    #[inline]
    fn log(&mut self, v: &T) -> T {
        v.ln()
    }

    #[inline]
    fn log1p(&mut self, v: &T) -> T {
        T::ln(T::one() + *v)
    }

    #[inline]
    fn sin(&mut self, v: &T) -> T {
        v.sin()
    }

    #[inline]
    fn cos(&mut self, v: &T) -> T {
        v.cos()
    }

    #[inline]
    fn tanh(&mut self, v: &T) -> T {
        v.tanh()
    }

    #[inline]
    fn sigmoid(&mut self, v: &T) -> T {
        T::one() / (T::one() + T::exp(-*v))
    }

    #[inline]
    fn reciprocal(&mut self, v: &T) -> T {
        T::one() / *v
    }

    #[inline]
    fn sqrt(&mut self, v: &T) -> T {
        v.sqrt()
    }

    #[inline]
    fn div(&mut self, v0: &T, v1: &T) -> Result<T> {
        Ok(*v0 / *v1)
    }

    #[inline]
    fn pow(&mut self, v0: &T, v1: &T) -> Result<T> {
        Ok(v0.powf(*v1))
    }
}

impl AnalyticAlgebra<()> for Check {
    #[inline]
    fn exp(&mut self, _v: &()) {}

    #[inline]
    fn log(&mut self, _v: &()) {}

    #[inline]
    fn log1p(&mut self, _v: &()) {}

    #[inline]
    fn sin(&mut self, _v: &()) {}

    #[inline]
    fn cos(&mut self, _v: &()) {}

    #[inline]
    fn tanh(&mut self, _v: &()) {}

    #[inline]
    fn sigmoid(&mut self, _v: &()) {}

    #[inline]
    fn reciprocal(&mut self, _v: &()) {}

    #[inline]
    fn sqrt(&mut self, _v: &()) {}

    #[inline]
    fn div(&mut self, _v0: &(), _v1: &()) -> Result<()> {
        Ok(())
    }

    #[inline]
    fn pow(&mut self, _v0: &(), _v1: &()) -> Result<()> {
        Ok(())
    }
}

macro_rules! impl_graph {
    ($config:ident) => {
        impl<D, E, Dims> AnalyticAlgebra<Value<D>> for Graph<$config<E>>
        where
            E: Default
                + Clone
                + CoreAlgebra<D, Value = D>
                + AnalyticAlgebra<D>
                + ArithAlgebra<D>
                + ConstArithAlgebra<D, i8>
                + LinkedAlgebra<Value<D>, D>,
            D: HasDims<Dims = Dims> + Clone + 'static + Send + Sync,
            Dims: PartialEq + std::fmt::Debug + Clone + 'static + Send + Sync,
        {
            fn exp(&mut self, v: &Value<D>) -> Value<D> {
                let result = self.eval().exp(v.data());
                self.make_node(result, vec![v.input()], {
                    let v = v.clone();
                    move |graph, store, gradient| {
                        if let Some(id) = v.id() {
                            let v = graph.link(&v);
                            let k = graph.exp(v);
                            let grad = graph.mul(&gradient, &k)?;
                            store.add_gradient(graph, id, &grad)?;
                        }
                        Ok(())
                    }
                })
            }

            fn log(&mut self, v: &Value<D>) -> Value<D> {
                let result = self.eval().log(v.data());
                self.make_node(result, vec![v.input()], {
                    let v = v.clone();
                    move |graph, store, gradient| {
                        if let Some(id) = v.id() {
                            let v = graph.link(&v);
                            let grad = graph.div(&gradient, &v)?;
                            store.add_gradient(graph, id, &grad)?;
                        }
                        Ok(())
                    }
                })
            }

            fn log1p(&mut self, v: &Value<D>) -> Value<D> {
                let result = self.eval().log1p(v.data());
                self.make_node(result, vec![v.input()], {
                    let v = v.clone();
                    move |graph, store, gradient| {
                        if let Some(id) = v.id() {
                            let v = graph.link(&v);
                            let v1p = graph.addc(v, 1);
                            let grad = graph.div(&gradient, &v1p)?;
                            store.add_gradient(graph, id, &grad)?;
                        }
                        Ok(())
                    }
                })
            }

            fn sin(&mut self, v: &Value<D>) -> Value<D> {
                let result = self.eval().sin(v.data());
                self.make_node(result, vec![v.input()], {
                    let v = v.clone();
                    move |graph, store, gradient| {
                        if let Some(id) = v.id() {
                            let v = graph.link(&v);
                            let k = graph.cos(&v);
                            let grad = graph.mul(&gradient, &k)?;
                            store.add_gradient(graph, id, &grad)?;
                        }
                        Ok(())
                    }
                })
            }

            fn cos(&mut self, v: &Value<D>) -> Value<D> {
                let result = self.eval().cos(v.data());
                self.make_node(result, vec![v.input()], {
                    let v = v.clone();
                    move |graph, store, gradient| {
                        if let Some(id) = v.id() {
                            let v = graph.link(&v);
                            let c = graph.sin(&v);
                            let k = graph.neg(&c);
                            let grad = graph.mul(&gradient, &k)?;
                            store.add_gradient(graph, id, &grad)?;
                        }
                        Ok(())
                    }
                })
            }

            fn tanh(&mut self, v: &Value<D>) -> Value<D> {
                let result = self.eval().tanh(v.data());
                self.make_node(result, vec![v.input()], {
                    let v = v.clone();
                    move |graph, store, gradient| {
                        if let Some(id) = v.id() {
                            let v = graph.link(&v);
                            let t = graph.tanh(&v);
                            let c = graph.mul(&t, &t)?;
                            let c = graph.neg(&c);
                            let k = graph.addc(&c, 1);
                            let grad = graph.mul(&gradient, &k)?;
                            store.add_gradient(graph, id, &grad)?;
                        }
                        Ok(())
                    }
                })
            }

            fn sigmoid(&mut self, v: &Value<D>) -> Value<D> {
                let result = self.eval().sigmoid(v.data());
                self.make_node(result, vec![v.input()], {
                    let v = v.clone();
                    move |graph, store, gradient| {
                        if let Some(id) = v.id() {
                            let v = graph.link(&v);
                            let c = graph.sigmoid(&v);
                            let d = graph.neg(&c);
                            let d = graph.addc(&d, 1);
                            let k = graph.mul(&c, &d)?;
                            let grad = graph.mul(&gradient, &k)?;
                            store.add_gradient(graph, id, &grad)?;
                        }
                        Ok(())
                    }
                })
            }

            fn reciprocal(&mut self, v: &Value<D>) -> Value<D> {
                let result = self.eval().reciprocal(v.data());
                self.make_node(result, vec![v.input()], {
                    let v = v.clone();
                    move |graph, store, gradient| {
                        if let Some(id) = v.id() {
                            let v = graph.link(&v);
                            let c = graph.mul(&v, &v)?;
                            let c = graph.neg(&c);
                            let k = graph.reciprocal(&c);
                            let grad = graph.mul(&gradient, &k)?;
                            store.add_gradient(graph, id, &grad)?;
                        }
                        Ok(())
                    }
                })
            }

            fn sqrt(&mut self, v: &Value<D>) -> Value<D> {
                let result = self.eval().sqrt(v.data());
                self.make_node(result, vec![v.input()], {
                    let v = v.clone();
                    move |graph, store, gradient| {
                        if let Some(id) = v.id() {
                            let v = graph.link(&v);
                            let c = graph.sqrt(&v);
                            let c = graph.mulc(&c, 2);
                            let k = graph.reciprocal(&c);
                            let grad = graph.mul(&gradient, &k)?;
                            store.add_gradient(graph, id, &grad)?;
                        }
                        Ok(())
                    }
                })
            }

            fn div(&mut self, v0: &Value<D>, v1: &Value<D>) -> Result<Value<D>> {
                let result = self.eval().div(v0.data(), v1.data())?;
                let value = self.make_node(result, vec![v0.input(), v1.input()], {
                    let v0 = v0.clone();
                    let v1 = v1.clone();
                    move |graph, store, gradient| {
                        let c1 = graph.link(&v1);
                        let r1 = graph.reciprocal(&c1);
                        let g0 = graph.mul(&gradient, &r1)?;
                        if let Some(id) = v0.id() {
                            store.add_gradient(graph, id, &g0)?;
                        }
                        if let Some(id) = v1.id() {
                            let c0 = graph.link(&v0);
                            let c = graph.mul(&g0, &r1)?;
                            let c = graph.mul(&c, &c0)?;
                            let g1 = graph.neg(&c);
                            store.add_gradient(graph, id, &g1)?;
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
