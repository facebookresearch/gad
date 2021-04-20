// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    arith::ArithAlgebra,
    core::{CoreAlgebra, HasDims},
    error::Result,
    graph::{Config1, ConfigN, Graph, Value},
    linked::LinkedAlgebra,
    store::GradientStore,
    Check, Eval, Number,
};

/// Element-wise comparison operations.
pub trait CompareAlgebra<Value> {
    /// Element-wise minimum `min(v0, v1)`.
    fn min(&mut self, v0: &Value, v1: &Value) -> Result<Value> {
        self.select_argmax(v0, v1, Some(v1), Some(v0))
    }

    /// Element-wise maximum `max(v0, v1)`.
    fn max(&mut self, v0: &Value, v1: &Value) -> Result<Value> {
        self.select_argmax(v0, v1, Some(v0), Some(v1))
    }

    /// Element-wise absolute value `|v|`.
    fn abs(&mut self, v: &Value) -> Value
    where
        Self: ArithAlgebra<Value>,
    {
        let neg_v = self.neg(v);
        self.select_argmax(v, &neg_v, Some(v), Some(&neg_v))
            .expect("max should not fail")
    }

    /// Element-wise sign value `sign(v)`.
    fn sign(&mut self, v: &Value) -> Value
    where
        Self: ArithAlgebra<Value>,
    {
        let neg_v = self.neg(v);
        let one = self.ones(v);
        let minus_one = self.neg(&one);
        self.select_argmax(v, &neg_v, Some(&one), Some(&minus_one))
            .expect("sign should not fail")
    }

    /// Element-wise value `relu(v) = max(v, 0)`.
    fn relu(&mut self, v: &Value) -> Value
    where
        Self: ArithAlgebra<Value>,
    {
        let zero = self.zeros(v);
        self.max(&zero, v).expect("relu should not fail")
    }

    /// Element-wise selection by comparison: `if v0 >= v1 then r0 else r1`
    /// None arguments are taken as zeroes.
    fn select_argmax(
        &mut self,
        v0: &Value,
        v1: &Value,
        r0: Option<&Value>,
        r1: Option<&Value>,
    ) -> Result<Value>;
}

#[cfg(feature = "arrayfire")]
mod af_arith {
    use super::*;
    use crate::error::check_equal_dimensions;
    use arrayfire as af;

    impl<T> CompareAlgebra<af::Array<T>> for Eval
    where
        Self: CoreAlgebra<af::Array<T>, Value = af::Array<T>>,
        T: af::HasAfEnum
            + af::ImplicitPromote<T, Output = T>
            + af::ConstGenerator<OutType = T>
            + num::Zero,
    {
        #[inline]
        fn min(&mut self, v0: &af::Array<T>, v1: &af::Array<T>) -> Result<af::Array<T>> {
            self.check().min(&v0.dims(), &v1.dims())?;
            Ok(af::minof(v0, v1, false))
        }

        #[inline]
        fn max(&mut self, v0: &af::Array<T>, v1: &af::Array<T>) -> Result<af::Array<T>> {
            self.check().max(&v0.dims(), &v1.dims())?;
            Ok(af::maxof(v0, v1, false))
        }

        fn select_argmax(
            &mut self,
            v0: &af::Array<T>,
            v1: &af::Array<T>,
            r0: Option<&af::Array<T>>,
            r1: Option<&af::Array<T>>,
        ) -> Result<af::Array<T>> {
            self.check().select_argmax(
                &v0.dims(),
                &v1.dims(),
                r0.map(|r| r.dims()).as_ref(),
                r1.map(|r| r.dims()).as_ref(),
            )?;
            let cmp = af::ge(v0, v1, false);
            match (r0, r1) {
                (Some(r0), Some(r1)) => Ok(af::select(r0, &cmp, r1)),
                (None, Some(r1)) => Ok(af::selectl(0.0, &cmp, r1)),
                (Some(r0), None) => Ok(af::selectr(r0, &cmp, 0.0)),
                (None, None) => Ok(af::constant(T::zero(), v0.dims())),
            }
        }
    }

    impl CompareAlgebra<af::Dim4> for Check {
        #[inline]
        fn min(&mut self, v0: &af::Dim4, v1: &af::Dim4) -> Result<af::Dim4> {
            check_equal_dimensions(func_name!(), &[v0, v1])
        }

        #[inline]
        fn max(&mut self, v0: &af::Dim4, v1: &af::Dim4) -> Result<af::Dim4> {
            check_equal_dimensions(func_name!(), &[v0, v1])
        }

        #[inline]
        fn select_argmax(
            &mut self,
            v0: &af::Dim4,
            v1: &af::Dim4,
            r0: Option<&af::Dim4>,
            r1: Option<&af::Dim4>,
        ) -> Result<af::Dim4> {
            check_equal_dimensions(func_name!(), &[v0, v1])?;
            if let Some(r0) = r0 {
                check_equal_dimensions(func_name!(), &[v0, r0])?;
            }
            if let Some(r1) = r1 {
                check_equal_dimensions(func_name!(), &[v1, r1])?;
            }
            Ok(v0.dims())
        }
    }
}

impl<T: Number + PartialOrd> CompareAlgebra<T> for Eval {
    #[inline]
    fn min(&mut self, v0: &T, v1: &T) -> Result<T> {
        if *v0 <= *v1 {
            Ok(*v0)
        } else {
            Ok(*v1)
        }
    }

    #[inline]
    fn max(&mut self, v0: &T, v1: &T) -> Result<T> {
        if *v0 >= *v1 {
            Ok(*v0)
        } else {
            Ok(*v1)
        }
    }

    #[inline]
    fn select_argmax(&mut self, v0: &T, v1: &T, r0: Option<&T>, r1: Option<&T>) -> Result<T> {
        if *v0 >= *v1 {
            Ok(r0.cloned().unwrap_or_else(T::zero))
        } else {
            Ok(r1.cloned().unwrap_or_else(T::zero))
        }
    }
}

impl CompareAlgebra<()> for Check {
    #[inline]
    fn select_argmax(
        &mut self,
        _v0: &(),
        _v1: &(),
        _r0: Option<&()>,
        _r1: Option<&()>,
    ) -> Result<()> {
        Ok(())
    }
}

macro_rules! impl_graph {
    ($config:ident) => {
        impl<D, E, Dims> CompareAlgebra<Value<D>> for Graph<$config<E>>
        where
            E: Default
                + Clone
                + CoreAlgebra<D, Value = D>
                + CompareAlgebra<D>
                + LinkedAlgebra<Value<D>, D>,
            D: HasDims<Dims = Dims> + Clone + 'static + Send + Sync,
            Dims: PartialEq + std::fmt::Debug + Clone + 'static + Send + Sync,
        {
            fn select_argmax(
                &mut self,
                v0: &Value<D>,
                v1: &Value<D>,
                r0: Option<&Value<D>>,
                r1: Option<&Value<D>>,
            ) -> Result<Value<D>> {
                let result = self.eval().select_argmax(
                    v0.data(),
                    v1.data(),
                    r0.map(|r| r.data()),
                    r1.map(|r| r.data()),
                )?;
                let inputs = {
                    let mut i = Vec::new();
                    if let Some(r) = r0 {
                        i.push(r.input());
                    }
                    if let Some(r) = r1 {
                        i.push(r.input());
                    }
                    i
                };
                let value = self.make_node(result, inputs, {
                    let v0 = v0.clone();
                    let v1 = v1.clone();
                    let id0 = r0.and_then(Value::id);
                    let id1 = r1.and_then(Value::id);
                    move |graph, store, gradient| {
                        let c0 = graph.link(&v0);
                        let c1 = graph.link(&v1);
                        if let Some(id) = id0 {
                            let grad = graph.select_argmax(c0, c1, Some(&gradient), None)?;
                            store.add_gradient(graph, id, &grad)?;
                        }
                        if let Some(id) = id1 {
                            let grad = graph.select_argmax(c0, c1, None, Some(&gradient))?;
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
