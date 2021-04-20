// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    arith::ArithAlgebra,
    array::ArrayAlgebra,
    compare::CompareAlgebra,
    core::{CoreAlgebra, HasDims},
    error::Result,
    graph::{Config1, ConfigN, Graph, Value},
    linked::LinkedAlgebra,
    store::GradientStore,
};

/// Array-oriented comparison operations.
pub trait ArrayCompareAlgebra<Value>: CompareAlgebra<Value> + ArrayAlgebra<Value> {
    fn max_as(&mut self, v: &Value, dims: Self::Dims) -> Result<Value>;

    fn argmax_as(&mut self, v: &Value, dims: Self::Dims) -> Result<Value>;

    fn softmax_as(&mut self, v: &Value, dims: Self::Dims) -> Result<Value>;
}

#[cfg(feature = "arrayfire")]
mod af_arith {
    use super::*;
    use crate::{analytic::AnalyticAlgebra, error, Check, Eval};
    use arrayfire as af;

    impl<T> ArrayCompareAlgebra<af::Array<T>> for Eval
    where
        Self: CoreAlgebra<af::Array<T>, Value = af::Array<T>> + AnalyticAlgebra<af::Array<T>>,
        T: crate::arrayfire::Float
            + af::ImplicitPromote<T, Output = T>
            + af::ConstGenerator<OutType = T>
            + num::Zero,
    {
        fn max_as(&mut self, v: &af::Array<T>, rdims: af::Dim4) -> Result<af::Array<T>> {
            self.check().max_as(&v.dims(), rdims)?;
            let vdims = v.dims();
            let mut result = v.clone();
            for i in 0..4 {
                if rdims[i] == vdims[i] {
                    continue;
                }
                result = af::max(&result, i as i32);
            }
            Ok(result)
        }

        fn argmax_as(&mut self, v: &af::Array<T>, rdims: af::Dim4) -> Result<af::Array<T>> {
            let dims = v.dims();
            let max = {
                let rmax = self.max_as(v, rdims)?;
                self.tile_as(&rmax, dims)?
            };
            let ones = af::constant(T::one(), dims);
            let arg = self.select_argmax(v, &max, Some(&ones), None)?;
            let sum = {
                let rsum = self.sum_as(&arg, rdims)?;
                self.tile_as(&rsum, dims)?
            };
            self.div(&arg, &sum)
        }

        fn softmax_as(&mut self, v: &af::Array<T>, rdims: af::Dim4) -> Result<af::Array<T>> {
            let dims = v.dims();
            let max = {
                let rmax = self.max_as(v, rdims)?;
                self.tile_as(&rmax, dims)?
            };
            let exp = {
                let delta = self.sub(v, &max)?;
                self.exp(&delta)
            };
            let sum = {
                let rsum = self.sum_as(&exp, rdims)?;
                self.tile_as(&rsum, dims)?
            };
            self.div(&exp, &sum)
        }
    }

    impl ArrayCompareAlgebra<af::Dim4> for Check {
        #[inline]
        fn max_as(&mut self, v: &af::Dim4, rdims: af::Dim4) -> Result<af::Dim4> {
            error::af::check_reduced_dimensions(func_name!(), *v, rdims)
        }

        #[inline]
        fn argmax_as(&mut self, v: &af::Dim4, rdims: af::Dim4) -> Result<af::Dim4> {
            error::af::check_reduced_dimensions(func_name!(), *v, rdims)?;
            Ok(*v)
        }

        #[inline]
        fn softmax_as(&mut self, v: &af::Dim4, rdims: af::Dim4) -> Result<af::Dim4> {
            error::af::check_reduced_dimensions(func_name!(), *v, rdims)?;
            Ok(*v)
        }
    }
}

macro_rules! impl_graph {
    ($config:ident) => {
        impl<D, E, T, Dims> ArrayCompareAlgebra<Value<D>> for Graph<$config<E>>
        where
            E: Default
                + Clone
                + CoreAlgebra<D, Value = D>
                + CoreAlgebra<T, Value = T>
                + CompareAlgebra<D>
                + ArrayCompareAlgebra<D>
                + ArrayAlgebra<D, Dims = Dims>
                + ArithAlgebra<D>
                + ArrayAlgebra<D, Scalar = T, Dims = Dims>
                + LinkedAlgebra<Value<D>, D>
                + LinkedAlgebra<Value<T>, T>,
            T: crate::Number,
            D: HasDims<Dims = Dims> + Clone + 'static + Send + Sync,
            Dims: PartialEq + std::fmt::Debug + Default + Copy + Clone + 'static + Send + Sync,
        {
            fn max_as(&mut self, v: &Value<D>, rdims: Dims) -> Result<Value<D>> {
                let result = self.eval().max_as(v.data(), rdims)?;
                let value = self.make_node(result, vec![v.input()], {
                    let v = v.clone();
                    move |graph, store, gradient| {
                        if let Some(id) = v.id() {
                            let v = graph.link(&v);
                            let mask = graph.argmax_as(v, rdims)?;
                            let tiled = graph.tile_as(&gradient, v.dims())?;
                            let grad = graph.mul(&tiled, &mask)?;
                            store.add_gradient::<D, _>(graph, id, &grad)?;
                        }
                        Ok(())
                    }
                });
                Ok(value)
            }

            fn argmax_as(&mut self, v: &Value<D>, rdims: Dims) -> Result<Value<D>> {
                let result = self.eval().argmax_as(v.data(), rdims)?;
                Ok(self.constant(result))
            }

            fn softmax_as(&mut self, v: &Value<D>, rdims: Dims) -> Result<Value<D>> {
                let result = self.eval().softmax_as(v.data(), rdims)?;
                let value = self.make_node(result, vec![v.input()], {
                    let v = v.clone();
                    let dims = v.dims();
                    move |graph, store, gradient| {
                        if let Some(id) = v.id() {
                            let v = graph.link(&v);
                            let res = graph.softmax_as(v, rdims)?;
                            let g1 = graph.mul(&gradient, &res)?;
                            let g2 = {
                                let rg = graph.sum_as(&g1, rdims)?;
                                let g = graph.tile_as(&rg, dims)?;
                                graph.mul(&g, &res)?
                            };
                            let grad = graph.sub(&g1, &g2)?;
                            store.add_gradient::<D, _>(graph, id, &grad)?;
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
