// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    core::{CoreAlgebra, HasDims},
    error::Result,
    graph::{Config1, ConfigN, Graph, Value},
    linked::LinkedAlgebra,
    store::GradientStore,
};

/// Array operations.
pub trait ArrayAlgebra<Value> {
    type Dims;
    type Scalar;

    /// Re-shape the input into a single dimension array.
    fn flat(&mut self, v: &Value) -> Value;

    /// Re-shape the input into an array of the given dimensions.
    fn moddims(&mut self, v: &Value, dims: Self::Dims) -> Result<Value>;

    /// Repeats the input to match the given shape.
    fn tile_as(&mut self, v: &Value, dims: Self::Dims) -> Result<Value>;

    /// Sums some of the dimension of the input to fit the given shape.
    fn sum_as(&mut self, v: &Value, dims: Self::Dims) -> Result<Value>;

    /// Fill an array of the given shape with the given scalar value.
    fn constant_as(&mut self, v: &Self::Scalar, dims: Self::Dims) -> Value;

    /// Read the scalar value in a one-element array.
    fn as_scalar(&mut self, v: &Value) -> Result<Self::Scalar>;

    /// Multiply the array element-wise by the given scalar.
    fn scale(&mut self, lambda: &Self::Scalar, v: &Value) -> Value;

    /// Compute the dot-product of two arrays of the same shape.
    fn dot(&mut self, v1: &Value, v2: &Value) -> Result<Self::Scalar>;

    /// Compute the L2-norm of an array.
    fn norm2(&mut self, v: &Value) -> Self::Scalar {
        self.dot(v, v).expect("norm2 should not fail")
    }
}

#[cfg(feature = "arrayfire")]
mod af_arith {
    use crate::{
        array::ArrayAlgebra,
        arrayfire::Float,
        error::{check_equal_dimensions, Error, Result},
        Check, Eval,
    };
    use arrayfire as af;

    impl<T> ArrayAlgebra<af::Array<T>> for Eval
    where
        T: Float,
    {
        type Dims = af::Dim4;
        type Scalar = T;

        #[inline]
        fn flat(&mut self, v: &af::Array<T>) -> af::Array<T> {
            af::flat(v)
        }

        #[inline]
        fn moddims(&mut self, v: &af::Array<T>, dims: af::Dim4) -> Result<af::Array<T>> {
            self.check().moddims(&v.dims(), dims)?;
            Ok(af::moddims(v, dims))
        }

        #[inline]
        fn tile_as(&mut self, v: &af::Array<T>, rdims: af::Dim4) -> Result<af::Array<T>> {
            self.check().tile_as(&v.dims(), rdims)?;
            let vdims = v.dims();
            let mut tdims = [1u64; 4];
            for i in 0..4 {
                tdims[i] = rdims[i] / vdims[i];
            }
            Ok(af::tile(&v, af::Dim4::new(&tdims)))
        }

        #[inline]
        fn sum_as(&mut self, v: &af::Array<T>, rdims: af::Dim4) -> Result<af::Array<T>> {
            self.check().sum_as(&v.dims(), rdims)?;
            let vdims = v.dims();
            let mut result = v.clone();
            for i in 0..4 {
                if rdims[i] == vdims[i] {
                    continue;
                }
                result = af::sum(&result, i as i32).cast();
            }
            Ok(result)
        }

        #[inline]
        fn constant_as(&mut self, v: &T, dims: af::Dim4) -> af::Array<T> {
            af::constant(*v, dims)
        }

        #[inline]
        fn as_scalar(&mut self, v: &af::Array<T>) -> Result<T> {
            self.check().as_scalar(&v.dims())?;
            let mut res = vec![T::zero(); 1];
            v.host(&mut res);
            Ok(res[0])
        }

        #[inline]
        fn scale(&mut self, lambda: &T, v: &af::Array<T>) -> af::Array<T> {
            v * (*lambda)
        }

        #[inline]
        fn dot(&mut self, v1: &af::Array<T>, v2: &af::Array<T>) -> Result<T> {
            self.check().dot(&v1.dims(), &v2.dims())?;
            let v1 = af::flat(v1);
            let v2 = af::flat(v2);
            let mut res = vec![T::zero(); 1];
            af::dot(&v1, &v2, af::MatProp::CONJ, af::MatProp::NONE).host(&mut res);
            Ok(res[0])
        }
    }

    impl ArrayAlgebra<af::Dim4> for Check {
        type Dims = af::Dim4;
        type Scalar = ();

        #[inline]
        fn flat(&mut self, v: &af::Dim4) -> af::Dim4 {
            af::dim4!(v.elements())
        }

        #[inline]
        fn moddims(&mut self, v: &af::Dim4, dims: af::Dim4) -> Result<af::Dim4> {
            if v.elements() != dims.elements() {
                Err(Error::dimensions(func_name!(), &[v, &dims]))
            } else {
                Ok(dims)
            }
        }

        #[inline]
        fn tile_as(&mut self, v: &af::Dim4, rdims: af::Dim4) -> Result<af::Dim4> {
            let mut tdims = [1u64; 4];
            for i in 0..4 {
                if rdims[i] % v[i] != 0 {
                    return Err(Error::dimensions(func_name!(), &[v, &rdims]));
                }
                tdims[i] = rdims[i] / v[i];
            }
            Ok(rdims)
        }

        #[inline]
        fn sum_as(&mut self, v: &af::Dim4, rdims: af::Dim4) -> Result<af::Dim4> {
            for i in 0..4 {
                if rdims[i] == v[i] {
                    continue;
                }
                if rdims[i] != 1 {
                    return Err(Error::dimensions(func_name!(), &[v, &rdims]));
                }
            }
            Ok(rdims)
        }

        #[inline]
        fn constant_as(&mut self, _v: &(), dims: af::Dim4) -> af::Dim4 {
            dims
        }

        #[inline]
        fn as_scalar(&mut self, v: &af::Dim4) -> Result<()> {
            check_equal_dimensions(func_name!(), &[v, &af::dim4!(1)])?;
            Ok(())
        }

        #[inline]
        fn scale(&mut self, _lambda: &(), v: &af::Dim4) -> af::Dim4 {
            *v
        }

        #[inline]
        fn dot(&mut self, v1: &af::Dim4, v2: &af::Dim4) -> Result<()> {
            check_equal_dimensions(func_name!(), &[v1, v2])?;
            Ok(())
        }
    }
}

macro_rules! impl_graph {
    ($config:ident) => {
        impl<D, E, T, Dims> ArrayAlgebra<Value<D>> for Graph<$config<E>>
        where
            E: Default
                + Clone
                + CoreAlgebra<D, Value = D>
                + CoreAlgebra<T, Value = T>
                + LinkedAlgebra<Value<D>, D>
                + LinkedAlgebra<Value<T>, T>
                + ArrayAlgebra<D, Scalar = T, Dims = Dims>,
            Dims: PartialEq + Clone + Copy + std::fmt::Debug + Default + 'static + Send + Sync,
            D: HasDims<Dims = Dims> + Clone + 'static + Send + Sync,
            T: crate::Number,
        {
            type Dims = Dims;
            type Scalar = Value<T>;

            fn flat(&mut self, v: &Value<D>) -> Value<D> {
                let result = self.eval().flat(v.data());
                self.make_node(result, vec![v.input()], {
                    let vdims = v.data().dims();
                    let id = v.id();
                    move |graph, store, gradient| {
                        if let Some(id) = id {
                            let x = graph.moddims(&gradient, vdims)?;
                            store.add_gradient::<D, _>(graph, id, &x)?;
                        }
                        Ok(())
                    }
                })
            }

            fn moddims(&mut self, v: &Value<D>, rdims: Dims) -> Result<Value<D>> {
                let result = self.eval().moddims(v.data(), rdims)?;
                let value = self.make_node(result, vec![v.input()], {
                    let vdims = v.data().dims();
                    let id = v.id();
                    move |graph, store, gradient| {
                        if let Some(id) = id {
                            let x = graph.moddims(&gradient, vdims)?;
                            store.add_gradient::<D, _>(graph, id, &x)?;
                        }
                        Ok(())
                    }
                });
                Ok(value)
            }

            fn tile_as(&mut self, v: &Value<D>, rdims: Dims) -> Result<Value<D>> {
                let result = self.eval().tile_as(v.data(), rdims)?;
                let value = self.make_node(result, vec![v.input()], {
                    let vdims = v.data().dims();
                    let id = v.id();
                    move |graph, store, gradient| {
                        if let Some(id) = id {
                            let x = graph.sum_as(&gradient, vdims)?;
                            store.add_gradient::<D, _>(graph, id, &x)?;
                        }
                        Ok(())
                    }
                });
                Ok(value)
            }

            fn sum_as(&mut self, v: &Value<D>, rdims: Dims) -> Result<Value<D>> {
                let result = self.eval().sum_as(v.data(), rdims)?;
                let value = self.make_node(result, vec![v.input()], {
                    let vdims = v.data().dims();
                    let id = v.id();
                    move |graph, store, gradient| {
                        if let Some(id) = id {
                            let x = graph.tile_as(&gradient, vdims)?;
                            store.add_gradient::<D, _>(graph, id, &x)?;
                        }
                        Ok(())
                    }
                });
                Ok(value)
            }

            fn constant_as(&mut self, v: &Value<T>, dims: Dims) -> Value<D> {
                let result = self.eval().constant_as(v.data(), dims);
                let value = self.make_generic_node::<T, D, _, _, _, _>(result, vec![v.input()], {
                    let id = v.id();
                    move |graph, store, gradient| {
                        if let Some(id) = id {
                            let x = graph.sum_as(&gradient, Dims::default())?;
                            let y = graph.as_scalar(&x)?;
                            store.add_gradient::<T, _>(graph, id, &y)?;
                        }
                        Ok(())
                    }
                });
                value
            }

            fn as_scalar(&mut self, v: &Value<D>) -> Result<Value<T>> {
                let result = self.eval().as_scalar(v.data())?;
                let value = self.make_generic_node::<D, T, _, _, _, _>(result, vec![v.input()], {
                    let vdims = v.dims();
                    let id = v.id();
                    move |graph, store, gradient| {
                        if let Some(id) = id {
                            let x = graph.constant_as(&gradient, vdims);
                            store.add_gradient::<D, _>(graph, id, &x)?;
                        }
                        Ok(())
                    }
                });
                Ok(value)
            }

            fn scale(&mut self, v1: &Value<T>, v2: &Value<D>) -> Value<D> {
                let result = self.eval().scale(v1.data(), v2.data());
                let value = self.make_node(result, vec![v1.input(), v2.input()], {
                    let v1 = v1.clone();
                    let v2 = v2.clone();
                    move |graph, store, gradient| {
                        if let Some(id) = v1.id() {
                            let c2 = graph.link(&v2);
                            let grad = graph.dot(&gradient, c2)?;
                            store.add_gradient::<T, _>(graph, id, &grad)?;
                        }
                        if let Some(id) = v2.id() {
                            let c1 = graph.link(&v1);
                            let grad = graph.scale(c1, &gradient);
                            store.add_gradient::<D, _>(graph, id, &grad)?;
                        }
                        Ok(())
                    }
                });
                value
            }

            fn dot(&mut self, v1: &Value<D>, v2: &Value<D>) -> Result<Value<T>> {
                let result = self.eval().dot(v1.data(), v2.data())?;
                let value = self.make_node(result, vec![v1.input(), v2.input()], {
                    let v1 = v1.clone();
                    let v2 = v2.clone();
                    move |graph, store, gradient| {
                        if let Some(id) = v1.id() {
                            let c2 = graph.link(&v2);
                            let grad = graph.scale(&gradient, c2);
                            store.add_gradient::<D, _>(graph, id, &grad)?;
                        }
                        if let Some(id) = v2.id() {
                            let c1 = graph.link(&v1);
                            let grad = graph.scale(&gradient, c1);
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
