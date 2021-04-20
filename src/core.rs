// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    error::{Error, Result},
    graph::{Config1, ConfigN, Graph, Value},
    store::GradientStore,
    Check, Eval, Number,
};

/// Core trait for data operations that support differentiation of values.
pub trait CoreAlgebra<Data> {
    /// Tracked values of underlying type Data.
    type Value;

    /// A differential input to the computation.
    fn variable(&mut self, data: Data) -> Self::Value;

    /// A non-differential input to the computation.
    fn constant(&mut self, data: Data) -> Self::Value;

    /// Compute the sum of two values `v1 + v2`.
    fn add(&mut self, v1: &Self::Value, v2: &Self::Value) -> Result<Self::Value>;

    /// Compute the sum of several values.
    fn add_all(&mut self, values: &[&Self::Value]) -> Result<Self::Value>
    where
        Self::Value: Clone,
    {
        let mut values = values.iter();
        let mut result: Self::Value =
            (*values.next().ok_or_else(|| Error::empty(func_name!()))?).clone();
        for value in values {
            result = self.add(&result, *value)?;
        }
        Ok(result)
    }
}

/// Obtain the dimensions of a value.
pub trait HasDims {
    type Dims;

    fn dims(&self) -> Self::Dims;
}

impl<A> HasDims for crate::graph::Value<A>
where
    A: HasDims,
{
    type Dims = A::Dims;

    #[inline]
    fn dims(&self) -> Self::Dims {
        self.data().dims()
    }
}

impl<T: Number> HasDims for T {
    type Dims = ();

    #[inline]
    fn dims(&self) {}
}

impl<T: HasDims> HasDims for std::sync::Arc<T> {
    type Dims = T::Dims;

    #[inline]
    fn dims(&self) -> Self::Dims {
        self.as_ref().dims()
    }
}

impl<T: Number> CoreAlgebra<T> for Check {
    type Value = ();

    #[inline]
    fn variable(&mut self, _data: T) {}

    #[inline]
    fn constant(&mut self, _data: T) {}

    #[inline]
    fn add(&mut self, _v0: &(), _v1: &()) -> Result<()> {
        Ok(())
    }

    #[inline]
    fn add_all(&mut self, _values: &[&()]) -> Result<()> {
        Ok(())
    }
}

impl<T: Number> CoreAlgebra<T> for Eval {
    type Value = T;

    #[inline]
    fn variable(&mut self, data: T) -> T {
        data
    }

    #[inline]
    fn constant(&mut self, data: T) -> T {
        data
    }

    #[inline]
    fn add(&mut self, v0: &T, v1: &T) -> Result<T> {
        Ok(*v0 + *v1)
    }
}

#[cfg(feature = "arrayfire")]
mod af_core {
    use super::*;
    use crate::error::check_equal_dimensions;
    use arrayfire as af;

    impl<T: af::HasAfEnum> HasDims for af::Array<T> {
        type Dims = af::Dim4;

        #[inline]
        fn dims(&self) -> af::Dim4 {
            self.dims()
        }
    }

    impl HasDims for af::Dim4 {
        type Dims = af::Dim4;

        #[inline]
        fn dims(&self) -> af::Dim4 {
            *self
        }
    }

    impl<T: af::HasAfEnum> CoreAlgebra<af::Array<T>> for Check {
        type Value = af::Dim4;

        #[inline]
        fn variable(&mut self, array: af::Array<T>) -> af::Dim4 {
            array.dims()
        }

        #[inline]
        fn constant(&mut self, array: af::Array<T>) -> af::Dim4 {
            array.dims()
        }

        #[inline]
        fn add(&mut self, v0: &af::Dim4, v1: &af::Dim4) -> Result<af::Dim4> {
            check_equal_dimensions(func_name!(), &[v0, v1])
        }

        #[inline]
        fn add_all(&mut self, values: &[&af::Dim4]) -> Result<af::Dim4> {
            check_equal_dimensions(func_name!(), values)
        }
    }

    impl<T> CoreAlgebra<af::Array<T>> for Eval
    where
        T: af::HasAfEnum + af::ImplicitPromote<T, Output = T>,
    {
        type Value = af::Array<T>;

        #[inline]
        fn variable(&mut self, array: af::Array<T>) -> af::Array<T> {
            array
        }

        #[inline]
        fn constant(&mut self, array: af::Array<T>) -> af::Array<T> {
            array
        }

        #[inline]
        fn add(&mut self, v0: &af::Array<T>, v1: &af::Array<T>) -> Result<af::Array<T>> {
            <Check as CoreAlgebra<af::Array<T>>>::add(self.check(), &v0.dims(), &v1.dims())?;
            Ok(v0 + v1)
        }
    }
}

// Cannot implement Graph<C> generically over C: Config... for now because of
// compiler limitations while resolving recursive trait requirements.
//
// impl<D, G, C: Config, Dims> CoreAlgebra<D> for Graph<C>
// where
//     C::EvalAlgebra: CoreAlgebra<D, Value = D>,
//     C::GradientAlgebra: CoreAlgebra<D, Value = G>, // <-- causes recursion when C::GradientAlgebra = Graph<C>
//     C::GradientStore: GradientStore<G, GradientId<D>>,
//     D: HasDims<Dims = Dims> + Clone + 'static,
//     G: HasDims<Dims = Dims> + Clone + 'static,
//     Dims: PartialEq + std::fmt::Debug + Clone + 'static,
// {
//     type Value = Value(D);
//
//     ...
// }

macro_rules! impl_graph {
    ($config:ident) => {
        impl<D, E, Dims> CoreAlgebra<D> for Graph<$config<E>>
        where
            E: Default + Clone + CoreAlgebra<D, Value = D>,
            D: HasDims<Dims = Dims> + Clone + 'static + Send + Sync,
            Dims: PartialEq + std::fmt::Debug + Clone + 'static + Send + Sync,
        {
            type Value = Value<D>;

            fn variable(&mut self, data: D) -> Value<D> {
                self.make_variable(data)
            }

            fn constant(&mut self, data: D) -> Value<D> {
                Value::constant(data)
            }

            fn add(&mut self, v1: &Value<D>, v2: &Value<D>) -> Result<Value<D>> {
                let result = self.eval().add(v1.data(), v2.data())?;
                let value = self.make_node(result, vec![v1.input(), v2.input()], {
                    let id1 = v1.id();
                    let id2 = v2.id();
                    move |graph, store, gradient| {
                        if let Some(id) = id1 {
                            store.add_gradient(graph, id, &gradient)?;
                        }
                        if let Some(id) = id2 {
                            store.add_gradient(graph, id, &gradient)?;
                        }
                        Ok(())
                    }
                });
                Ok(value)
            }

            fn add_all(&mut self, values: &[&Value<D>]) -> Result<Value<D>> {
                let result = self
                    .eval()
                    .add_all(&values.iter().map(|v| v.data()).collect::<Vec<_>>())?;
                let inputs = values.iter().map(|v| v.input()).collect::<Vec<_>>();
                let value = self.make_node(result, inputs, {
                    let ids = values.iter().map(|v| v.id()).collect::<Vec<_>>();
                    move |graph, store, gradient| {
                        for id in &ids {
                            if let Some(id) = id {
                                store.add_gradient(graph, *id, &gradient)?;
                            }
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
