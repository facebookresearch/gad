// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

use gad::prelude::*;

pub trait UserAlgebra<Value> {
    fn square(&mut self, v: &Value) -> Result<Value>;
}

#[cfg(feature = "arrayfire")]
mod af_arith {
    use super::*;
    use arrayfire as af;

    impl<T> UserAlgebra<af::Array<T>> for Eval
    where
        T: af::HasAfEnum
            + af::ImplicitPromote<T, Output = T>
            + af::ConstGenerator<OutType = T>
            + num::Zero,
    {
        #[inline]
        fn square(&mut self, v: &af::Array<T>) -> Result<af::Array<T>> {
            Ok(v * v)
        }
    }

    impl UserAlgebra<af::Dim4> for Check {
        #[inline]
        fn square(&mut self, v: &af::Dim4) -> Result<af::Dim4> {
            Ok(v.dims())
        }
    }
}

// Sadly, we cannot quantify over T: Number until negative traits are available in Rust:
// https://github.com/rust-lang/rust/issues/68318 is fixed.
macro_rules! impl_eval {
    ($T:ident) => {
        impl UserAlgebra<$T> for Eval {
            #[inline]
            fn square(&mut self, v: &$T) -> Result<$T> {
                Ok((*v) * (*v))
            }
        }
    };
}

impl_eval!(i32);
impl_eval!(i64);
impl_eval!(f32);
impl_eval!(f64);

impl UserAlgebra<()> for Check {
    #[inline]
    fn square(&mut self, _v: &()) -> Result<()> {
        Ok(())
    }
}

macro_rules! impl_graph {
    ($config:ident) => {
        impl<D, E, Dims> UserAlgebra<Value<D>> for Graph<$config<E>>
        where
            E: Default
                + Clone
                + CoreAlgebra<D, Value = D>
                + UserAlgebra<D>
                + ArithAlgebra<D>
                + LinkedAlgebra<Value<D>, D>,
            D: HasDims<Dims = Dims> + Clone + 'static + Send + Sync,
            Dims: PartialEq + std::fmt::Debug + Clone + 'static + Send + Sync,
        {
            fn square(&mut self, v: &Value<D>) -> Result<Value<D>> {
                let result = self.eval().square(v.data())?;
                let value = self.make_node(result, vec![v.input()], {
                    let v = v.clone();
                    move |graph, store, gradient| {
                        if let Some(id) = v.id() {
                            let c = graph.link(&v);
                            let grad1 = graph.mul(&gradient, c)?;
                            let grad2 = graph.mul(c, &gradient)?;
                            store.add_gradient(graph, id, &grad1)?;
                            store.add_gradient(graph, id, &grad2)?;
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

#[test]
fn test_square() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(3i32);
    let b = g.square(&a)?;
    assert_eq!(*b.data(), 9);
    let gradients = g.evaluate_gradients_once(b.gid()?, 1)?;
    assert_eq!(*gradients.get(a.gid()?).unwrap(), 6);
    Ok(())
}
