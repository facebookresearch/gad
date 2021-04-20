// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    core::{CoreAlgebra, HasDims},
    error::Result,
    graph::{Config1, ConfigN, Graph, Value},
    linked::LinkedAlgebra,
    store::GradientStore,
};

/// Whether a matrix should be transposed and/or conjugated before applying a matrix operation.
#[derive(Default, Debug, Copy, Clone, Eq, PartialEq)]
pub struct MatProp {
    pub transposed: bool,
    pub conjugated: bool,
}

/// Matric operations such as multiplication and transposition.
pub trait MatrixAlgebra<Value> {
    /// Multiplication of two matrices after some optional transpositions.
    fn matmul(&mut self, v1: &Value, v2: &Value, prop1: MatProp, prop2: MatProp) -> Result<Value>;

    /// Transpose (and optionally conjuguate) a matrix.
    fn transpose(&mut self, v: &Value, conjugate: bool) -> Result<Value>;

    /// Non-transposed multiplication of two matrices.
    #[inline]
    fn matmul_nn(&mut self, v1: &Value, v2: &Value) -> Result<Value> {
        self.matmul(v1, v2, MatProp::default(), MatProp::default())
    }
}

#[cfg(feature = "arrayfire")]
mod af_arith {
    use super::*;
    use crate::{arrayfire::Float, error::Error, Check, Eval};
    use arrayfire as af;

    impl<T> MatrixAlgebra<af::Array<T>> for Eval
    where
        T: Float,
    {
        #[inline]
        fn matmul(
            &mut self,
            v1: &af::Array<T>,
            v2: &af::Array<T>,
            prop1: MatProp,
            prop2: MatProp,
        ) -> Result<af::Array<T>> {
            self.check().matmul(&v1.dims(), &v2.dims(), prop1, prop2)?;
            Ok(af::matmul(v1, v2, prop1.into(), prop2.into()))
        }

        #[inline]
        fn transpose(&mut self, v: &af::Array<T>, conjugate: bool) -> Result<af::Array<T>> {
            self.check().transpose(&v.dims(), conjugate)?;
            Ok(af::transpose(v, conjugate))
        }
    }

    impl From<MatProp> for af::MatProp {
        fn from(p: MatProp) -> af::MatProp {
            match p {
                MatProp {
                    transposed: false,
                    conjugated: false,
                } => af::MatProp::NONE,
                MatProp {
                    transposed: true,
                    conjugated: false,
                } => af::MatProp::TRANS,
                MatProp {
                    transposed: false,
                    conjugated: true,
                } => af::MatProp::CONJ,
                MatProp {
                    transposed: true,
                    conjugated: true,
                } => af::MatProp::CTRANS,
            }
        }
    }

    impl MatrixAlgebra<af::Dim4> for Check {
        #[inline]
        fn matmul(
            &mut self,
            v1: &af::Dim4,
            v2: &af::Dim4,
            prop1: MatProp,
            prop2: MatProp,
        ) -> Result<af::Dim4> {
            let tv1 = if prop1.transposed {
                self.transpose(v1, false)?
            } else {
                *v1
            };
            let tv2 = if prop2.transposed {
                self.transpose(v2, false)?
            } else {
                *v2
            };
            if tv1[1] != tv2[0] {
                return Err(Error::dimensions(func_name!(), &[v1, v2]));
            }
            let r = match (tv1[2], tv1[3], tv2[2], tv2[3]) {
                (1, 1, a, b) | (a, b, 1, 1) => [tv1[0], tv2[1], a, b],
                (a, b, c, d) if a == c && b == d => [tv1[0], tv2[1], a, b],
                _ => {
                    return Err(Error::dimensions(func_name!(), &[v1, v2]));
                }
            };
            Ok(af::Dim4::new(&r))
        }

        #[inline]
        fn transpose(&mut self, v: &af::Dim4, _conjugate: bool) -> Result<af::Dim4> {
            if (v[2], v[3]) != (1, 1) {
                Err(Error::dimensions(func_name!(), &[v]))
            } else {
                Ok(af::Dim4::new(&[v[1], v[0], 1, 1]))
            }
        }
    }

    #[test]
    fn test_af_matprop() {
        let p = MatProp::new();
        assert_eq!(af::MatProp::from(p), af::MatProp::NONE);
        assert_eq!(af::MatProp::from(p.transpose()), af::MatProp::TRANS);
        assert_eq!(af::MatProp::from(p.conjugate()), af::MatProp::CONJ);
        assert_eq!(
            af::MatProp::from(p.transpose().conjugate()),
            af::MatProp::CTRANS
        );
    }
}

macro_rules! impl_graph {
    ($config:ident) => {
        impl<D, E, Dims> MatrixAlgebra<Value<D>> for Graph<$config<E>>
        where
            E: Default
                + Clone
                + CoreAlgebra<D, Value = D>
                + LinkedAlgebra<Value<D>, D>
                + MatrixAlgebra<D>,
            D: HasDims<Dims = Dims> + Clone + 'static + Send + Sync,
            Dims: PartialEq + std::fmt::Debug + Clone + 'static + Send + Sync,
        {
            fn matmul(
                &mut self,
                v1: &Value<D>,
                v2: &Value<D>,
                prop1: MatProp,
                prop2: MatProp,
            ) -> Result<Value<D>> {
                let result = self.eval().matmul(v1.data(), v2.data(), prop1, prop2)?;
                let value = self.make_node(result, vec![v1.input(), v2.input()], {
                    let v1 = v1.clone();
                    let v2 = v2.clone();
                    move |graph, store, gradient| {
                        if let Some(id) = v1.id() {
                            let c2 = graph.link(&v2);
                            let grad = graph.matmul(&gradient, c2, prop1, prop2.transpose())?;
                            store.add_gradient(graph, id, &grad)?;
                        }
                        if let Some(id) = v2.id() {
                            let c1 = graph.link(&v1);
                            let grad = graph.matmul(c1, &gradient, prop1.transpose(), prop2)?;
                            store.add_gradient(graph, id, &grad)?;
                        }
                        Ok(())
                    }
                });
                Ok(value)
            }

            fn transpose(&mut self, v: &Value<D>, conjugate: bool) -> Result<Value<D>> {
                let result = self.eval().transpose(v.data(), conjugate)?;
                let value = self.make_node(result, vec![v.input()], {
                    let id = v.id();
                    move |graph, store, gradient| {
                        if let Some(id) = id {
                            let grad = graph.transpose(&gradient, conjugate)?;
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

impl MatProp {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn transpose(self) -> Self {
        Self {
            transposed: !self.transposed,
            conjugated: self.conjugated,
        }
    }

    pub fn conjugate(self) -> Self {
        Self {
            transposed: self.transposed,
            conjugated: !self.conjugated,
        }
    }
}

#[test]
fn test_matprop() {
    let p = MatProp::new();
    assert!(p.transpose().transposed);
    assert_eq!(p.transpose().transpose(), p);
    assert!(p.conjugate().conjugated);
}
