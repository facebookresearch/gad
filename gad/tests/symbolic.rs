// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use gad::prelude::*;
use std::sync::Arc;

/// Symbolic evaluation.
#[derive(Clone, Default)]
struct SymEval;

/// Symbolic expression of type T (unboxed).
#[derive(Debug, PartialEq)]
enum Exp_<T> {
    Zero,
    One,
    Num(T),
    Neg(Exp<T>),
    Add(Exp<T>, Exp<T>),
    Mul(Exp<T>, Exp<T>),
}

/// Symbolic expression of type T (boxed.)
type Exp<T> = Arc<Exp_<T>>;

impl<T> Exp_<T> {
    fn num(x: T) -> Exp<T> {
        Arc::new(Exp_::Num(x))
    }
}

impl<T> HasDims for Exp_<T> {
    type Dims = ();

    #[inline]
    fn dims(&self) {}
}

impl<T> CoreAlgebra<Exp<T>> for SymEval {
    type Value = Exp<T>;

    fn variable(&mut self, data: Exp<T>) -> Self::Value {
        data
    }

    fn constant(&mut self, data: Exp<T>) -> Self::Value {
        data
    }

    fn add(&mut self, v1: &Self::Value, v2: &Self::Value) -> Result<Self::Value> {
        Ok(Arc::new(Exp_::Add(v1.clone(), v2.clone())))
    }
}

impl<T> ArithAlgebra<Exp<T>> for SymEval {
    fn zeros(&mut self, _v: &Exp<T>) -> Exp<T> {
        Arc::new(Exp_::Zero)
    }

    fn ones(&mut self, _v: &Exp<T>) -> Exp<T> {
        Arc::new(Exp_::One)
    }

    fn neg(&mut self, v: &Exp<T>) -> Exp<T> {
        Arc::new(Exp_::Neg(v.clone()))
    }

    fn sub(&mut self, v1: &Exp<T>, v2: &Exp<T>) -> Result<Exp<T>> {
        let v2 = self.neg(v2);
        Ok(Arc::new(Exp_::Add(v1.clone(), v2)))
    }

    fn mul(&mut self, v1: &Exp<T>, v2: &Exp<T>) -> Result<Exp<T>> {
        Ok(Arc::new(Exp_::Mul(v1.clone(), v2.clone())))
    }
}

impl<T: std::fmt::Display> std::fmt::Display for Exp_<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Exp_::*;
        match self {
            Zero => write!(f, "0"),
            One => write!(f, "1"),
            Num(x) => write!(f, "{}", x),
            Neg(e) => write!(f, "(-{})", *e),
            Add(e1, e2) => write!(f, "({}+{})", *e1, *e2),
            Mul(e1, e2) => write!(f, "{}{}", *e1, *e2),
        }
    }
}

type SymGraph1 = Graph<Config1<SymEval>>;
// type SymGraphN = Graph<ConfigN<SymEval>>;

#[test]
fn test_symgraph1() -> Result<()> {
    let mut g = SymGraph1::new();
    let a = CoreAlgebra::variable(&mut g, Exp_::num("a"));
    let b = g.variable(Exp_::num("b"));
    let c = g.mul(&a, &b)?;
    let d = g.mul(&a, &c)?;
    assert_eq!(format!("{}", d.data()), "aab");
    let gradients = g.evaluate_gradients_once(d.gid()?, Exp_::num("1"))?;
    assert_eq!(format!("{}", gradients.get(a.gid()?).unwrap()), "(1ab+a1b)");
    assert_eq!(format!("{}", gradients.get(b.gid()?).unwrap()), "aa1");
    Ok(())
}
