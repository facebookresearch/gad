// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::unnecessary_wraps)]

use gad::prelude::*;

#[test]
fn test_zero() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(3i32);
    let b = g.zeros(&a);
    assert_eq!(*b.data(), 0);
    assert_eq!(b.id(), None);
    Ok(())
}

#[test]
fn test_one() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(3i32);
    let b = g.ones(&a);
    assert_eq!(*b.data(), 1);
    assert_eq!(b.id(), None);
    Ok(())
}

#[test]
fn test_neg() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(3i32);
    let b = g.neg(&a);
    assert_eq!(*b.data(), -3);
    let gradients = g.evaluate_gradients_once(b.gid()?, 1)?;
    assert_eq!(*gradients.get(a.gid()?).unwrap(), -1);
    Ok(())
}

#[test]
fn test_sub() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(1i32);
    let b = g.variable(2i32);
    let c = g.sub(&a, &b)?;
    assert_eq!(*c.data(), -1);
    let gradients = g.evaluate_gradients_once(c.gid()?, 1)?;
    assert_eq!(*gradients.get(a.gid()?).unwrap(), 1);
    assert_eq!(*gradients.get(b.gid()?).unwrap(), -1);
    Ok(())
}

#[test]
fn test_mul() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(1i32);
    let b = g.variable(2i32);
    let c = g.mul(&a, &b)?;
    assert_eq!(*c.data(), 2);
    let gradients = g.evaluate_gradients_once(c.gid()?, 1)?;
    assert_eq!(*gradients.get(a.gid()?).unwrap(), 2);
    assert_eq!(*gradients.get(b.gid()?).unwrap(), 1);
    Ok(())
}

#[cfg(feature = "arrayfire")]
mod af_arith_test {
    use super::*;
    use arrayfire as af;

    #[test]
    fn test_neg() -> Result<()> {
        let dims = af::Dim4::new(&[4, 3, 1, 1]);

        let mut g = Graph1::new();
        let a = g.variable(af::randu::<f32>(dims));
        let b = g.neg(&a);

        let direction = af::constant(1f32, dims);
        let gradients = g.evaluate_gradients_once(b.gid()?, direction.clone())?;

        let grad = gradients.get(a.gid()?).unwrap();
        let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| 0.0f32 - x);
        testing::assert_almost_all_equal(&grad, &est, 0.001);
        Ok(())
    }

    #[test]
    fn test_sub() -> Result<()> {
        let dims = af::Dim4::new(&[4, 3, 1, 1]);

        let mut g = Graph1::new();
        let a = g.variable(af::randu::<f32>(dims));
        let b = g.variable(af::randu::<f32>(dims));
        let c = g.sub(&a, &b)?;
        let direction = af::constant(1f32, dims);
        let gradients = g.evaluate_gradients_once(c.gid()?, direction.clone())?;
        {
            let grad = gradients.get(a.gid()?).unwrap();
            let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| x - b.data());
            testing::assert_almost_all_equal(&grad, &est, 0.001);
        }
        {
            let grad = gradients.get(b.gid()?).unwrap();
            let est = testing::estimate_gradient(b.data(), &direction, 0.001f32, |x| a.data() - x);
            testing::assert_almost_all_equal(&grad, &est, 0.001);
        }
        Ok(())
    }

    #[test]
    fn test_mul() -> Result<()> {
        let dims = af::Dim4::new(&[4, 3, 1, 1]);

        let mut g = Graph1::new();
        let a = g.variable(af::randu::<f32>(dims));
        let b = g.variable(af::randu::<f32>(dims));
        let c = g.mul(&a, &b)?;
        let direction = af::constant(1f32, dims);
        let gradients = g.evaluate_gradients_once(c.gid()?, direction.clone())?;
        {
            let grad = gradients.get(a.gid()?).unwrap();
            let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| x * b.data());
            testing::assert_almost_all_equal(&grad, &est, 0.001);
        }
        {
            let grad = gradients.get(b.gid()?).unwrap();
            let est = testing::estimate_gradient(b.data(), &direction, 0.001f32, |x| a.data() * x);
            testing::assert_almost_all_equal(&grad, &est, 0.001);
        }
        Ok(())
    }
}
