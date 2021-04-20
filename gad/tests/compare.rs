// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]
#![allow(clippy::unnecessary_wraps)]

use gad::prelude::*;

#[test]
fn test_min() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(1i32);
    let b = g.variable(2i32);
    let c = g.min(&a, &b)?;
    assert_eq!(*c.data(), 1);
    let gradients = g.evaluate_gradients_once(c.gid()?, 1)?;
    assert_eq!(*gradients.get(a.gid()?).unwrap(), 1);
    assert_eq!(*gradients.get(b.gid()?).unwrap(), 0);
    Ok(())
}

#[test]
fn test_max() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(1i32);
    let b = g.variable(2i32);
    let c = g.max(&a, &b)?;
    assert_eq!(*c.data(), 2);
    let gradients = g.evaluate_gradients_once(c.gid()?, 1)?;
    assert_eq!(*gradients.get(a.gid()?).unwrap(), 0);
    assert_eq!(*gradients.get(b.gid()?).unwrap(), 1);
    Ok(())
}

#[test]
fn test_abs() -> Result<()> {
    let mut g = Graph1::new();
    let a1 = g.variable(-1i32);
    let a2 = g.variable(2i32);
    let b1 = g.abs(&a1);
    let b2 = g.abs(&a2);
    assert_eq!(*b1.data(), 1);
    assert_eq!(*b2.data(), 2);
    let gradients = g.evaluate_gradients(b1.gid()?, 3)?;
    assert_eq!(*gradients.get(a1.gid()?).unwrap(), -3);
    let gradients = g.evaluate_gradients(b2.gid()?, 3)?;
    assert_eq!(*gradients.get(a2.gid()?).unwrap(), 3);
    Ok(())
}

#[test]
fn test_relu() -> Result<()> {
    let mut g = Graph1::new();
    let a1 = g.variable(-1i32);
    let a2 = g.variable(2i32);
    let b1 = g.relu(&a1);
    let b2 = g.relu(&a2);
    assert_eq!(*b1.data(), 0);
    assert_eq!(*b2.data(), 2);
    let gradients = g.evaluate_gradients(b1.gid()?, 3)?;
    assert_eq!(*gradients.get(a1.gid()?).unwrap(), 0);
    let gradients = g.evaluate_gradients(b2.gid()?, 3)?;
    assert_eq!(*gradients.get(a2.gid()?).unwrap(), 3);
    Ok(())
}

#[test]
fn test_sign() -> Result<()> {
    let mut g = Graph1::new();
    let a1 = g.variable(-1i32);
    let a2 = g.variable(2i32);
    let b1 = g.sign(&a1);
    let b2 = g.sign(&a2);
    assert_eq!(*b1.data(), -1);
    assert_eq!(b1.id(), None);
    assert_eq!(b2.id(), None);
    Ok(())
}

#[test]
fn test_select_argmax() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(1i32);
    let b = g.variable(2i32);
    let c = g.variable(3i32);
    let d = g.variable(4i32);
    let e = g.select_argmax(&a, &b, Some(&c), Some(&d))?;
    assert_eq!(*e.data(), 4);
    let gradients = g.evaluate_gradients_once(e.gid()?, 1)?;
    assert_eq!(gradients.get(a.gid()?), None);
    assert_eq!(gradients.get(b.gid()?), None);
    assert_eq!(*gradients.get(c.gid()?).unwrap(), 0);
    assert_eq!(*gradients.get(d.gid()?).unwrap(), 1);
    Ok(())
}

#[cfg(feature = "arrayfire")]
mod af_arith_test {
    use super::*;
    use arrayfire as af;

    #[test]
    fn test_max() -> Result<()> {
        let dims = af::Dim4::new(&[4, 3, 1, 1]);

        let mut g = Graph1::new();
        let a = g.variable(af::randu::<f32>(dims));
        let b = g.variable(af::randu::<f32>(dims));
        let c = g.max(&a, &b)?;
        let direction = af::constant(1f32, dims);
        let gradients = g.evaluate_gradients_once(c.gid()?, direction.clone())?;
        {
            let grad = gradients.get(a.gid()?).unwrap();
            let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| {
                af::maxof(x, b.data(), false)
            });
            testing::assert_almost_all_equal(&grad, &est, 0.001);
        }
        {
            let grad = gradients.get(b.gid()?).unwrap();
            let est = testing::estimate_gradient(b.data(), &direction, 0.001f32, |x| {
                af::maxof(a.data(), x, false)
            });
            testing::assert_almost_all_equal(&grad, &est, 0.001);
        }
        Ok(())
    }
}
