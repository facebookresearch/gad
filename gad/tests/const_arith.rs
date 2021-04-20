// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

use gad::prelude::*;

#[test]
fn test_addc() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(2i32);
    let b = g.addc(&a, 1u8);
    assert_eq!(*b.data(), 3);
    let gradients = g.evaluate_gradients_once(b.gid()?, 1)?;
    assert_eq!(*gradients.get(a.gid()?).unwrap(), 1);
    Ok(())
}

#[test]
fn test_mulc() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(2i32);
    let b = g.mulc(&a, 3u8);
    assert_eq!(*b.data(), 6);
    let gradients = g.evaluate_gradients_once(b.gid()?, 1)?;
    assert_eq!(*gradients.get(a.gid()?).unwrap(), 3);
    Ok(())
}

#[test]
fn test_powc() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(2i32);
    let b = g.powc(&a, 3u8);
    assert_eq!(*b.data(), 8);
    let gradients = g.evaluate_gradients_once(b.gid()?, 1)?;
    assert_eq!(*gradients.get(a.gid()?).unwrap(), 12);
    Ok(())
}

#[cfg(feature = "arrayfire")]
mod af_arith_test {
    use super::*;
    use arrayfire as af;

    #[test]
    fn test_addc() -> Result<()> {
        let dims = af::Dim4::new(&[4, 3, 1, 1]);

        let mut g = Graph1::new();
        let a = g.variable(af::randu::<f32>(dims));
        let b = g.addc(&a, 4f32);
        let direction = af::constant(1f32, dims);
        let gradients = g.evaluate_gradients_once(b.gid()?, direction.clone())?;
        let grad = gradients.get(a.gid()?).unwrap();
        // eps = 0.001f32 is not precise enough in some configurations.
        let est =
            testing::estimate_gradient(a.data(), &direction, 0.01f32, |x| af::add(x, &4, false));
        testing::assert_almost_all_equal(&grad, &est, 0.001);
        Ok(())
    }

    #[test]
    fn test_mulc() -> Result<()> {
        let dims = af::Dim4::new(&[4, 3, 1, 1]);

        let mut g = Graph1::new();
        let a = g.variable(af::randu::<f32>(dims));
        let b = g.mulc(&a, 4i16);
        let direction = af::constant(1f32, dims);
        let gradients = g.evaluate_gradients_once(b.gid()?, direction.clone())?;
        let grad = gradients.get(a.gid()?).unwrap();
        let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| x * 4);
        testing::assert_almost_all_equal(&grad, &est, 0.001);
        Ok(())
    }

    #[test]
    fn test_powc() -> Result<()> {
        let dims = af::Dim4::new(&[4, 3, 1, 1]);

        let mut g = Graph1::new();
        let a = g.variable(af::randu::<f32>(dims));
        let b = g.powc(&a, 4i16);
        let direction = af::constant(1f32, dims);
        let gradients = g.evaluate_gradients_once(b.gid()?, direction.clone())?;
        let grad = gradients.get(a.gid()?).unwrap();
        let est =
            testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| af::pow(x, &4, false));
        testing::assert_almost_all_equal(&grad, &est, 0.001);
        Ok(())
    }
}
