// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

use gad::prelude::*;

#[test]
fn test_add() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(3i32);
    let b = g.variable(2i32);
    let c = g.add(&a, &b)?;
    let gradients = g.evaluate_gradients_once(c.gid()?, 5i32)?;
    assert_eq!(*gradients.get(a.gid()?).unwrap(), 5);
    assert_eq!(*gradients.get(b.gid()?).unwrap(), 5);
    Ok(())
}

#[test]
#[allow(clippy::many_single_char_names)]
fn test_add_all() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(1i32);
    let b = g.variable(2i32);
    let c = g.constant(3i32);
    let d = g.add_all(&[&a, &b, &c])?;
    assert_eq!(*d.data(), 6i32);
    let gradients = g.evaluate_gradients_once(d.gid()?, 1)?;
    assert_eq!(*gradients.get(a.gid()?).unwrap(), 1);
    assert_eq!(*gradients.get(b.gid()?).unwrap(), 1);
    Ok(())
}

#[cfg(feature = "arrayfire")]
mod af_core_test {

    use super::*;
    use arrayfire as af;

    #[test]
    fn test_add() -> Result<()> {
        let dims = af::Dim4::new(&[4, 3, 1, 1]);

        let mut g = Graph1::new();
        let a = g.variable(af::randu::<f32>(dims));
        let b = g.variable(af::randu::<f32>(dims));
        let c = g.add(&a, &b)?;
        let direction = af::constant(1f32, dims);
        let gradients = g.evaluate_gradients_once(c.gid()?, direction.clone())?;
        {
            let grad = gradients.get(a.gid()?).unwrap();
            let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| x + b.data());
            testing::assert_almost_all_equal(&grad, &est, 0.001);
        }
        {
            let grad = gradients.get(b.gid()?).unwrap();
            let est = testing::estimate_gradient(b.data(), &direction, 0.001f32, |x| a.data() + x);
            testing::assert_almost_all_equal(&grad, &est, 0.001);
        }
        Ok(())
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn test_add_all() -> Result<()> {
        let dims = af::Dim4::new(&[4, 3, 1, 1]);

        let mut g = Graph1::new();
        let a = g.variable(af::randu::<f32>(dims));
        let b = g.variable(af::randu::<f32>(dims));
        let c = g.constant(af::randu::<f32>(dims));
        let d = g.add_all(&[&a, &b, &c])?;
        let direction = af::constant(1f32, dims);
        let gradients = g.evaluate_gradients_once(d.gid()?, direction.clone())?;
        {
            let grad = gradients.get(a.gid()?).unwrap();
            let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| {
                x + b.data() + c.data()
            });
            testing::assert_almost_all_equal(&grad, &est, 0.001);
        }
        {
            let grad = gradients.get(b.gid()?).unwrap();
            let est = testing::estimate_gradient(b.data(), &direction, 0.001f32, |x| {
                x + a.data() + c.data()
            });
            testing::assert_almost_all_equal(&grad, &est, 0.001);
        }
        Ok(())
    }
}
