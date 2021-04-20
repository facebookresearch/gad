// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "arrayfire")]

use af::dim4;
use arrayfire as af;
use gad::prelude::*;

#[test]
fn test_transpose() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(af::randu::<f32>(dim4!(4, 3)));
    let b = g.transpose(&a, false)?;
    let direction = af::constant(1f32, dim4!(3, 4));
    let gradients = g.evaluate_gradients_once(b.gid()?, direction.clone())?;

    let grad = gradients.get(a.gid()?).unwrap();
    let est =
        testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| af::transpose(x, false));
    testing::assert_almost_all_equal(&grad, &est, 0.001);
    Ok(())
}

#[test]
fn test_matmul() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(af::randu::<f32>(dim4!(4, 3)));
    let b = g.variable(af::randu::<f32>(dim4!(3, 5)));
    let c = g.matmul_nn(&a, &b)?;
    let direction = af::constant(1f32, dim4!(4, 5));
    let gradients = g.evaluate_gradients_once(c.gid()?, direction.clone())?;
    {
        let grad = gradients.get(a.gid()?).unwrap();
        let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| {
            af::matmul(x, b.data(), af::MatProp::NONE, af::MatProp::NONE)
        });
        testing::assert_almost_all_equal(&grad, &est, 0.002);
    }
    {
        let grad = gradients.get(b.gid()?).unwrap();
        let est = testing::estimate_gradient(b.data(), &direction, 0.001f32, |x| {
            af::matmul(a.data(), x, af::MatProp::NONE, af::MatProp::NONE)
        });
        testing::assert_almost_all_equal(&grad, &est, 0.002);
    }
    Ok(())
}
