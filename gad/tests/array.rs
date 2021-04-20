// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "arrayfire")]

use arrayfire as af;
use gad::prelude::*;

#[test]
fn test_flat() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(af::randu::<f32>(af::dim4!(4, 3)));
    let b = g.flat(&a);
    let direction = af::constant(1f32, af::dim4!(12));
    let gradients = g.evaluate_gradients_once(b.gid()?, direction.clone())?;

    let grad = gradients.get(a.gid()?).unwrap();
    let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| af::flat(x));
    testing::assert_almost_all_equal(&grad, &est, 0.001);
    Ok(())
}

#[test]
fn test_moddims() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(af::randu::<f32>(af::dim4!(4, 3)));
    let b = g.moddims(&a, af::dim4!(2, 2, 3))?;
    let direction = af::constant(1f32, af::dim4!(2, 2, 3));
    let gradients = g.evaluate_gradients_once(b.gid()?, direction.clone())?;

    let grad = gradients.get(a.gid()?).unwrap();
    let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| {
        af::moddims(x, af::dim4!(2, 2, 3))
    });
    testing::assert_almost_all_equal(&grad, &est, 0.001);
    Ok(())
}

#[test]
fn test_sum_as() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(af::randu::<f32>(af::dim4!(4, 3)));
    let b = g.sum_as(&a, af::dim4!(1, 3))?;
    let direction = af::constant(1f32, af::dim4!(1, 3));
    let gradients = g.evaluate_gradients_once(b.gid()?, direction.clone())?;

    let grad = gradients.get(a.gid()?).unwrap();
    let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| af::sum(x, 0));
    testing::assert_almost_all_equal(&grad, &est, 0.001);
    Ok(())
}

#[test]
fn test_tile_as() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(af::randu::<f32>(af::dim4!(1, 3)));
    let rdims = af::dim4!(4, 3);
    let b = g.tile_as(&a, rdims)?;
    let direction = af::constant(1f32, rdims);
    let gradients = g.evaluate_gradients_once(b.gid()?, direction.clone())?;

    let grad = gradients.get(a.gid()?).unwrap();
    let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| {
        af::tile(x, af::dim4!(4))
    });
    testing::assert_almost_all_equal(&grad, &est, 0.001);
    Ok(())
}

#[test]
fn test_norm2() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(af::randu::<f32>(af::dim4!(4, 3)));
    let b = {
        let c = g.norm2(&a);
        g.constant_as(&c, af::dim4!(1))
    };
    let direction = af::constant(1f32, af::dim4!(1));
    let gradients = g.evaluate_gradients_once(b.gid()?, direction.clone())?;

    let grad = gradients.get(a.gid()?).unwrap();
    let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| {
        af::constant(Eval::default().norm2(x), af::dim4!(1))
    });
    testing::assert_almost_all_equal(&grad, &est, 0.001);
    Ok(())
}
