// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]
#![cfg(feature = "arrayfire")]

use arrayfire as af;
use gad::prelude::*;

#[test]
fn test_max_as() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(af::randu::<f32>(af::dim4!(4, 3)));
    let b = g.max_as(&a, af::dim4!(1, 3))?;
    let direction = af::constant(1f32, af::dim4!(1, 3));
    let gradients = g.evaluate_gradients_once(b.gid()?, direction.clone())?;

    let grad = gradients.get(a.gid()?).unwrap();
    let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| af::max(x, 0));
    testing::assert_almost_all_equal(&grad, &est, 0.001);
    Ok(())
}

#[test]
fn test_argmax_as() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(af::randu::<f32>(af::dim4!(4, 3)));

    let b = g.argmax_as(&a, af::dim4!(1, 3))?;
    let c = g.sum_as(&b, af::dim4!(1))?;
    let d = g.as_scalar(&c)?;
    assert_eq!(b.id(), None);
    assert!((d.data() - 3.0).abs() < f32::EPSILON);

    let b = g.argmax_as(&a, af::dim4!(4, 1))?;
    let c = g.sum_as(&b, af::dim4!(1))?;
    let d = g.as_scalar(&c)?;
    assert_eq!(b.id(), None);
    assert!((d.data() - 4.0).abs() < f32::EPSILON);

    let b = g.argmax_as(&a, af::dim4!(1))?;
    let c = g.sum_as(&b, af::dim4!(1))?;
    let d = g.as_scalar(&c)?;
    assert_eq!(b.id(), None);
    assert!((d.data() - 1.0).abs() < f32::EPSILON);
    Ok(())
}

#[test]
fn test_softmax_as() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(af::randu::<f32>(af::dim4!(4, 3)));

    let b = g.softmax_as(&a, af::dim4!(1, 3))?;
    let c = g.sum_as(&b, af::dim4!(1))?;
    let d = g.as_scalar(&c)?;
    assert!((d.data() - 3.0).abs() < f32::EPSILON);

    let direction = af::constant(2f32, af::dim4!(4, 3));
    let gradients = g.evaluate_gradients(b.gid()?, direction.clone())?;
    let grad = gradients.get(a.gid()?).unwrap();
    let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| {
        Eval::default().softmax_as(&x, af::dim4!(1, 3)).unwrap()
    });
    testing::assert_almost_all_equal(&grad, &est, 0.001);

    let b = g.softmax_as(&a, af::dim4!(4, 1))?;
    let c = g.sum_as(&b, af::dim4!(1))?;
    let d = g.as_scalar(&c)?;
    assert!((d.data() - 4.0).abs() < f32::EPSILON);

    let b = g.softmax_as(&a, af::dim4!(1))?;
    let c = g.sum_as(&b, af::dim4!(1))?;
    let d = g.as_scalar(&c)?;
    assert!((d.data() - 1.0).abs() < f32::EPSILON);
    Ok(())
}
