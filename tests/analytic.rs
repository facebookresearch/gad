// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

use gad::prelude::*;

#[inline]
fn assert_near(x: f32, y: f32) {
    assert!((x - y).abs() < 0.001);
}

#[test]
fn test_exp() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(1f32);
    let b = {
        let x = g.mulc(&a, 2i16);
        g.exp(&x)
    };
    assert_near(*b.data(), 2f32.exp());
    let direction = 1.7;
    let gradients = g.evaluate_gradients_once(b.gid()?, direction)?;
    assert_near(
        *gradients.get(a.gid()?).unwrap(),
        b.data() * 2.0 * direction,
    );
    Ok(())
}

#[test]
fn test_log() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(1f32);
    let b = {
        let x = g.mulc(&a, 2i16);
        g.log(&x)
    };
    assert_near(*b.data(), 2f32.ln());
    let direction = 1.7;
    let gradients = g.evaluate_gradients_once(b.gid()?, direction)?;
    assert_near(*gradients.get(a.gid()?).unwrap(), 1.0 * direction);
    Ok(())
}

#[test]
fn test_log1p() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(1f32);
    let b = g.log1p(&a);
    assert_near(*b.data(), 2f32.ln());
    let direction = 1.7;
    let gradients = g.evaluate_gradients_once(b.gid()?, direction)?;
    assert_near(*gradients.get(a.gid()?).unwrap(), 0.5 * direction);
    Ok(())
}

#[test]
fn test_sin() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(1f32);
    let b = g.sin(&a);
    assert_near(*b.data(), 1f32.sin());
    let direction = 1.7;
    let gradients = g.evaluate_gradients_once(b.gid()?, direction)?;
    assert_near(*gradients.get(a.gid()?).unwrap(), f32::cos(1.0) * direction);
    Ok(())
}

#[test]
fn test_cos() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(1f32);
    let b = g.cos(&a);
    assert_near(*b.data(), 1f32.cos());
    let direction = 1.7;
    let gradients = g.evaluate_gradients_once(b.gid()?, direction)?;
    assert_near(
        *gradients.get(a.gid()?).unwrap(),
        -f32::sin(1.0) * direction,
    );
    Ok(())
}

#[test]
fn test_tanh() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(1f32);
    let b = g.tanh(&a);
    assert_near(*b.data(), 1f32.tanh());
    let direction = 1.7;
    let gradients = g.evaluate_gradients_once(b.gid()?, direction)?;
    assert_near(
        *gradients.get(a.gid()?).unwrap(),
        (1.0 - b.data() * b.data()) * direction,
    );
    Ok(())
}

#[test]
fn test_sigmoid() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(1f32);
    let b = g.sigmoid(&a);
    assert_near(*b.data(), 1.0 / (1.0 + f32::exp(-1.0)));
    let direction = 1.7;
    let gradients = g.evaluate_gradients_once(b.gid()?, direction)?;
    assert_near(
        *gradients.get(a.gid()?).unwrap(),
        (1.0 - b.data()) * b.data() * direction,
    );
    Ok(())
}

#[test]
fn test_reciprocal() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(2f32);
    let b = g.reciprocal(&a);
    assert_near(*b.data(), 0.5);
    let direction = 1.7;
    let gradients = g.evaluate_gradients_once(b.gid()?, direction)?;
    assert_near(*gradients.get(a.gid()?).unwrap(), -0.25 * direction);
    Ok(())
}

#[test]
fn test_sqrt() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(2f32);
    let b = g.sqrt(&a);
    assert_near(*b.data(), f32::sqrt(2.0));
    let direction = 1.7;
    let gradients = g.evaluate_gradients_once(b.gid()?, direction)?;
    assert_near(
        *gradients.get(a.gid()?).unwrap(),
        0.5 / f32::sqrt(2.0) * direction,
    );
    Ok(())
}

#[test]
fn test_div() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(2f32);
    let b = g.variable(3f32);
    let c = g.div(&a, &b)?;
    assert_near(*c.data(), 2.0 / 3.0);
    let direction = 1.7;
    let gradients = g.evaluate_gradients_once(c.gid()?, direction)?;
    assert_near(*gradients.get(a.gid()?).unwrap(), 1.0 / 3.0 * direction);
    assert_near(*gradients.get(b.gid()?).unwrap(), -2.0 / 9.0 * direction);
    Ok(())
}

#[test]
fn test_pow() -> Result<()> {
    let mut g = Graph1::new();
    let a = g.variable(2f32);
    let b = g.variable(3f32);
    let c = g.pow(&a, &b)?;
    assert_near(*c.data(), 8.0);
    let direction = 1.7;
    let gradients = g.evaluate_gradients_once(c.gid()?, direction)?;
    assert_near(*gradients.get(a.gid()?).unwrap(), 12.0 * direction);
    assert_near(
        *gradients.get(b.gid()?).unwrap(),
        f32::ln(2.0) * c.data() * direction,
    );
    Ok(())
}

#[cfg(feature = "arrayfire")]
mod af_arith_test {
    use super::*;
    use arrayfire as af;

    #[test]
    fn test_exp() -> Result<()> {
        let dims = af::Dim4::new(&[4, 3, 1, 1]);

        let mut g = Graph1::new();
        let a = g.variable(af::randu::<f32>(dims));
        let b = g.exp(&a);
        let direction = af::constant(1f32, dims);
        let gradients = g.evaluate_gradients_once(b.gid()?, direction.clone())?;
        let grad = gradients.get(a.gid()?).unwrap();
        let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| af::exp(x));
        testing::assert_almost_all_equal(&grad, &est, 0.001);
        Ok(())
    }

    #[test]
    fn test_log() -> Result<()> {
        let dims = af::Dim4::new(&[4, 3, 1, 1]);

        let mut g = Graph1::new();
        let a = g.variable(af::randu::<f32>(dims) + 2);
        let b = g.log(&a);
        let direction = af::constant(1f32, dims);
        let gradients = g.evaluate_gradients_once(b.gid()?, direction.clone())?;
        let grad = gradients.get(a.gid()?).unwrap();
        let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| af::log(x));
        testing::assert_almost_all_equal(&grad, &est, 0.001);
        Ok(())
    }

    #[test]
    fn test_log1p() -> Result<()> {
        let dims = af::Dim4::new(&[4, 3, 1, 1]);

        let mut g = Graph1::new();
        let a = g.variable(af::randu::<f32>(dims));
        let b = g.log1p(&a);
        let direction = af::constant(1f32, dims);
        let gradients = g.evaluate_gradients_once(b.gid()?, direction.clone())?;
        let grad = gradients.get(a.gid()?).unwrap();
        let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| af::log1p(x));
        testing::assert_almost_all_equal(&grad, &est, 0.001);
        Ok(())
    }

    #[test]
    fn test_sin() -> Result<()> {
        let dims = af::Dim4::new(&[4, 3, 1, 1]);

        let mut g = Graph1::new();
        let a = g.variable(af::randu::<f32>(dims));
        let b = g.sin(&a);
        let direction = af::constant(1f32, dims);
        let gradients = g.evaluate_gradients_once(b.gid()?, direction.clone())?;
        let grad = gradients.get(a.gid()?).unwrap();
        let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| af::sin(x));
        testing::assert_almost_all_equal(&grad, &est, 0.001);
        Ok(())
    }

    #[test]
    fn test_cos() -> Result<()> {
        let dims = af::Dim4::new(&[4, 3, 1, 1]);

        let mut g = Graph1::new();
        let a = g.variable(af::randu::<f32>(dims));
        let b = g.cos(&a);
        let direction = af::constant(1f32, dims);
        let gradients = g.evaluate_gradients_once(b.gid()?, direction.clone())?;
        let grad = gradients.get(a.gid()?).unwrap();
        let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| af::cos(x));
        testing::assert_almost_all_equal(&grad, &est, 0.001);
        Ok(())
    }

    #[test]
    fn test_tanh() -> Result<()> {
        let dims = af::Dim4::new(&[4, 3, 1, 1]);

        let mut g = Graph1::new();
        let a = g.variable(af::randu::<f32>(dims));
        let b = g.tanh(&a);
        let direction = af::constant(1f32, dims);
        let gradients = g.evaluate_gradients_once(b.gid()?, direction.clone())?;
        let grad = gradients.get(a.gid()?).unwrap();
        let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| af::tanh(x));
        testing::assert_almost_all_equal(&grad, &est, 0.001);
        Ok(())
    }

    #[test]
    fn test_sigmoid() -> Result<()> {
        let dims = af::Dim4::new(&[4, 3, 1, 1]);

        let mut g = Graph1::new();
        let a = g.variable(af::randu::<f32>(dims));
        let b = g.sigmoid(&a);
        let direction = af::constant(1f32, dims);
        let gradients = g.evaluate_gradients_once(b.gid()?, direction.clone())?;
        let grad = gradients.get(a.gid()?).unwrap();
        let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| af::sigmoid(x));
        testing::assert_almost_all_equal(&grad, &est, 0.001);
        Ok(())
    }

    #[test]
    fn test_sqrt() -> Result<()> {
        let dims = af::Dim4::new(&[4, 3, 1, 1]);

        let mut g = Graph1::new();
        let a = g.variable(af::randu::<f32>(dims));
        let b = g.sqrt(&a);
        let direction = af::constant(1f32, dims);
        let gradients = g.evaluate_gradients_once(b.gid()?, direction.clone())?;
        let grad = gradients.get(a.gid()?).unwrap();
        let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| af::sqrt(x));
        testing::assert_almost_all_equal(&grad, &est, 0.001);
        Ok(())
    }

    #[test]
    fn test_div() -> Result<()> {
        let dims = af::Dim4::new(&[4, 3, 1, 1]);

        let mut g = Graph1::new();
        let a = g.variable(af::randu::<f32>(dims));
        let b = g.variable(af::randu::<f32>(dims) + 0.5f32);
        let c = g.div(&a, &b)?;
        let direction = af::constant(1f32, dims);
        let gradients = g.evaluate_gradients_once(c.gid()?, direction.clone())?;
        {
            let grad = gradients.get(a.gid()?).unwrap();
            let est = testing::estimate_gradient(a.data(), &direction, 0.001f32, |x| x / b.data());
            testing::assert_almost_all_equal(&grad, &est, 0.001);
        }
        {
            let grad = gradients.get(b.gid()?).unwrap();
            let est = testing::estimate_gradient(b.data(), &direction, 0.001f32, |x| a.data() / x);
            testing::assert_almost_all_equal(&grad, &est, 0.001);
        }
        Ok(())
    }
}
