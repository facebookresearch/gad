// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use gad::prelude::*;

#[test]
fn test_gradient_simple() -> Result<()> {
    let mut g = Graph1::new();

    let a = g.variable(3i32);
    let c = {
        let x = g.mul(&a, &a)?;
        g.add(&a, &x)?
    };
    assert_eq!(*c.data(), 3 * 3 + 3);

    let (a, c) = (a.gid()?, c.gid()?);
    let g1 = g.clone().evaluate_gradients_once(c, 1)?;
    let g2 = g.evaluate_gradients_once(c, 2)?;

    assert_eq!(*g1.get(a).unwrap(), 7);
    assert_eq!(*g2.get(a).unwrap(), 14);
    Ok(())
}

#[test]
fn test_hessian_and_more() -> Result<()> {
    let mut g = GraphN::new();

    let x = g.variable(1.0f32);
    let y = g.variable(0.4f32);
    // z = x * y^2
    let z = {
        let h = g.mul(&x, &y)?;
        g.mul(&h, &y)?
    };

    let (x, y, z) = (x.gid()?, y.gid()?, z.gid()?);

    let dz = g.constant(1f32);
    let dz_d = g.compute_gradients(z, dz)?;
    let dz_dx = dz_d.get(x).unwrap();
    let dz_dy = dz_d.get(y).unwrap();

    let ddz = g.constant(1f32);
    let ddz_dxd = g.compute_gradients(dz_dx.gid()?, ddz.clone())?;
    let ddz_dyd = g.compute_gradients(dz_dy.gid()?, ddz)?;

    let ddz_dxdx = ddz_dxd.get(x).map(Value::data);
    let ddz_dxdy = ddz_dxd.get(y).map(Value::data);
    let ddz_dydx = ddz_dyd.get(x).map(Value::data);
    let ddz_dydy = ddz_dyd.get(y).map(Value::data);

    assert_eq!(ddz_dxdx, None);
    assert_eq!(ddz_dxdy, Some(&0.8)); // 2y
    assert_eq!(ddz_dydx, Some(&0.8));
    assert_eq!(ddz_dydy, Some(&2.0));

    let dddz = g.constant(1f32);
    let dddz_dxdyd = g.compute_gradients(ddz_dxd.get(y).unwrap().gid()?, dddz)?;

    let dddz_dxdydx = dddz_dxdyd.get(x).map(Value::data);
    let dddz_dxdydy = dddz_dxdyd.get(y).map(Value::data);
    assert_eq!(dddz_dxdydx, None);
    assert_eq!(dddz_dxdydy, Some(&2.0));
    Ok(())
}

#[cfg(feature = "arrayfire")]
mod af_graph_test {
    use super::*;
    use af::dim4;
    use arrayfire as af;

    #[test]
    fn test_gradient_simple() -> Result<()> {
        let dims = dim4!(2, 2);
        let mut g = Graph1::new();

        let a = g.variable(af::randu::<f32>(dims));
        let c = {
            let x = g.matmul_nn(&a, &a)?;
            g.add(&a, &x)?
        };

        let est = testing::estimate_gradient(a.data(), &af::constant(1f32, dims), 0.001f32, |x| {
            af::matmul(&x, &x, af::MatProp::NONE, af::MatProp::NONE) + x
        });

        // Forgetting the values
        let (a, c) = (a.gid()?, c.gid()?);

        let g1 = g
            .clone()
            .evaluate_gradients_once(c, af::constant(1f32, dims))?;
        let g2 = g.evaluate_gradients_once(c, af::constant(2f32, dims))?;

        let ga1 = g1.get(a).unwrap();
        let ga2 = 0.5f32 * g2.get(a).unwrap();
        testing::assert_almost_all_equal(&ga1, &ga2, 0.001);
        testing::assert_almost_all_equal(&ga1, &est, 0.001);
        Ok(())
    }

    #[test]
    fn test_hessian_and_more() -> Result<()> {
        let dims = dim4!(1);
        let mut g = GraphN::new();

        let x = g.variable(1.0f32);
        let y = g.variable(0.4f32);
        // z = x * y^2
        let z = {
            // Using matrix operations for fun.
            let x = g.constant_as(&x, dims);
            let y = g.constant_as(&y, dims);
            let h = g.matmul_nn(&x, &y)?;
            let i = g.matmul_nn(&h, &y)?;
            g.as_scalar(&i)?
        };

        let (x, y, z) = (x.gid()?, y.gid()?, z.gid()?);

        let dz = g.constant(1f32);
        let dz_d = g.compute_gradients(z, dz)?;
        let dz_dx = dz_d.get(x).unwrap();
        let dz_dy = dz_d.get(y).unwrap();

        let ddz = g.constant(1f32);
        let ddz_dxd = g.compute_gradients(dz_dx.gid()?, ddz.clone())?;
        let ddz_dyd = g.compute_gradients(dz_dy.gid()?, ddz)?;

        let ddz_dxdx = ddz_dxd.get(x).map(Value::data);
        let ddz_dxdy = ddz_dxd.get(y).map(Value::data);
        let ddz_dydx = ddz_dyd.get(x).map(Value::data);
        let ddz_dydy = ddz_dyd.get(y).map(Value::data);

        assert_eq!(ddz_dxdx, None);
        assert_eq!(ddz_dxdy, Some(&0.8)); // 2y
        assert_eq!(ddz_dydx, Some(&0.8));
        assert_eq!(ddz_dydy, Some(&2.0));

        let dddz = g.constant(1f32);
        let dddz_dxdy = g.compute_gradients(ddz_dxd.get(y).unwrap().gid()?, dddz)?;

        let dddz_dxdydx = dddz_dxdy.get(x).map(Value::data);
        let dddz_dxdydy = dddz_dxdy.get(y).map(Value::data);
        assert_eq!(dddz_dxdydx, None);
        assert_eq!(dddz_dxdydy, Some(&2.0));
        Ok(())
    }
}
