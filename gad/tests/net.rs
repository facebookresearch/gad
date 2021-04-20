// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "arrayfire")]

use arrayfire as af;
use gad::prelude::*;

struct TestNet<A, T: Float> {
    dims: af::Dim4,
    weights: af::Array<T>,
    marker: std::marker::PhantomData<A>,
}

impl<T, A> TestNet<A, T>
where
    T: Float,
{
    pub fn new(dims: af::Dim4, weights: af::Array<T>) -> Self {
        Self {
            dims,
            weights,
            marker: std::marker::PhantomData,
        }
    }
}

impl<T, A> Net<A> for TestNet<A, T>
where
    T: Float,
    A: AfAlgebra<T>,
{
    type Input = af::Array<T>;
    type Output = <A as AfAlgebra<T>>::Value;
    type Weights = af::Array<T>;
    type GradientInfo = <<A as AfAlgebra<T>>::Value as HasGradientId>::GradientId;

    fn eval_with_gradient_info(
        &self,
        g: &mut A,
        input: Self::Input,
    ) -> Result<(Self::Output, Self::GradientInfo)> {
        assert_eq!(input.dims(), self.dims);
        let input = g.constant(input);
        let weights = g.variable(self.weights.clone());
        let output = g.matmul_nn(&input, &weights)?;
        let id = weights.gid()?;
        Ok((output, id))
    }

    fn get_weights(&self) -> Self::Weights {
        self.weights.clone()
    }

    fn update_weights(&mut self, delta: Self::Weights) -> Result<()> {
        check_equal_dimensions(func_name!(), &[&delta.dims(), &self.weights.dims()])?;
        self.weights += delta;
        Ok(())
    }

    fn set_weights(&mut self, weights: Self::Weights) -> Result<()> {
        check_equal_dimensions(func_name!(), &[&weights.dims(), &self.weights.dims()])?;
        self.weights = weights;
        Ok(())
    }

    fn read_weight_gradients(
        &self,
        info: Self::GradientInfo,
        reader: &<A as HasGradientReader>::GradientReader,
    ) -> Result<Self::Weights> {
        Ok(reader
            .read(info)
            .ok_or_else(|| Error::missing_gradient(func_name!()))?
            .clone())
    }
}

fn make_net<A, T>(
    n: u64,
) -> impl Net<A, Input = af::Array<T>, Output = <A as AfAlgebra<T>>::Value, Weights = impl WeightOps<T>>
where
    T: Float,
    A: AfAlgebra<T>,
{
    let input = InputData::<af::Array<T>, A>::new(af::dim4!(n, n));
    let weight = WeightData::new(af::randn!(T; n, n));
    input.using(weight).map(|g, (i, w)| g.matmul_nn(&i, &w))
}

#[test]
fn test_testnet() -> anyhow::Result<()> {
    let mut train = TestNet::new(af::dim4!(3, 3), af::randn!(f32; 3, 3)).add_square_loss();

    let a = af::Array::<f32>::new(
        &[1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 0.0, -2.0, -1.0],
        af::dim4!(3, 3),
    );
    let i = af::identity(af::dim4!(3, 3));
    let samples = vec![(a.clone(), i.clone())];
    loop {
        let loss = train.apply_gradient_step(-0.01, samples.clone())?;
        assert!(loss.is_finite());
        if loss < 0.000001 {
            break;
        }
    }

    let mut net = TestNet::new(af::dim4!(3, 3), af::randn!(f32; 3, 3));
    net.set_weights(train.get_weights())?;
    let i2 = net.evaluate(a)?;
    testing::assert_almost_all_equal(&i, &i2, 0.01);
    Ok(())
}

#[test]
fn test_make_net() -> anyhow::Result<()> {
    let mut train = make_net(3).add_square_loss();

    let a = af::Array::<f32>::new(
        &[1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 0.0, -2.0, -1.0],
        af::dim4!(3, 3),
    );
    let i = af::identity(af::dim4!(3, 3));
    let samples = vec![(a.clone(), i.clone())];
    loop {
        let loss = train.apply_gradient_step(-0.01, samples.clone())?;
        assert!(loss.is_finite());
        if loss < 0.000001 {
            break;
        }
    }

    // Because we used `impl WeightOps<T>` in the return type of `make_net`,
    // calls with different parameters `A` generates incomparable types.
    let bytes = bincode::serialize(&train.get_weights())?;

    // Note that the type of `weights` is inferred and cannot be written down in Rust
    // at the moment.
    let weights = bincode::deserialize(&bytes)?;
    let mut net = make_net(3);
    net.set_weights(weights)?;
    let i2 = net.evaluate(a.clone())?;
    testing::assert_almost_all_equal(&i, &i2, 0.01);

    // Check dimensions.
    let weights = bincode::deserialize(&bytes)?;
    let mut net = make_net(3);
    net.set_weights(weights)?;
    let i2 = net.check(a)?;
    assert_eq!(i.dims(), i2);

    Ok(())
}
