// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    arith::ArithAlgebra,
    array::ArrayAlgebra,
    core::{CoreAlgebra, HasDims},
    error::{check_equal_dimensions, Error, Result},
    graph::Value,
    matrix::MatrixAlgebra,
    net::{HasGradientId, HasGradientReader, Net, WeightOps},
    Graph1, Number,
};
use serde::{Deserialize, Serialize};

/// Extensions trait when the network has a single output value (e.g. a single multi-dimensional array).
pub trait SingleOutputNet<Data, Algebra>: Net<Algebra>
where
    Algebra: HasGradientReader + CoreAlgebra<Data, Value = Self::Output>,
{
    /// A network that takes an additional input and returns the L2-distance with
    /// the output of the initial network.
    fn add_square_loss(self) -> SquareLoss<Self, Data>
    where
        Self: Sized,
    {
        SquareLoss(self, std::marker::PhantomData)
    }
}

impl<Data, Algebra, N> SingleOutputNet<Data, Algebra> for N
where
    N: Net<Algebra>,
    Algebra: HasGradientReader + CoreAlgebra<Data, Value = Self::Output>,
{
}

/// Extension trait when the algebra is [`crate::Graph1`] and the output is a scalar.
pub trait DiffNet<T>: Net<Graph1, Output = Value<T>>
where
    T: Number,
    Self::Weights: WeightOps<T>,
{
    /// Apply a "mini-batch" gradient step.
    /// * `Self::Output = Value<T>` is a scalar value representing the error.
    /// * `lambda` is expected to be negative for loss minimization.
    fn apply_gradient_step(&mut self, lambda: T, batch: Vec<Self::Input>) -> Result<T> {
        let mut delta: Option<Self::Weights> = None;
        let mut cumulated_output: Option<T> = None;
        for example in batch {
            // Forward pass
            let mut g = Graph1::new();
            let (output, info) = self.eval_with_gradient_info(&mut g, example)?;
            match &mut cumulated_output {
                opt @ None => *opt = Some(*output.data()),
                Some(val) => *val = *val + *output.data(),
            }
            // Backward pass
            let store = g.evaluate_gradients_once(output.gid()?, T::one())?;
            // Accumulate gradient.
            let gradients = self.read_weight_gradients(info, &store)?;
            match &mut delta {
                opt @ None => *opt = Some(gradients.scale(lambda)),
                Some(val) => val.add_assign(gradients.scale(lambda))?,
            }
        }
        // Update weights.
        if let Some(delta) = delta {
            self.update_weights(delta)?;
        }
        // Report cumulated error
        cumulated_output.ok_or_else(|| Error::empty(func_name!()))
    }
}

impl<N, T> DiffNet<T> for N
where
    T: Number,
    N: Net<Graph1, Output = Value<T>>,
    N::Weights: WeightOps<T>,
{
}

/// The result of [`SingleOutputNet::add_square_loss`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SquareLoss<N, Data>(N, std::marker::PhantomData<Data>);

impl<Data, Algebra, N> Net<Algebra> for SquareLoss<N, Data>
where
    Algebra: HasGradientReader
        + CoreAlgebra<Data, Value = N::Output>
        + ArrayAlgebra<N::Output>
        + ArithAlgebra<N::Output>
        + MatrixAlgebra<N::Output>,
    N: Net<Algebra>,
    Data: HasDims,
    N::Output: HasDims<Dims = Data::Dims>,
    Data::Dims: Clone + PartialEq + std::fmt::Debug,
{
    type Input = (N::Input, Data);
    type Output = <Algebra as ArrayAlgebra<N::Output>>::Scalar;
    type Weights = N::Weights;
    type GradientInfo = N::GradientInfo;

    fn eval_with_gradient_info(
        &self,
        graph: &mut Algebra,
        input: Self::Input,
    ) -> Result<(Self::Output, Self::GradientInfo)> {
        let (output, info) = self.0.eval_with_gradient_info(graph, input.0)?;
        check_equal_dimensions(
            "eval_with_gradient_info",
            &[&output.dims(), &input.1.dims()],
        )?;
        let target = graph.constant(input.1);
        let delta = graph.sub(&target, &output)?;
        let loss = graph.norm2(&delta);
        Ok((loss, info))
    }

    fn get_weights(&self) -> Self::Weights {
        self.0.get_weights()
    }

    fn set_weights(&mut self, weights: Self::Weights) -> Result<()> {
        self.0.set_weights(weights)
    }

    fn update_weights(&mut self, delta: Self::Weights) -> Result<()> {
        self.0.update_weights(delta)
    }

    fn read_weight_gradients(
        &self,
        info: Self::GradientInfo,
        store: &Algebra::GradientReader,
    ) -> Result<Self::Weights> {
        self.0.read_weight_gradients(info, store)
    }
}
