// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    core::{CoreAlgebra, HasDims},
    error::{check_equal_dimensions, check_equal_lengths, Error, Result},
    graph,
    store::GradientReader,
    Check, Eval,
};
use serde::{Deserialize, Serialize};

#[cfg(doc)]
use crate::prelude::*;

/// A value with a gradient-id.
/// * This is possibly a placeholder in the case of algebras without gradients.
/// * Used to define [`Net::GradientInfo`] in a generic way.
pub trait HasGradientId {
    type GradientId;

    fn gid(&self) -> Result<Self::GradientId>;
}

/// Associate a gradient reader to a [`CoreAlgebra`].
/// * The gradient reader is possibly a placeholder in the case of algebras without gradients.
/// * Used to define [`Net::read_weight_gradients`] in a generic way.
pub trait HasGradientReader {
    type GradientReader;
}

/// A Neural Network over an algebra of operations.
pub trait Net<Algebra: HasGradientReader> {
    /// Input of the network.
    type Input;
    /// Output of the network.
    type Output;
    /// External representation for the weights of the network.
    type Weights;
    /// How to read the gradients of the weights after a backward pass.
    type GradientInfo;

    fn eval_with_gradient_info(
        &self,
        graph: &mut Algebra,
        input: Self::Input,
    ) -> Result<(Self::Output, Self::GradientInfo)>;

    fn get_weights(&self) -> Self::Weights;

    fn update_weights(&mut self, delta: Self::Weights) -> Result<()>;

    fn set_weights(&mut self, weight: Self::Weights) -> Result<()>;

    fn read_weight_gradients(
        &self,
        info: Self::GradientInfo,
        reader: &Algebra::GradientReader,
    ) -> Result<Self::Weights>;

    fn eval(&self, graph: &mut Algebra, input: Self::Input) -> Result<Self::Output> {
        Ok(self.eval_with_gradient_info(graph, input)?.0)
    }

    fn map<F, O>(self, f: F) -> Map<Self, F>
    where
        Self: Sized,
        F: Fn(&mut Algebra, Self::Output) -> Result<O>,
    {
        Map(self, f)
    }

    fn using<N>(self, net: N) -> Using<Self, N>
    where
        Self: Sized,
        N: Net<Algebra, Input = ()>,
    {
        Using(self, net)
    }

    fn then<N>(self, net: N) -> Then<Self, N>
    where
        Self: Sized,
        N: Net<Algebra, Input = Self::Output>,
    {
        Then(self, net)
    }

    fn and<N>(self, net: N) -> (Self, N)
    where
        Self: Sized,
        N: Net<Algebra>,
    {
        (self, net)
    }
}

/// Operations supported by weight types [`Net::Weights`]
// TODO: add Debug when af::Array supports it.
pub trait WeightOps<T>: serde::Serialize + serde::de::DeserializeOwned + Clone + Sized {
    fn add_assign(&mut self, other: Self) -> Result<()>;

    fn scale(&self, lambda: T) -> Self;
}

impl<C: graph::Config> HasGradientReader for graph::Graph<C> {
    type GradientReader = C::GradientStore;
}

impl HasGradientReader for Eval {
    type GradientReader = crate::store::EmptyGradientMap;
}

impl HasGradientReader for Check {
    type GradientReader = crate::store::EmptyGradientMap;
}

/// Extensions trait when the algebra is [`crate::Eval`].
pub trait EvalNet: Net<Eval> {
    /// Run a forward pass to evaluate the network on the given input.
    fn evaluate(&self, input: Self::Input) -> Result<Self::Output> {
        self.eval(&mut Eval::default(), input)
    }
}

impl<N> EvalNet for N where N: Net<Eval> {}

/// Extensions trait when the algebra is [`crate::Check`].
pub trait CheckNet: Net<Check> {
    /// Run a forward pass that only checks dimensions.
    fn check(&self, input: Self::Input) -> Result<Self::Output> {
        self.eval(&mut Check::default(), input)
    }
}

impl<N> CheckNet for N where N: Net<Check> {}

#[cfg(feature = "arrayfire")]
mod af_net {
    use super::*;
    use arrayfire as af;

    impl<T: af::HasAfEnum> HasGradientId for af::Array<T> {
        type GradientId = ();

        #[inline]
        fn gid(&self) -> Result<Self::GradientId> {
            Ok(())
        }
    }

    impl HasGradientId for af::Dim4 {
        type GradientId = ();

        #[inline]
        fn gid(&self) -> Result<Self::GradientId> {
            Ok(())
        }
    }

    impl<T> WeightOps<T> for af::Array<T>
    where
        T: af::HasAfEnum
            + Default
            + Copy
            + serde::Serialize
            + serde::de::DeserializeOwned
            + std::fmt::Debug
            + af::ConstGenerator<OutType = T>,
    {
        fn add_assign(&mut self, other: Self) -> Result<()> {
            check_equal_dimensions(func_name!(), &[&other.dims(), &self.dims()])?;
            *self += other;
            Ok(())
        }

        fn scale(&self, lambda: T) -> Self {
            self * lambda
        }
    }
}

impl<A> HasGradientId for graph::Value<A> {
    type GradientId = crate::store::GradientId<A>;

    #[inline]
    fn gid(&self) -> Result<Self::GradientId> {
        self.id().ok_or_else(|| Error::missing_id(func_name!()))
    }
}

/// A network that takes a single user data as input and returns it (after a dimension check).
#[derive(Debug, Clone)]
pub struct InputData<Data, Algebra>
where
    Data: HasDims,
{
    dims: Data::Dims,
    marker: std::marker::PhantomData<(Data, Algebra)>,
}

/// A network that takes no inputs and always returns the same data.
#[derive(Debug, Clone)]
pub struct ConstantData<Data, Algebra> {
    data: Data,
    marker: std::marker::PhantomData<Algebra>,
}

/// A network that takes no inputs and always returns the weights.
#[derive(Debug, Clone)]
pub struct WeightData<Data, Algebra> {
    data: Data,
    marker: std::marker::PhantomData<Algebra>,
}

impl<Data, Algebra> InputData<Data, Algebra>
where
    Data: HasDims,
{
    pub fn new(dims: Data::Dims) -> Self {
        Self {
            dims,
            marker: std::marker::PhantomData,
        }
    }
}

impl<Data, Algebra> ConstantData<Data, Algebra> {
    pub fn new(data: Data) -> Self {
        Self {
            data,
            marker: std::marker::PhantomData,
        }
    }

    pub fn get(&self) -> &Data {
        &self.data
    }
}

impl<Data, Algebra> WeightData<Data, Algebra> {
    pub fn new(data: Data) -> Self {
        Self {
            data,
            marker: std::marker::PhantomData,
        }
    }

    pub fn get(&self) -> &Data {
        &self.data
    }
}

impl<Data, Value, Dims, Algebra> Net<Algebra> for InputData<Data, Algebra>
where
    Algebra: HasGradientReader + CoreAlgebra<Data, Value = Value>,
    Data: HasDims<Dims = Dims>,
    Dims: Clone + PartialEq + std::fmt::Debug,
{
    type Input = Data;
    type Output = Value;
    type Weights = ();
    type GradientInfo = ();

    fn eval_with_gradient_info(
        &self,
        graph: &mut Algebra,
        input: Self::Input,
    ) -> Result<(Self::Output, Self::GradientInfo)> {
        check_equal_dimensions(func_name!(), &[&input.dims(), &self.dims])?;
        Ok((graph.constant(input), ()))
    }

    fn get_weights(&self) -> Self::Weights {}

    fn set_weights(&mut self, _weights: Self::Weights) -> Result<()> {
        Ok(())
    }

    fn update_weights(&mut self, _delta: Self::Weights) -> Result<()> {
        Ok(())
    }

    fn read_weight_gradients(
        &self,
        _info: Self::GradientInfo,
        _reader: &Algebra::GradientReader,
    ) -> Result<Self::Weights> {
        Ok(())
    }
}

impl<Data, Value, Algebra> Net<Algebra> for ConstantData<Data, Algebra>
where
    Data: Clone,
    Algebra: HasGradientReader + CoreAlgebra<Data, Value = Value>,
{
    type Input = ();
    type Output = Value;
    type Weights = ();
    type GradientInfo = ();

    fn eval_with_gradient_info(
        &self,
        graph: &mut Algebra,
        _input: Self::Input,
    ) -> Result<(Self::Output, Self::GradientInfo)> {
        Ok((graph.constant(self.data.clone()), ()))
    }

    fn get_weights(&self) -> Self::Weights {}

    fn set_weights(&mut self, _weights: Self::Weights) -> Result<()> {
        Ok(())
    }

    fn update_weights(&mut self, _delta: Self::Weights) -> Result<()> {
        Ok(())
    }

    fn read_weight_gradients(
        &self,
        _info: Self::GradientInfo,
        _reader: &Algebra::GradientReader,
    ) -> Result<Self::Weights> {
        Ok(())
    }
}

impl<Data, Value, Algebra> Net<Algebra> for WeightData<Data, Algebra>
where
    Algebra: HasGradientReader + CoreAlgebra<Data, Value = Value>,
    Data: Clone + HasDims + std::ops::AddAssign,
    Value: HasGradientId,
    Data::Dims: Clone + PartialEq + std::fmt::Debug,
    Algebra::GradientReader: GradientReader<Value::GradientId, Data>,
{
    type Input = ();
    type Output = Value;
    type Weights = Data;
    type GradientInfo = Value::GradientId;

    fn eval_with_gradient_info(
        &self,
        graph: &mut Algebra,
        _input: Self::Input,
    ) -> Result<(Self::Output, Self::GradientInfo)> {
        let value = graph.variable(self.data.clone());
        let id = value.gid()?;
        Ok((value, id))
    }

    fn get_weights(&self) -> Self::Weights {
        self.data.clone()
    }

    fn set_weights(&mut self, weights: Self::Weights) -> Result<()> {
        check_equal_dimensions(func_name!(), &[&weights.dims(), &self.data.dims()])?;
        self.data = weights;
        Ok(())
    }

    fn update_weights(&mut self, delta: Self::Weights) -> Result<()> {
        check_equal_dimensions(func_name!(), &[&delta.dims(), &self.data.dims()])?;
        self.data += delta;
        Ok(())
    }

    fn read_weight_gradients(
        &self,
        info: Self::GradientInfo,
        reader: &Algebra::GradientReader,
    ) -> Result<Self::Weights> {
        let data = reader
            .read(info)
            .ok_or_else(|| Error::missing_gradient(func_name!()))?
            .clone();
        Ok(data)
    }
}

/// The result of [`Net::map`]
#[derive(Debug, Clone)]
pub struct Map<N, F>(N, F);

impl<Algebra, N, F, O> Net<Algebra> for Map<N, F>
where
    Algebra: HasGradientReader,
    N: Net<Algebra>,
    F: Fn(&mut Algebra, N::Output) -> Result<O>,
{
    type Input = N::Input;
    type Output = O;
    type Weights = N::Weights;
    type GradientInfo = N::GradientInfo;

    fn eval_with_gradient_info(
        &self,
        graph: &mut Algebra,
        input: Self::Input,
    ) -> Result<(Self::Output, Self::GradientInfo)> {
        let (output, info) = self.0.eval_with_gradient_info(graph, input)?;
        Ok(((self.1)(graph, output)?, info))
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
        reader: &Algebra::GradientReader,
    ) -> Result<Self::Weights> {
        self.0.read_weight_gradients(info, reader)
    }
}

/// The result of [`Net::then`]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Then<N1, N2>(N1, N2);

impl<Algebra, N1, N2> Net<Algebra> for Then<N1, N2>
where
    Algebra: HasGradientReader,
    N1: Net<Algebra>,
    N2: Net<Algebra, Input = N1::Output>,
{
    type Input = N1::Input;
    type Output = N2::Output;
    type Weights = Then<N1::Weights, N2::Weights>;
    type GradientInfo = Then<N1::GradientInfo, N2::GradientInfo>;

    fn eval_with_gradient_info(
        &self,
        graph: &mut Algebra,
        input: Self::Input,
    ) -> Result<(Self::Output, Self::GradientInfo)> {
        let (output0, info0) = self.0.eval_with_gradient_info(graph, input)?;
        let (output1, info1) = self.1.eval_with_gradient_info(graph, output0)?;
        Ok((output1, Then(info0, info1)))
    }

    fn get_weights(&self) -> Self::Weights {
        Then(self.0.get_weights(), self.1.get_weights())
    }

    fn set_weights(&mut self, weights: Self::Weights) -> Result<()> {
        self.0.set_weights(weights.0)?;
        self.1.set_weights(weights.1)
    }

    fn update_weights(&mut self, delta: Self::Weights) -> Result<()> {
        self.0.update_weights(delta.0)?;
        self.1.update_weights(delta.1)
    }

    fn read_weight_gradients(
        &self,
        info: Self::GradientInfo,
        reader: &Algebra::GradientReader,
    ) -> Result<Self::Weights> {
        Ok(Then(
            self.0.read_weight_gradients(info.0, reader)?,
            self.1.read_weight_gradients(info.1, reader)?,
        ))
    }
}

impl<T, W1, W2> WeightOps<T> for Then<W1, W2>
where
    T: Copy,
    W1: WeightOps<T>,
    W2: WeightOps<T>,
{
    fn add_assign(&mut self, other: Self) -> Result<()> {
        self.0.add_assign(other.0)?;
        self.1.add_assign(other.1)
    }

    fn scale(&self, rhs: T) -> Self {
        Then(self.0.scale(rhs), self.1.scale(rhs))
    }
}

/// The result of [`Net::using`]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Using<N1, N2>(pub N1, pub N2);

impl<Algebra, N1, N2> Net<Algebra> for Using<N1, N2>
where
    Algebra: HasGradientReader,
    N1: Net<Algebra>,
    N2: Net<Algebra, Input = ()>,
{
    type Input = N1::Input;
    type Output = (N1::Output, N2::Output);
    type Weights = Using<N1::Weights, N2::Weights>;
    type GradientInfo = Using<N1::GradientInfo, N2::GradientInfo>;

    fn eval_with_gradient_info(
        &self,
        graph: &mut Algebra,
        input: Self::Input,
    ) -> Result<(Self::Output, Self::GradientInfo)> {
        let (output0, info0) = self.0.eval_with_gradient_info(graph, input)?;
        let (output1, info1) = self.1.eval_with_gradient_info(graph, ())?;
        Ok(((output0, output1), Using(info0, info1)))
    }

    fn get_weights(&self) -> Self::Weights {
        Using(self.0.get_weights(), self.1.get_weights())
    }

    fn set_weights(&mut self, weights: Self::Weights) -> Result<()> {
        self.0.set_weights(weights.0)?;
        self.1.set_weights(weights.1)
    }

    fn update_weights(&mut self, delta: Self::Weights) -> Result<()> {
        self.0.update_weights(delta.0)?;
        self.1.update_weights(delta.1)
    }

    fn read_weight_gradients(
        &self,
        info: Self::GradientInfo,
        reader: &Algebra::GradientReader,
    ) -> Result<Self::Weights> {
        Ok(Using(
            self.0.read_weight_gradients(info.0, reader)?,
            self.1.read_weight_gradients(info.1, reader)?,
        ))
    }
}

impl<T, W1, W2> WeightOps<T> for Using<W1, W2>
where
    T: Copy,
    W1: WeightOps<T>,
    W2: WeightOps<T>,
{
    fn add_assign(&mut self, other: Self) -> Result<()> {
        self.0.add_assign(other.0)?;
        self.1.add_assign(other.1)
    }

    fn scale(&self, rhs: T) -> Self {
        Using(self.0.scale(rhs), self.1.scale(rhs))
    }
}

macro_rules! impl_net_tuple {
        ( $($name:ident $idx:tt)*) => (
impl<Algebra: HasGradientReader, $($name: Net<Algebra>),*> Net<Algebra> for ($($name,)*)
{
    type Input = ($($name::Input,)*);
    type Output = ($($name::Output,)*);
    type Weights = ($($name::Weights,)*);
    type GradientInfo = ($($name::GradientInfo,)*);

    #[allow(non_snake_case)]
    fn eval_with_gradient_info(
        &self,
        _graph: &mut Algebra,
        _input: Self::Input,
    ) -> Result<(Self::Output, Self::GradientInfo)> {
        $(let $name = self.$idx.eval_with_gradient_info(_graph, _input.$idx)?;)*
        let output = ($($name.0,)*);
        let info = ($($name.1,)*);
        Ok((output, info))
    }

    fn get_weights(&self) -> Self::Weights { ($(self.$idx.get_weights(),)*) }

    fn set_weights(&mut self, _weights: Self::Weights) -> Result<()> {
        $(self.$idx.set_weights(_weights.$idx)?;)*
        Ok(())
    }

    fn update_weights(&mut self, _delta: Self::Weights) -> Result<()> {
        $(self.$idx.update_weights(_delta.$idx)?;)*
        Ok(())
    }

    fn read_weight_gradients(
        &self,
        _info: Self::GradientInfo,
        _reader: &Algebra::GradientReader,
    ) -> Result<Self::Weights> {
        Ok(($(self.$idx.read_weight_gradients(_info.$idx, _reader)?,)*))
    }
}

impl<T, $($name),*> WeightOps<T> for ($($name,)*)
where
    T: Copy,
    $($name: WeightOps<T>),*
{
    fn add_assign(&mut self, _other: Self) -> Result<()> {
        $(self.$idx.add_assign(_other.$idx)?;)*
        Ok(())
    }

    fn scale(&self, _rhs: T) -> Self {
        ($(self.$idx.scale(_rhs),)*)
    }
}
)}

impl_net_tuple! {}
impl_net_tuple! { A 0 }
impl_net_tuple! { A 0 B 1 }
impl_net_tuple! { A 0 B 1 C 2 }
impl_net_tuple! { A 0 B 1 C 2 D 3 }
impl_net_tuple! { A 0 B 1 C 2 D 3 E 4 }
impl_net_tuple! { A 0 B 1 C 2 D 3 E 4 F 5 }
impl_net_tuple! { A 0 B 1 C 2 D 3 E 4 F 5 G 6 }
impl_net_tuple! { A 0 B 1 C 2 D 3 E 4 F 5 G 6 H 7 }
impl_net_tuple! { A 0 B 1 C 2 D 3 E 4 F 5 G 6 H 7 I 8 }
impl_net_tuple! { A 0 B 1 C 2 D 3 E 4 F 5 G 6 H 7 I 8 J 9}

impl<Algebra, N> Net<Algebra> for Vec<N>
where
    Algebra: HasGradientReader,
    N: Net<Algebra>,
{
    type Input = Vec<N::Input>;
    type Output = Vec<N::Output>;
    type Weights = Vec<N::Weights>;
    type GradientInfo = Vec<N::GradientInfo>;

    fn eval_with_gradient_info(
        &self,
        graph: &mut Algebra,
        input: Self::Input,
    ) -> Result<(Self::Output, Self::GradientInfo)> {
        check_equal_lengths(func_name!(), &[self.len(), input.len()])?;
        Ok(input
            .into_iter()
            .enumerate()
            .map(|(i, x)| self[i].eval_with_gradient_info(graph, x))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .unzip())
    }

    fn get_weights(&self) -> Self::Weights {
        self.iter().map(Net::get_weights).collect()
    }

    fn set_weights(&mut self, weights: Self::Weights) -> Result<()> {
        check_equal_lengths(func_name!(), &[self.len(), weights.len()])?;
        weights
            .into_iter()
            .enumerate()
            .try_for_each(|(i, x)| self[i].set_weights(x))
    }

    fn update_weights(&mut self, delta: Self::Weights) -> Result<()> {
        check_equal_lengths(func_name!(), &[self.len(), delta.len()])?;
        delta
            .into_iter()
            .enumerate()
            .try_for_each(|(i, x)| self[i].update_weights(x))
    }

    fn read_weight_gradients(
        &self,
        info: Self::GradientInfo,
        reader: &Algebra::GradientReader,
    ) -> Result<Self::Weights> {
        check_equal_lengths(func_name!(), &[self.len(), info.len()])?;
        info.into_iter()
            .enumerate()
            .map(|(i, x)| self[i].read_weight_gradients(x, reader))
            .collect()
    }
}

impl<N, T> WeightOps<T> for Vec<N>
where
    T: Copy,
    N: WeightOps<T>,
{
    fn add_assign(&mut self, other: Self) -> Result<()> {
        check_equal_lengths(func_name!(), &[self.len(), other.len()])?;
        self.iter_mut()
            .zip(other.into_iter())
            .try_for_each(|(x, y)| x.add_assign(y))
    }

    fn scale(&self, rhs: T) -> Self {
        self.iter().map(|x| x.scale(rhs)).collect()
    }
}
