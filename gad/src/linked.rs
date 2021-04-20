// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::graph::{Config, Graph, Value};

/// How to reference values from another algebra.
/// This is needed for higher-order differentials in order to propagate
/// gradients all the way to the initial variables.
pub trait LinkedAlgebra<SourceValue, TargetValue> {
    fn link<'a>(&mut self, value: &'a SourceValue) -> &'a TargetValue;
}

/// Ignore the link when evaluating pure arrays.
impl<D, A> LinkedAlgebra<Value<D>, D> for A
where
    A: crate::core::CoreAlgebra<D, Value = D>,
{
    #[inline]
    fn link<'a>(&mut self, value: &'a Value<D>) -> &'a D {
        value.data()
    }
}

/// Assume that we link into a copy of the original graph.
impl<V, C: Config> LinkedAlgebra<V, V> for Graph<C> {
    #[inline]
    fn link<'a>(&mut self, value: &'a V) -> &'a V {
        value
    }
}
