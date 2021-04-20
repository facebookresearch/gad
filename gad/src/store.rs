// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{core::CoreAlgebra, error::Result, graph::Value};
use std::collections::BTreeMap;

#[cfg(doc)]
use crate::prelude::*;

/// Index of a computation node tracked in a graph.
/// Note: Offset is non-zero to optimize `std::mem::size_of<Option<Id>>()`
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Id {
    arena_id: u32,
    index: std::num::NonZeroU32,
}

/// A node Id associated with an underlying gradient type `T`.
/// Such an association must assign a unique type to inner Id.
pub struct GradientId<T> {
    pub(crate) inner: Id,
    marker: std::marker::PhantomData<T>,
}

/// Trait for reading gradient values of type `T` given a handle of type `Id`.
/// Value may be converted if needed.
pub trait GradientReader<Id, T> {
    fn read(&self, id: Id) -> Option<&T>;
}

/// Trait for accessing gradient values of type `T` given a handle of type `Id`.
pub trait GradientStore<Id, T>: GradientReader<Id, T> {
    fn insert(&mut self, id: Id, gradient: T);

    fn get(&self, id: Id) -> Option<&T> {
        self.read(id)
    }

    fn get_mut(&mut self, id: Id) -> Option<&mut T>;

    /// Update a gradient during backward propagation. This is used to define operators
    /// together with [`Graph::make_node`].
    /// The parameter `graph` is used for higher-order differentials (see [`GraphN`]).
    fn add_gradient<A, G>(&mut self, graph: &mut G, id: Id, value: &T) -> Result<()>
    where
        G: CoreAlgebra<A, Value = T> + ?Sized,
        Id: Copy,
        T: Clone + 'static,
    {
        match self.get_mut(id) {
            None => self.insert(id, value.clone()),
            Some(current) => *current = graph.add(current, value)?,
        }
        Ok(())
    }
}

/// Gradient store used by [`Graph1`].
/// Indices of type `GradientId<T>` are mapped to values of type `T`.
#[derive(Debug)]
pub struct GenericGradientMap1 {
    values: BTreeMap<Id, Box<dyn std::any::Any>>,
}

impl Default for GenericGradientMap1 {
    fn default() -> Self {
        Self {
            values: BTreeMap::new(),
        }
    }
}

impl<T: 'static> GradientReader<GradientId<T>, T> for GenericGradientMap1 {
    fn read(&self, id: GradientId<T>) -> Option<&T> {
        self.values.get(&id.inner).map(|val| {
            val.downcast_ref::<T>()
                .expect("indices should have a unique type")
        })
    }
}

impl<T: 'static> GradientStore<GradientId<T>, T> for GenericGradientMap1 {
    fn insert(&mut self, id: GradientId<T>, gradient: T) {
        self.values.insert(id.inner, Box::new(gradient));
    }

    fn get_mut(&mut self, id: GradientId<T>) -> Option<&mut T> {
        self.values.get_mut(&id.inner).map(|val| {
            val.downcast_mut::<T>()
                .expect("indices should have a unique type")
        })
    }
}

/// Gradient store used by [`GraphN`].
/// Indices of type `GradientId<T>` are mapped to values of type `Value<T>`.
#[derive(Debug)]
pub struct GenericGradientMapN {
    values: BTreeMap<Id, Box<dyn std::any::Any>>,
}

impl Default for GenericGradientMapN {
    fn default() -> Self {
        Self {
            values: BTreeMap::new(),
        }
    }
}

impl<T: 'static> GradientReader<GradientId<T>, Value<T>> for GenericGradientMapN {
    fn read(&self, id: GradientId<T>) -> Option<&Value<T>> {
        self.values.get(&id.inner).map(|val| {
            val.downcast_ref::<Value<T>>()
                .expect("indices should have a unique type")
        })
    }
}

impl<T: 'static> GradientReader<GradientId<T>, T> for GenericGradientMapN {
    fn read(&self, id: GradientId<T>) -> Option<&T> {
        self.values.get(&id.inner).map(|val| {
            val.downcast_ref::<Value<T>>()
                .expect("indices should have a unique type")
                .data()
        })
    }
}

impl<T: 'static> GradientStore<GradientId<T>, Value<T>> for GenericGradientMapN {
    fn insert(&mut self, id: GradientId<T>, gradient: Value<T>) {
        self.values.insert(id.inner, Box::new(gradient));
    }

    fn get_mut(&mut self, id: GradientId<T>) -> Option<&mut Value<T>> {
        self.values.get_mut(&id.inner).map(|val| {
            val.downcast_mut::<Value<T>>()
                .expect("indices should have a unique type")
        })
    }
}

/// A gradient store that contains no value. This is used as a placeholder
/// when instantiating networks [`Net`] on algebras without backward propagation
/// such as [`Eval`] and [`Check`].
#[derive(Debug, Default)]
pub struct EmptyGradientMap;

impl<T> GradientReader<(), T> for EmptyGradientMap {
    fn read(&self, _id: ()) -> Option<&T> {
        None
    }
}

/// Configuration for id_arena.
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub(crate) struct GraphArenaBehavior;

impl id_arena::ArenaBehavior for GraphArenaBehavior {
    type Id = Id;

    #[inline]
    fn new_id(arena_id: u32, idx: usize) -> Self::Id {
        Self::Id {
            arena_id,
            index: std::num::NonZeroU32::new((idx + 1) as u32).expect("Too many nodes"),
        }
    }

    #[inline]
    fn index(id: Self::Id) -> usize {
        u32::from(id.index) as usize - 1
    }

    #[inline]
    fn arena_id(id: Self::Id) -> u32 {
        id.arena_id
    }
}

impl<T> GradientId<T> {
    /// Associate a gradient type with an index.
    pub(crate) fn new(id: Id) -> Self {
        Self {
            inner: id,
            marker: std::marker::PhantomData,
        }
    }
}

impl<T> Clone for GradientId<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner,
            marker: std::marker::PhantomData,
        }
    }
}

impl<T> Copy for GradientId<T> {}

impl<T> PartialEq for GradientId<T> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<T> Eq for GradientId<T> {}

impl Id {
    pub(crate) fn next_id(&self) -> Self {
        Self {
            arena_id: self.arena_id,
            index: std::num::NonZeroU32::new((self.index.get() + 1) as u32)
                .expect("Too many nodes"),
        }
    }
}

impl<T> std::hash::Hash for GradientId<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl<T> std::fmt::Debug for GradientId<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(f, "{:?}", self.inner)
    }
}

impl std::fmt::Debug for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        if f.alternate() {
            write!(f, "{} @ {}", self.index, self.arena_id)
        } else {
            write!(f, "{}", self.index)
        }
    }
}
