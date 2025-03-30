#![allow(dead_code)]

use std::{
    hash::{DefaultHasher, Hash, Hasher},
    ptr::copy_nonoverlapping,
};

#[derive(Debug, Clone)]
struct BathroomMap<K, V> {
    /// The stored items the key-value pairs
    items: Box<[Option<(K, V)>]>,
    /// Current number of elements in the hash table
    size: usize,
    /// The step size used for probing, which is adjusted dynamically based on occupancy.
    step_size: usize,
    /// The ratio of the number of occupied slots to the total number of slots in the table
    /// (0-100).
    load_factor: usize,
}

impl<K, V> BathroomMap<K, V>
where
    K: Clone + Hash,
    V: Clone,
{
    fn new() -> Self {
        Self { items: Box::new([None]), size: 0, step_size: 1, load_factor: 0 }
    }

    fn insert(&mut self, k: K, _v: V) {
        // TODO make a resizing based on the load factor.
        // (The ratio of the number of occupied slots to the total number of slots in the table)
        self.resize();

        let _index = self.get_index(&k);
    }

    fn get_index(&self, k: &K) -> usize {
        assert_ne!(self.items.len(), 0);

        let mut hasher = DefaultHasher::new();
        k.hash(&mut hasher);
        let hash = hasher.finish();
        let last_index = self.items.len().saturating_sub(1);
        // TODO use bitwise and operation instead of `%`
        #[allow(trivial_numeric_casts)]
        if size_of::<usize>() > size_of::<u64>() {
            // as we checked that size of the `u64` could not exceeds size of the `usize`, its safe to
            // cast `u64` to `usize` without any losses
            (hash as usize) % last_index
        } else {
            // as we checked that size of the usize could not exceeds size of the `u64`, its safe to
            // cast `usize` to `u64` without any losses.
            // Also as final result cannot exceeds the `last_index` value which is origninall `usize` type its also safe to cast back to `usize`
            (hash % (last_index as u64)) as usize
        }
    }

    /// Resizes the hash table when it gets too full
    /// Increses the size on 50%
    fn resize(&mut self) {
        assert_ne!(self.items.len(), 0);
        let new_capacity = self.items.len() + self.items.len() / 2;
        let mut new_items = vec![None; new_capacity].into_boxed_slice();

        // Move elements from the old box to the new one
        unsafe {
            let src_ptr = self.items.as_ptr();
            let dest_ptr = new_items.as_mut_ptr();
            // Move elements without cloning (efficient and avoids extra copies)
            copy_nonoverlapping(src_ptr, dest_ptr, self.items.len());
        }

        self.items = new_items;
    }
}
