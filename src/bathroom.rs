#![allow(
    missing_docs,
    clippy::missing_docs_in_private_items,
    clippy::cast_possible_truncation,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing
)]

use std::hash::{DefaultHasher, Hash, Hasher};

#[derive(Debug, Clone)]
pub struct BathroomMap<K, V> {
    /// The stored items the key-value pairs
    items: Vec<Option<(K, V)>>,
    /// Current number of elements in the hash table
    size: usize,
}

impl<K, V> Default for BathroomMap<K, V>
where
    K: Clone,
    V: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> BathroomMap<K, V>
where
    K: Clone,
    V: Clone,
{
    #[must_use]
    pub fn new() -> Self {
        Self::new_with_capacity(100)
    }

    fn new_with_capacity(capacity: usize) -> Self {
        Self { items: vec![None; capacity], size: 0 }
    }
}

impl<K, V> BathroomMap<K, V>
where
    K: Clone + Hash + PartialEq,
    V: Clone,
{
    pub fn insert(&mut self, k: K, v: V) {
        /// Load factor threshold
        const LOAD_FACTOR_THRESHOLD: usize = 80;
        if self.load_factor() > LOAD_FACTOR_THRESHOLD {
            // double
            self.resize(2);
        }

        self.insert_impl(k, v);
    }

    fn insert_impl(&mut self, k: K, v: V) {
        let mut index = self.get_index(&k);

        let table_size = self.items.len();
        // The step size used for probing, which is adjusted dynamically based on occupancy.
        let step_size: usize = 1;
        debug_assert!(index < table_size);
        while self.items[index].is_some() {
            // TODO update `step_size` based on the occupancy threshold
            // (A threshold value that determines when to increase or decrease the step size).
            // If the number of consecutive occupied slots exceeds the occupancy threshold, increase
            // the step size. If the number of consecutive occupied slots is below the
            // threshold, decrease the step size.
            index = (index + step_size) % table_size;
        }
        // Found an empty slot, inserting the value
        self.items[index] = Some((k, v));
        self.size += 1;
    }

    pub fn get(&self, k: &K) -> Option<&V> {
        let mut index = self.get_index(k);
        let table_size = self.items.len();
        debug_assert!(index < table_size);

        // The step size used for probing, which is adjusted dynamically based on occupancy.
        let step_size: usize = 1;
        while let Some((found_k, found_v)) = self.items[index].as_ref() {
            if found_k == k {
                return Some(found_v);
            }
            // `index` adjusment, must be the same as for `insert` function
            index = (index + step_size) % table_size;
        }

        // Found nothing for the provided `k` value
        None
    }

    fn get_index(&self, k: &K) -> usize {
        debug_assert_ne!(self.items.len(), 0);

        let mut hasher = DefaultHasher::new();
        k.hash(&mut hasher);
        let hash = hasher.finish();
        let table_size = self.items.len();
        // TODO use bitwise and operation instead of `%`
        #[allow(trivial_numeric_casts)]
        let index = if size_of::<usize>() > size_of::<u64>() {
            // as we checked that size of the `u64` could not exceeds size of the `usize`, its safe
            // to cast `u64` to `usize` without any losses
            (hash as usize) % table_size
        } else {
            // as we checked that size of the usize could not exceeds size of the `u64`, its safe to
            // cast `usize` to `u64` without any losses.
            // Also as final result cannot exceeds the `last_index` value which is origninall
            // `usize` type its also safe to cast back to `usize`
            (hash % (table_size as u64)) as usize
        };
        debug_assert!(index < table_size);
        index
    }

    /// Calculates a current load factor, as percentage (0-100).
    /// The ratio of the number of occupied slots to the total number of slots in the table.
    fn load_factor(&self) -> usize {
        let load_factor = self.size * 100 / self.items.len();
        debug_assert!(load_factor <= 100);
        load_factor
    }

    /// Resizes the hash table when it gets too full
    fn resize(&mut self, resize_multiplier: usize) {
        debug_assert_ne!(self.items.len(), 0);
        debug_assert_ne!(resize_multiplier, 0);
        let new_capacity = self.items.len() * resize_multiplier;
        debug_assert!(new_capacity > self.items.len());
        let mut new_table = Self::new_with_capacity(new_capacity);
        let filtered_iter = std::mem::take(&mut self.items).into_iter().flatten();
        for (k, v) in filtered_iter {
            new_table.insert_impl(k, v);
        }
        *self = new_table;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use proptest::property_test;

    #[test]
    fn bathroom_map_check() {
        let mut map = BathroomMap::new();

        map.insert("key1", "value1");
        assert_eq!(map.get(&"key1"), Some(&"value1"));

        map.insert("key2", "value2");
        assert_eq!(map.get(&"key2"), Some(&"value2"));
    }

    #[test]
    fn resize_check() {
        let mut map = BathroomMap::new();
        let init_size = map.items.len();

        map.insert("key1", "value1");
        map.insert("key2", "value2");
        assert_eq!(map.items.len(), init_size);

        map.resize(2);
        assert_eq!(map.items.len(), init_size * 2);
        assert_eq!(map.get(&"key1"), Some(&"value1"));
        assert_eq!(map.get(&"key2"), Some(&"value2"));
    }

    #[property_test]
    fn bathroom_map_test(values: [(String, String); 200]) {
        let values: HashMap<_, _> = values.into_iter().collect();
        let mut map = BathroomMap::new();

        for (k, v) in values.clone() {
            map.insert(k, v);
        }
        for (k, v) in values {
            assert_eq!(map.get(&k), Some(&v));
        }
    }
}
