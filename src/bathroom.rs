#![allow(dead_code)]

use std::hash::{DefaultHasher, Hash, Hasher};

#[derive(Debug, Clone)]
struct BathroomMap<K, V> {
    /// The stored items the key-value pairs
    items: Vec<Option<(K, V)>>,
    /// Current number of elements in the hash table
    size: usize,
}

impl<K, V> BathroomMap<K, V>
where
    K: Clone + Hash + PartialEq,
    V: Clone,
{
    pub fn new() -> Self {
        Self { items: vec![None], size: 0 }
    }

    pub fn insert(&mut self, k: K, v: V) {
        // TODO make a resizing based on the load factor.
        // (The ratio of the number of occupied slots to the total number of slots in the table)
        self.resize();
        let mut index = self.get_index(&k);

        let table_size = self.items.len();
        assert!(index < table_size);

        // The step size used for probing, which is adjusted dynamically based on occupancy.
        let step_size: usize = 1;
        while let Some(_) = self.items[index] {
            // TODO update `step_size` based on the occupancy threshold
            // (A threshold value that determines when to increase or decrease the step size).
            // If the number of consecutive occupied slots exceeds the occupancy threshold, increase
            // the step size. If the number of consecutive occupied slots is below the
            // threshold, decrease the step size.
            index = (index + step_size) % table_size;
            println!("new index {index}");
        }
        // Found an empty slot, inserting the value
        self.items[index] = Some((k, v));
        self.size += 1;
    }

    pub fn get(&self, k: &K) -> Option<&V> {
        let mut index = self.get_index(&k);
        let table_size = self.items.len();
        assert!(index < table_size);

        // The step size used for probing, which is adjusted dynamically based on occupancy.
        let step_size: usize = 1;
        while let Some((found_k, found_v)) = self.items[index].as_ref() {
            if found_k == k {
                return Some(found_v);
            }
            // `index` adjusment, must be the same as for `insert` function
            index = (index + step_size) % table_size;
            println!("new index {index}");
        }

        // Found nothing for the provided `k` value
        None
    }

    fn get_index(&self, k: &K) -> usize {
        assert_ne!(self.items.len(), 0);

        let mut hasher = DefaultHasher::new();
        k.hash(&mut hasher);
        let hash = hasher.finish();
        let table_size = self.items.len();
        // TODO use bitwise and operation instead of `%`
        #[allow(trivial_numeric_casts)]
        if size_of::<usize>() > size_of::<u64>() {
            // as we checked that size of the `u64` could not exceeds size of the `usize`, its safe
            // to cast `u64` to `usize` without any losses
            (hash as usize) % table_size
        } else {
            // as we checked that size of the usize could not exceeds size of the `u64`, its safe to
            // cast `usize` to `u64` without any losses.
            // Also as final result cannot exceeds the `last_index` value which is origninall
            // `usize` type its also safe to cast back to `usize`
            (hash % (table_size as u64)) as usize
        }
    }

    /// Resizes the hash table when it gets too full
    fn resize(&mut self) {
        assert_ne!(self.items.len(), 0);
        let new_capacity = self.items.len() * 2;
        assert!(new_capacity > self.items.len());
        self.items.resize(new_capacity, None);
    }
}

#[cfg(test)]
mod tests {
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

    #[property_test]
    fn bathroom_map_test(values: [(String, String); 2]) {
        let mut map = BathroomMap::new();

        for (k, v) in values.clone() {
            map.insert(k, v);
        }
        for (k, v) in values {
            println!("k {k}, v {v}");
            assert_eq!(map.get(&k), Some(&v));
        }
    }
}
