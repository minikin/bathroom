use std::{
    borrow::Borrow,
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem,
};

/// A bucket containing a key-value pair
#[derive(Debug, Clone)]
struct Bucket<K, V> {
    /// The key in the key-value pair
    key: K,
    /// The value associated with the key
    value: V,
    /// Flag indicating whether this entry has been deleted (tombstone)
    deleted: bool, // Tombstone flag for deletion
}

/// A hash table with adaptive probing strategy.
///
/// This implementation uses a dynamic step size adjustment based on occupancy patterns,
/// leading to more efficient lookups compared to traditional probing methods.
///
/// Note: This implementation is not thread-safe. For concurrent access, use `ConcurrentElasticMap`.
#[derive(Debug, Clone)]
pub struct ElasticHashMap<K, V> {
    /// The buckets storing the key-value pairs
    buckets: Vec<Option<Bucket<K, V>>>,
    /// Current number of elements in the hash table
    size: usize,
    /// Threshold for load factor before resizing - stored as percentage (0-100)
    load_factor_threshold: usize,
    /// Occupancy threshold for step size adjustment
    occupancy_threshold: usize,
    /// Minimum step size for probing
    min_step_size: usize,
    /// Maximum step size for probing (to avoid extremely large jumps)
    max_step_size: usize,
}

impl<K, V> Default for ElasticHashMap<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Extend<(K, V)> for ElasticHashMap<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K, V> ElasticHashMap<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Creates a new `ElasticHashMap` with the default initial capacity and parameters
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(64)
    }

    /// Creates a new `ElasticHashMap` with the specified initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        // Ensure capacity is at least 1 and a power of 2
        let capacity = capacity.max(1).next_power_of_two();

        Self {
            buckets: vec![None; capacity],
            size: 0,
            load_factor_threshold: 75,
            occupancy_threshold: 2,
            min_step_size: 1,
            max_step_size: capacity / 4, // Limit step size to 1/4 of capacity
        }
    }

    /// Computes the hash for a key
    #[allow(clippy::unused_self)]
    fn hash<Q: ?Sized + Hash>(&self, key: &Q) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Gets the index in the buckets for a hash
    #[allow(clippy::cast_possible_truncation)]
    fn get_index<Q: ?Sized + Hash>(&self, key: &Q) -> usize {
        let hash = self.hash(key);
        (hash as usize) & (self.buckets.len().saturating_sub(1))
    }

    /// Insert a key-value pair into the hash table
    #[allow(clippy::arithmetic_side_effects, clippy::cast_precision_loss)]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Check if we need to resize
        if (self.size as f64) / (self.buckets.len() as f64) >=
            self.load_factor_threshold as f64 / 100.0
        {
            self.resize();
        }

        let index = self.get_index(&key);
        let result = self.insert_at(index, key, value);

        if result.is_none() {
            self.size = self.size.saturating_add(1);
        }

        result
    }

    /// Inserts a key-value pair starting from the specified index
    fn insert_at(&mut self, start_index: usize, key: K, value: V) -> Option<V> {
        let bucket_count = self.buckets.len();
        let mut index = start_index;
        let mut step_size = self.min_step_size;
        let mut consecutive_occupied: usize = 0;
        let mut first_tombstone = None;

        // Elastic probing loop
        for _ in 0..bucket_count {
            // Check if current bucket matches the key
            #[allow(clippy::question_mark, clippy::manual_let_else)]
            let bucket_ref = match self.buckets.get_mut(index) {
                Some(option) => option,
                None => return None,
            };
            match bucket_ref {
                // Found an empty slot
                None => {
                    // If we found a tombstone earlier, use that position instead
                    if let Some(tombstone_index) = first_tombstone {
                        if let Some(tomb_bucket) = self.buckets.get_mut(tombstone_index) {
                            *tomb_bucket = Some(Bucket { key, value, deleted: false });
                            return None;
                        }
                    } else {
                        *bucket_ref = Some(Bucket { key, value, deleted: false });
                        return None;
                    }
                    return None; // Fallback if get_mut fails, shouldn't happen
                }

                // Found a bucket with data
                Some(bucket) => {
                    // If this is a tombstone and we haven't recorded one yet
                    if bucket.deleted && first_tombstone.is_none() {
                        first_tombstone = Some(index);
                    }
                    // If the key matches and it's not deleted, update the value
                    else if !bucket.deleted && bucket.key == key {
                        let old_value = mem::replace(&mut bucket.value, value);
                        return Some(old_value);
                    }

                    // Dynamic step size adjustment
                    if bucket.deleted {
                        consecutive_occupied = 0;
                        // Decrease step size when finding deleted slots or empty spaces
                        step_size = (step_size / 2).max(self.min_step_size);
                    } else {
                        consecutive_occupied = consecutive_occupied.saturating_add(1);
                        if consecutive_occupied > self.occupancy_threshold {
                            // Increase step size exponentially when encountering many occupied
                            // slots
                            step_size = (step_size.saturating_mul(2)).min(self.max_step_size);
                        }
                    }
                }
            }

            // Compute next index with the current step size
            index = (index.saturating_add(step_size)) & (bucket_count.saturating_sub(1));
        }

        // If we get here, we've probed all slots and couldn't find a place
        // This should never happen if we resize properly
        // Return None to indicate insertion failure instead of panicking
        None
    }

    /// Retrieve a value for a given key
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let index = self.get_index(key);
        self.get_from(index, key)
    }

    /// Get a value starting from the specified index
    fn get_from<Q>(&self, start_index: usize, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let bucket_count = self.buckets.len();
        let mut index = start_index;
        let mut step_size = self.min_step_size;
        let mut consecutive_occupied: usize = 0;

        // Elastic probing loop (for retrieval)
        for _ in 0..bucket_count {
            match self.buckets.get(index) {
                // Empty slot or None bucket, the key is not in the table
                None | Some(None) => return None,

                // Found a bucket with data
                Some(Some(bucket)) => {
                    // If the key matches and it's not deleted, return the value
                    if !bucket.deleted && bucket.key.borrow() == key {
                        return Some(&bucket.value);
                    }

                    // Dynamic step size adjustment (similar to insert)
                    if bucket.deleted {
                        consecutive_occupied = 0;
                        // Decrease step size when finding deleted slots
                        step_size = (step_size / 2).max(self.min_step_size);
                    } else {
                        consecutive_occupied = consecutive_occupied.saturating_add(1);
                        if consecutive_occupied > self.occupancy_threshold {
                            // Increase step size exponentially when encountering many occupied
                            // slots
                            step_size = (step_size.saturating_mul(2)).min(self.max_step_size);
                        }
                    }
                }
            }

            // Compute next index with current step size
            index = (index.saturating_add(step_size)) & (bucket_count.saturating_sub(1));
        }

        None
    }

    /// Get a mutable reference to a value for a given key
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let index = self.get_index(key);
        self.get_mut_from(index, key)
    }

    /// Get a mutable reference to a value starting from the specified index
    fn get_mut_from<Q>(&mut self, start_index: usize, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let bucket_count = self.buckets.len();
        let mut index = start_index;
        let mut step_size = self.min_step_size;
        let mut consecutive_occupied: usize = 0;

        // Elastic probing loop
        for _ in 0..bucket_count {
            // Check if current bucket matches the key
            #[allow(clippy::question_mark, clippy::manual_let_else)]
            let bucket_ref = match self.buckets.get_mut(index) {
                Some(option) => option,
                None => return None,
            };
            match bucket_ref {
                None => return None,
                Some(bucket) if !bucket.deleted && bucket.key.borrow() == key => {
                    // We found the key, return a mutable reference to its value
                    if let Some(Some(bucket_data)) = self.buckets.get_mut(index) {
                        return Some(&mut bucket_data.value);
                    }
                    return None; // This should never happen, but safer than unwrap()
                }
                Some(bucket) => {
                    // Dynamic step size adjustment
                    if bucket.deleted {
                        consecutive_occupied = 0;
                        // Decrease step size when finding deleted slots
                        step_size = (step_size / 2).max(self.min_step_size);
                    } else {
                        consecutive_occupied = consecutive_occupied.saturating_add(1);
                        if consecutive_occupied > self.occupancy_threshold {
                            step_size = (step_size.saturating_mul(2)).min(self.max_step_size);
                        }
                    }
                }
            }

            // Advance to next position
            index = (index.saturating_add(step_size)) & (bucket_count.saturating_sub(1));
        }

        None
    }

    /// Removes a key-value pair from the hash table
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let index = self.get_index(key);
        self.remove_from(index, key)
    }

    /// Removes a key-value pair starting from the specified index
    fn remove_from<Q>(&mut self, start_index: usize, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let bucket_count = self.buckets.len();
        let mut index = start_index;
        let mut step_size = self.min_step_size;
        let mut consecutive_occupied: usize = 0;

        // Elastic probing loop (for removal)
        for _ in 0..bucket_count {
            match self.buckets.get_mut(index) {
                // Found an empty slot (and not a tombstone), the key is not in the table
                None | Some(None) => return None,

                // Found a bucket with data
                Some(Some(bucket)) => {
                    // If the key matches and it's not deleted, remove it
                    if !bucket.deleted && bucket.key.borrow() == key {
                        bucket.deleted = true;
                        self.size = self.size.saturating_sub(1);
                        return Some(bucket.value.clone());
                    }

                    // Dynamic step size adjustment
                    if bucket.deleted {
                        consecutive_occupied = 0;
                        // Decrease step size when finding deleted slots
                        step_size = (step_size / 2).max(self.min_step_size);
                    } else {
                        consecutive_occupied = consecutive_occupied.saturating_add(1);
                        if consecutive_occupied > self.occupancy_threshold {
                            // Increase step size exponentially when encountering many occupied
                            // slots
                            step_size = (step_size.saturating_mul(2)).min(self.max_step_size);
                        }
                    }
                }
            }

            // Compute next index with current step size
            index = (index.saturating_add(step_size)) & (bucket_count.saturating_sub(1));
        }

        None
    }

    /// Returns the number of elements in the hash table
    #[must_use]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns true if the hash table is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Resizes the hash table when it gets too full
    fn resize(&mut self) {
        let new_capacity = self.buckets.len().saturating_mul(2);
        let mut new_table = Self {
            buckets: vec![None; new_capacity],
            size: 0,
            load_factor_threshold: self.load_factor_threshold,
            occupancy_threshold: self.occupancy_threshold,
            min_step_size: self.min_step_size,
            max_step_size: new_capacity / 4, // Update max step size for new capacity
        };

        // Move all non-deleted key-value pairs to the new table
        for bucket in self.buckets.iter().filter_map(|b| b.as_ref()) {
            if !bucket.deleted {
                new_table.insert(bucket.key.clone(), bucket.value.clone());
            }
        }

        // Replace the current table with the new one
        *self = new_table;
    }

    /// Provide a way to configure the occupancy threshold
    pub fn set_occupancy_threshold(&mut self, threshold: usize) {
        self.occupancy_threshold = threshold.max(1); // Ensure at least 1
    }

    /// Provide a way to configure the load factor threshold
    pub fn set_load_factor_threshold(&mut self, threshold: usize) {
        self.load_factor_threshold = threshold.clamp(1, 95); // Keep within reasonable range
    }

    /// Returns an iterator over the key-value pairs
    #[must_use]
    #[allow(clippy::iter_without_into_iter)]
    pub fn iter(&self) -> Iter<K, V> {
        Iter { buckets: &self.buckets, index: 0, _marker: PhantomData }
    }

    /// Clears the hash map, removing all key-value pairs
    pub fn clear(&mut self) {
        for bucket in &mut self.buckets {
            *bucket = None;
        }
        self.size = 0;
    }

    /// Returns the number of buckets in the hash map
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.buckets.len()
    }

    /// Returns the current load factor of the hash map
    #[must_use]
    #[allow(clippy::arithmetic_side_effects, clippy::cast_precision_loss)]
    pub fn load_factor(&self) -> f64 {
        self.size as f64 / self.buckets.len() as f64
    }
}

/// Iterator over the key-value pairs of the hash table
#[derive(Debug, Clone)]
pub struct Iter<'a, K, V> {
    /// Reference to the buckets in the hash map
    buckets: &'a [Option<Bucket<K, V>>],
    /// Current position in the iteration
    index: usize,
    /// Phantom data to hold the lifetime and type parameters
    _marker: PhantomData<&'a (K, V)>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.buckets.len() {
            if let Some(Some(bucket)) = self.buckets.get(self.index) {
                self.index = self.index.saturating_add(1);
                if !bucket.deleted {
                    return Some((&bucket.key, &bucket.value));
                }
            } else {
                self.index = self.index.saturating_add(1);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let mut map = ElasticHashMap::new();
        assert_eq!(map.insert("key1".to_string(), 1), None);
        assert_eq!(map.insert("key2".to_string(), 2), None);
        assert_eq!(map.insert("key3".to_string(), 3), None);

        assert_eq!(map.get("key1"), Some(&1));
        assert_eq!(map.get("key2"), Some(&2));
        assert_eq!(map.get("key3"), Some(&3));
        assert_eq!(map.get("key4"), None);
    }

    #[test]
    fn test_update() {
        let mut map = ElasticHashMap::new();
        assert_eq!(map.insert("key1".to_string(), 1), None);
        assert_eq!(map.insert("key1".to_string(), 10), Some(1));
        assert_eq!(map.get("key1"), Some(&10));
    }

    #[test]
    fn test_remove() {
        let mut map = ElasticHashMap::new();
        map.insert("key1".to_string(), 1);
        map.insert("key2".to_string(), 2);

        assert_eq!(map.remove("key1"), Some(1));
        assert_eq!(map.get("key1"), None);
        assert_eq!(map.get("key2"), Some(&2));
        assert_eq!(map.remove("key1"), None);
    }

    #[test]
    fn test_resize() {
        let mut map = ElasticHashMap::with_capacity(4);
        map.set_load_factor_threshold(50);

        // Initial capacity is 4, so after 2 inserts (load factor > 50%), it should resize
        map.insert("key1".to_string(), 1);
        map.insert("key2".to_string(), 2);
        map.insert("key3".to_string(), 3); // This should trigger resize to capacity 8

        assert_eq!(map.get("key1"), Some(&1));
        assert_eq!(map.get("key2"), Some(&2));
        assert_eq!(map.get("key3"), Some(&3));
        assert_eq!(map.capacity(), 8);
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut map = ElasticHashMap::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);

        map.insert("key1".to_string(), 1);
        assert!(!map.is_empty());
        assert_eq!(map.len(), 1);

        map.insert("key2".to_string(), 2);
        assert_eq!(map.len(), 2);

        map.remove("key1");
        assert_eq!(map.len(), 1);

        map.remove("key2");
        assert!(map.is_empty());
    }

    #[test]
    fn test_iter() {
        let mut map = ElasticHashMap::new();
        map.insert("key1".to_string(), 1);
        map.insert("key2".to_string(), 2);
        map.insert("key3".to_string(), 3);

        let mut count = 0;
        let mut sum = 0;
        for (_, &value) in map.iter() {
            count += 1;
            sum += value;
        }

        assert_eq!(count, 3);
        assert_eq!(sum, 6);
    }

    #[test]
    fn test_get_mut() {
        let mut map = ElasticHashMap::new();
        map.insert("key1".to_string(), 1);

        if let Some(value) = map.get_mut("key1") {
            *value += 10;
        }

        assert_eq!(map.get("key1"), Some(&11));
    }

    #[test]
    fn test_clear() {
        let mut map = ElasticHashMap::new();
        map.insert("key1".to_string(), 1);
        map.insert("key2".to_string(), 2);

        assert_eq!(map.len(), 2);

        map.clear();

        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert_eq!(map.get("key1"), None);
        assert_eq!(map.get("key2"), None);
    }

    #[test]
    fn test_with_high_load_factor() {
        let mut map = ElasticHashMap::with_capacity(16);
        map.set_load_factor_threshold(90);

        for i in 0..14 {
            map.insert(i.to_string(), i);
        }

        for i in 0..14 {
            assert_eq!(map.get(&i.to_string()), Some(&i));
        }

        // Check that the load factor is correctly reported
        assert!((map.load_factor() - 14.0 / 16.0).abs() < 0.01);
    }
}
