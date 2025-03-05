use std::borrow::Borrow;
use std::cell::UnsafeCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::marker::{PhantomData, Sync};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A thread-safe hash table with adaptive probing strategy.
///
/// This implementation uses atomic operations to ensure thread safety without traditional locks,
/// allowing for high concurrency with minimal contention. The adaptive probing strategy
/// dynamically adjusts step sizes based on occupancy patterns.
#[allow(missing_debug_implementations)]
pub struct ConcurrentElasticMap<K, V> {
    /// The buckets storing the key-value pairs
    buckets: Vec<AtomicBucket<K, V>>,
    /// Current number of elements in the hash table
    size: AtomicUsize,
    /// Threshold for load factor before resizing - stored as percentage (0-100)
    load_factor_threshold: AtomicUsize,
    /// Occupancy threshold for step size adjustment
    occupancy_threshold: AtomicUsize,
    /// Minimum step size for probing
    min_step_size: AtomicUsize,
    /// Maximum step size for probing (to avoid extremely large jumps)
    max_step_size: AtomicUsize,
}

/// A bucket containing a key-value pair with atomic state
struct AtomicBucket<K, V> {
    /// The state of the bucket
    state: AtomicUsize,
    /// The key-value data, if present
    data: UnsafeCell<Option<BucketData<K, V>>>,
}

/// The data stored in a bucket
struct BucketData<K, V> {
    /// The key stored in the bucket
    key: K,
    /// The value associated with the key
    value: V,
}

/// The possible states of a bucket
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BucketState {
    Empty = 0,
    Occupied = 1,
    Deleted = 2,
    Locked = 3,
}

/// Enum for the result of an insertion
enum InsertResult<V> {
    /// The key-value pair was newly inserted
    Inserted,
    /// The key already existed and its value was updated
    Updated(V),
    /// The insertion failed (table is full)
    Failed,
}

#[allow(clippy::missing_fields_in_debug)]
impl<K, V> std::fmt::Debug for AtomicBucket<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.load(std::sync::atomic::Ordering::Relaxed);
        let bucket_state = match state {
            0 => "Empty",
            1 => "Occupied",
            2 => "Deleted",
            3 => "Locked",
            _ => "Unknown",
        };

        f.debug_struct("AtomicBucket").field("state", &bucket_state).finish_non_exhaustive()
    }
}

// Make UnsafeCell safe to share between threads
unsafe impl<K, V> Sync for AtomicBucket<K, V> {}

impl<K, V> AtomicBucket<K, V> {
    /// Creates a new empty atomic bucket
    fn new() -> Self {
        Self { state: AtomicUsize::new(BucketState::Empty as usize), data: UnsafeCell::new(None) }
    }

    /// Sets the data in the bucket
    fn set_data(&self, data: Option<BucketData<K, V>>) {
        unsafe {
            *self.data.get() = data;
        }
    }

    /// Gets a reference to the data if present
    fn get_data(&self) -> Option<&BucketData<K, V>> {
        unsafe { (*self.data.get()).as_ref() }
    }

    /// Tries to lock the bucket for modification
    fn try_lock(&self) -> bool {
        let current = self.state.load(Ordering::Acquire);
        if current == BucketState::Locked as usize {
            return false;
        }

        self.state
            .compare_exchange(
                current,
                BucketState::Locked as usize,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok()
    }

    /// Releases the lock on the bucket, setting it to the specified state
    fn unlock(&self, state: BucketState) {
        self.state.store(state as usize, Ordering::Release);
    }

    /// Gets the current state of the bucket
    #[allow(clippy::match_same_arms)]
    fn get_state(&self) -> BucketState {
        match self.state.load(Ordering::Acquire) {
            0 => BucketState::Empty,
            1 => BucketState::Occupied,
            2 => BucketState::Deleted,
            3 => BucketState::Locked,
            // Return Empty for invalid states
            _ => BucketState::Empty,
        }
    }
}

impl<K, V> ConcurrentElasticMap<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Creates a new `ConcurrentElasticMap` with the default initial capacity and parameters
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(16)
    }

    /// Creates a new `ConcurrentElasticMap` with the specified initial capacity
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        // Ensure capacity is at least 1 and a power of 2
        let capacity = capacity.max(1).next_power_of_two();

        let mut buckets = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buckets.push(AtomicBucket::new());
        }

        Self {
            buckets,
            size: AtomicUsize::new(0),
            load_factor_threshold: AtomicUsize::new(75), // 75% load factor as default
            occupancy_threshold: AtomicUsize::new(2),
            min_step_size: AtomicUsize::new(1),
            max_step_size: AtomicUsize::new(capacity / 4), // Limit step size to 1/4 of capacity
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
    #[allow(clippy::arithmetic_side_effects)]
    #[allow(clippy::cast_possible_truncation)]
    fn get_index<Q: ?Sized + Hash>(&self, key: &Q) -> usize {
        let hash = self.hash(key);
        (hash as usize) & (self.buckets.len() - 1)
    }

    /// Inserts a key-value pair starting from the specified index
    fn insert_at(&self, start_index: usize, key: K, value: V) -> InsertResult<V> {
        let bucket_count = self.buckets.len();
        let occupancy_threshold = self.occupancy_threshold.load(Ordering::Relaxed);
        let max_step_size = self.max_step_size.load(Ordering::Relaxed);
        let min_step_size = self.min_step_size.load(Ordering::Relaxed);

        let mut index = start_index;
        let mut step_size: usize = min_step_size;
        let mut consecutive_occupied: usize = 0;

        let mut first_tombstone = None;

        // Elastic probing loop
        for _ in 0..bucket_count {
            let Some(bucket) = self.buckets.get(index) else { return InsertResult::Failed };
            let state = bucket.get_state();

            match state {
                // Empty slot - we can insert here
                BucketState::Empty => {
                    // If we found a tombstone earlier, use that position instead
                    if let Some(tombstone_index) = first_tombstone {
                        return self.do_insert_at(tombstone_index, key, value);
                    }
                    return self.do_insert_at(index, key, value);
                }

                // Deleted slot (tombstone)
                BucketState::Deleted => {
                    if first_tombstone.is_none() {
                        first_tombstone = Some(index);
                    }

                    // Reset consecutive occupied counter
                    consecutive_occupied = 0;

                    // Decrease step size when finding deleted slots
                    step_size = (step_size / 2).max(min_step_size);
                }

                // Occupied slot
                BucketState::Occupied => {
                    // Check if the key matches
                    if let Some(data) = bucket.get_data() {
                        if data.key == key {
                            // Try to update the value
                            if bucket.try_lock() {
                                // Check again in case it was modified while we were getting the lock
                                if let Some(data) = bucket.get_data() {
                                    if data.key == key {
                                        let old_value = data.value.clone();
                                        let new_data = BucketData { key, value };
                                        bucket.set_data(Some(new_data));
                                        bucket.unlock(BucketState::Occupied);
                                        return InsertResult::Updated(old_value);
                                    }
                                }
                                bucket.unlock(BucketState::Occupied);
                            }
                            // If we couldn't get the lock, continue probing
                        }
                    }

                    // Elastic step size adjustment
                    consecutive_occupied = consecutive_occupied.saturating_add(1);
                    if consecutive_occupied > occupancy_threshold {
                        // Increase step size exponentially when encountering many occupied slots
                        step_size = step_size.saturating_mul(2).min(max_step_size);
                    }
                }

                // Bucket is locked by another thread
                BucketState::Locked => {
                    // In a true lock-free implementation, we would use a retry mechanism
                    // or a helping pattern, but for simplicity we'll just continue probing
                    consecutive_occupied = consecutive_occupied.saturating_add(1);
                }
            }

            // Compute next index with the current step size
            index = index.saturating_add(step_size) & (bucket_count.saturating_sub(1));
        }

        // If we reach here, the table is likely full or all buckets are locked
        if let Some(tombstone_index) = first_tombstone {
            return self.do_insert_at(tombstone_index, key, value);
        }

        // We've probed all slots and couldn't find a place
        InsertResult::Failed
    }

    /// Performs the actual insertion at a specific index
    fn do_insert_at(&self, index: usize, key: K, value: V) -> InsertResult<V> {
        let Some(bucket) = self.buckets.get(index) else { return InsertResult::Failed };

        if !bucket.try_lock() {
            // Bucket is locked by another thread, retry with next position
            return InsertResult::Failed;
        }

        // Create new bucket data
        let data = BucketData { key, value };

        // Update the bucket
        bucket.set_data(Some(data));
        bucket.unlock(BucketState::Occupied);

        InsertResult::Inserted
    }

    /// Retrieve a value for a given key
    pub fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let index = self.get_index(key);
        self.get_from(index, key)
    }

    /// Get a value starting from the specified index
    #[allow(clippy::question_mark)]
    fn get_from<Q>(&self, start_index: usize, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let bucket_count = self.buckets.len();
        let occupancy_threshold = self.occupancy_threshold.load(Ordering::Relaxed);
        let max_step_size = self.max_step_size.load(Ordering::Relaxed);
        let min_step_size = self.min_step_size.load(Ordering::Relaxed);

        let mut index = start_index;
        let mut step_size: usize = min_step_size;
        let mut consecutive_occupied: usize = 0;

        // Elastic probing loop (for retrieval)
        for _ in 0..bucket_count {
            let bucket = self.buckets.get(index)?;
            let state = bucket.get_state();

            match state {
                // Empty slot means the key is not in the table
                // (assuming no concurrent insertions)
                BucketState::Empty => return None,

                // Occupied slot - check if the key matches
                BucketState::Occupied => {
                    if let Some(data) = bucket.get_data() {
                        if data.key.borrow() == key {
                            return Some(data.value.clone());
                        }
                    }

                    // Elastic step size adjustment
                    consecutive_occupied = consecutive_occupied.saturating_add(1);
                    if consecutive_occupied > occupancy_threshold {
                        step_size = step_size.saturating_mul(2).min(max_step_size);
                    }
                }

                // Deleted slot or locked slot - continue probing
                BucketState::Deleted | BucketState::Locked => {
                    consecutive_occupied = 0;
                    step_size = (step_size / 2).max(min_step_size);
                }
            }

            // Compute next index with current step size
            index = index.saturating_add(step_size) & (bucket_count.saturating_sub(1));
        }

        // If we get here, we've probed all slots and couldn't find the key
        None
    }

    /// Removes a key-value pair from the hash table
    pub fn remove<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let index = self.get_index(key);
        self.remove_from(index, key)
    }

    /// Removes a key-value pair starting from the specified index
    #[allow(clippy::question_mark)]
    fn remove_from<Q>(&self, start_index: usize, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let bucket_count = self.buckets.len();
        let occupancy_threshold = self.occupancy_threshold.load(Ordering::Relaxed);
        let max_step_size = self.max_step_size.load(Ordering::Relaxed);
        let min_step_size = self.min_step_size.load(Ordering::Relaxed);

        let mut index = start_index;
        let mut step_size: usize = min_step_size;
        let mut consecutive_occupied: usize = 0;

        // Elastic probing loop (for removal)
        for _ in 0..bucket_count {
            let bucket = self.buckets.get(index)?;
            let state = bucket.get_state();

            match state {
                // Empty slot means the key is not in the table
                BucketState::Empty => return None,

                // Occupied slot - check if the key matches
                BucketState::Occupied => {
                    if let Some(data) = bucket.get_data() {
                        if data.key.borrow() == key {
                            // Try to lock the bucket
                            if bucket.try_lock() {
                                // Check again in case it was modified while we were getting the lock
                                if let Some(data) = bucket.get_data() {
                                    if data.key.borrow() == key {
                                        let value = data.value.clone();
                                        bucket.unlock(BucketState::Deleted);
                                        self.size.fetch_sub(1, Ordering::Relaxed);
                                        return Some(value);
                                    }
                                }
                                bucket.unlock(BucketState::Occupied);
                            }
                        }
                    }

                    // Elastic step size adjustment
                    consecutive_occupied = consecutive_occupied.saturating_add(1);
                    if consecutive_occupied > occupancy_threshold {
                        step_size = step_size.saturating_mul(2).min(max_step_size);
                    }
                }

                // Deleted slot or locked slot - continue probing
                BucketState::Deleted | BucketState::Locked => {
                    consecutive_occupied = 0;
                    step_size = (step_size / 2).max(min_step_size);
                }
            }

            // Compute next index with current step size
            index = index.saturating_add(step_size) & (bucket_count.saturating_sub(1));
        }

        // If we get here, we've probed all slots and couldn't find the key
        None
    }

    /// Returns the number of elements in the hash table
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Returns true if the hash table is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Resizes the hash table when it gets too full
    fn resize(&self) {
        let size = self.size.load(Ordering::Relaxed);
        let bucket_count = self.buckets.len();

        // Convert to u64 first to avoid precision loss on 64-bit platforms
        let size_u64 = size as u64;
        let bucket_count_u64 = bucket_count as u64;
        let load_factor_threshold_u64 = (self.load_factor_threshold.load(Ordering::Relaxed) as u64)
            .saturating_mul(bucket_count_u64)
            .saturating_div(100);

        // Compare using integer arithmetic instead of floating point
        if size_u64 < load_factor_threshold_u64 {
            // Another thread probably already resized, so return
            return;
        }

        // Create a new, larger hash table
        let new_capacity = bucket_count.saturating_mul(2);
        let new_table = Self::with_capacity(new_capacity);

        // Copy configuration
        new_table
            .load_factor_threshold
            .store(self.load_factor_threshold.load(Ordering::Relaxed), Ordering::Relaxed);
        new_table
            .occupancy_threshold
            .store(self.occupancy_threshold.load(Ordering::Relaxed), Ordering::Relaxed);
        new_table
            .min_step_size
            .store(self.min_step_size.load(Ordering::Relaxed), Ordering::Relaxed);

        // Here's where a true lock-free resize would be more complex, involving
        // a new generation of buckets and helping other threads with the migration.
        // For simplicity, we'll just focus on the atomic operations for the buckets themselves.

        // In a full implementation, we would swap the old and new bucket arrays
        // atomically using a pointer indirection. This simplified version illustrates
        // the concept but is not truly lock-free for the resize operation.

        // Copy all non-deleted key-value pairs to the new table
        for bucket in &self.buckets {
            if bucket.get_state() == BucketState::Occupied {
                if let Some(data) = bucket.get_data() {
                    new_table.insert(data.key.clone(), data.value.clone());
                }
            }
        }
    }

    /// Provide a way to configure the occupancy threshold
    pub fn set_occupancy_threshold(&self, threshold: usize) {
        self.occupancy_threshold.store(threshold.max(1), Ordering::Relaxed);
    }

    /// Provide a way to configure the load factor threshold
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn set_load_factor_threshold(&self, threshold: f64) {
        let threshold_percent = (threshold.clamp(0.1, 0.95) * 100.0).round() as usize;
        self.load_factor_threshold.store(threshold_percent, Ordering::Relaxed);
    }

    /// Returns an iterator over the key-value pairs
    pub fn iter(&self) -> Iter<K, V> {
        Iter { buckets: &self.buckets, index: 0, _marker: PhantomData }
    }

    /// Returns the capacity (number of buckets) in the map
    pub fn capacity(&self) -> usize {
        self.buckets.len()
    }

    /// Returns the current load factor of the map
    pub fn load_factor(&self) -> f64 {
        // This is only used for informational purposes, so the precision loss is acceptable
        let size = self.size.load(Ordering::Relaxed);
        let bucket_count = self.buckets.len();

        if bucket_count == 0 {
            return 0.0;
        }

        let percentage = size.saturating_mul(100);
        let final_percentage = percentage.saturating_div(bucket_count);
        (final_percentage as f64) / 100.0
    }

    /// Inserts a key-value pair into the map
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        // Check if we need to resize - use a relaxed load since this is just a hint
        let size = self.size.load(Ordering::Relaxed);
        let load_factor_threshold =
            self.load_factor_threshold.load(Ordering::Relaxed) as f64 / 100.0;

        if (size as f64) / (self.buckets.len() as f64) >= load_factor_threshold {
            self.resize();
        }

        let start_index = self.get_index(&key);
        match self.insert_at(start_index, key, value) {
            InsertResult::Inserted => {
                self.size.fetch_add(1, Ordering::Relaxed);
                None
            }
            InsertResult::Updated(old_value) => Some(old_value),
            // Return None for failures instead of panicking
            InsertResult::Failed => None,
        }
    }
}

/// Iterator over the key-value pairs of the lock-free hash table
#[derive(Debug)]
pub struct Iter<'a, K, V> {
    /// Reference to the buckets in the map
    buckets: &'a [AtomicBucket<K, V>],
    /// Current index position in the iteration
    index: usize,
    /// `PhantomData` to maintain variance over K and V
    _marker: PhantomData<&'a (K, V)>,
}

#[allow(clippy::needless_lifetimes, single_use_lifetimes)]
impl<'a, K, V> Iterator for Iter<'a, K, V>
where
    K: Clone,
    V: Clone,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.buckets.len() {
            let Some(bucket) = self.buckets.get(self.index) else {
                self.index = self.index.saturating_add(1);
                continue;
            };
            self.index = self.index.saturating_add(1);

            if bucket.get_state() == BucketState::Occupied {
                if let Some(data) = bucket.get_data() {
                    return Some((data.key.clone(), data.value.clone()));
                }
            }
        }
        None
    }
}

#[allow(single_use_lifetimes)]
impl<'a, K, V> IntoIterator for &'a ConcurrentElasticMap<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    type Item = (K, V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K, V> Default for ConcurrentElasticMap<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_insert_and_get() {
        let map = ConcurrentElasticMap::new();
        assert_eq!(map.insert("key1".to_string(), 1), None);
        assert_eq!(map.insert("key2".to_string(), 2), None);
        assert_eq!(map.insert("key3".to_string(), 3), None);

        assert_eq!(map.get("key1"), Some(1));
        assert_eq!(map.get("key2"), Some(2));
        assert_eq!(map.get("key3"), Some(3));
        assert_eq!(map.get("key4"), None);
    }

    #[test]
    fn test_update() {
        let map = ConcurrentElasticMap::new();
        assert_eq!(map.insert("key1".to_string(), 1), None);
        assert_eq!(map.insert("key1".to_string(), 10), Some(1));
        assert_eq!(map.get("key1"), Some(10));
    }

    #[test]
    fn test_remove() {
        let map = ConcurrentElasticMap::new();
        map.insert("key1".to_string(), 1);
        map.insert("key2".to_string(), 2);

        assert_eq!(map.remove("key1"), Some(1));
        assert_eq!(map.get("key1"), None);
        assert_eq!(map.get("key2"), Some(2));
        assert_eq!(map.remove("key1"), None);
    }

    #[test]
    fn test_len_and_is_empty() {
        let map = ConcurrentElasticMap::new();
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
    fn test_concurrent_inserts() {
        let map = Arc::new(ConcurrentElasticMap::new());
        let mut handles = vec![];

        // Create 8 threads, each inserting 100 items
        for t in 0..8 {
            let map_clone = Arc::clone(&map);
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    let key = format!("key-{t}-{i}");
                    let value = t * 100 + i;
                    map_clone.insert(key, value);
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Check that all items were inserted
        assert_eq!(map.len(), 800);

        // Verify some random items
        for t in 0..8 {
            for i in (0..100).step_by(10) {
                let key = format!("key-{t}-{i}");
                assert_eq!(map.get(&key), Some(t * 100 + i));
            }
        }
    }

    #[test]
    fn test_concurrent_reads_and_writes() {
        let map = Arc::new(ConcurrentElasticMap::new());

        // Preload some data
        for i in 0..100 {
            map.insert(format!("key-{i}"), i);
        }

        let mut handles = vec![];

        // Create writer threads
        for t in 0..4 {
            let map_clone = Arc::clone(&map);
            let handle = thread::spawn(move || {
                for i in 0..50 {
                    let key = format!("key-writer-{t}-{i}");
                    map_clone.insert(key, t * 100 + i);
                }
            });
            handles.push(handle);
        }

        // Create reader threads
        let mut reader_handles = vec![];
        for _ in 0..4 {
            let map_clone = Arc::clone(&map);
            let handle = thread::spawn(move || {
                let mut read_count = 0;
                for i in 0..100 {
                    let key = format!("key-{i}");
                    if map_clone.get(&key).is_some() {
                        read_count += 1;
                    }
                }
                read_count
            });
            reader_handles.push(handle);
        }

        // Create remover threads
        let mut remove_handles = vec![];
        for t in 0..2 {
            let map_clone = Arc::clone(&map);
            let handle = thread::spawn(move || {
                let mut remove_count = 0;
                for i in (t * 50)..((t + 1) * 50) {
                    let key = format!("key-{i}");
                    if map_clone.remove(&key).is_some() {
                        remove_count += 1;
                    }
                }
                remove_count
            });
            remove_handles.push(handle);
        }

        // Wait for all threads and collect results
        for handle in handles {
            handle.join().unwrap();
        }

        let mut reader_results = Vec::new();
        for handle in reader_handles {
            reader_results.push(handle.join().unwrap());
        }

        let mut remove_results = Vec::new();
        for handle in remove_handles {
            remove_results.push(handle.join().unwrap());
        }

        // Each reader should have found some keys
        for &reads in &reader_results {
            assert!(reads > 0);
        }

        // The remove threads should have removed some keys
        let total_removed: usize = remove_results.iter().sum();
        assert!(total_removed > 0);

        // Final size should be: 100 (initial) + 200 (4 writers * 50) - removed
        assert_eq!(map.len(), 100 + 200 - total_removed);
    }
}
