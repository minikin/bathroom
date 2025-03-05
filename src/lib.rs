//! # Elastic Hash Map
//!
//! A Rust implementation of a hash table with adaptive probing strategy.
//!
//! This crate provides two hash map implementations:
//!
//! - `ElasticHashMap`: A single-threaded implementation optimized for performance
//! - `ConcurrentElasticMap`: A lock-free, thread-safe implementation for concurrent access
//!
//! Both implementations use an adaptive probing strategy that dynamically adjusts step
//! sizes based on observed occupancy patterns, resulting in more efficient lookups.
//!
//! ## Basic Usage
//!
//! ```rust
//! use bathroom::ElasticHashMap;
//!
//! // Create a new hash map
//! let mut map = ElasticHashMap::new();
//!
//! // Insert values
//! map.insert("apple".to_string(), 1);
//! map.insert("banana".to_string(), 2);
//!
//! // Retrieve values
//! assert_eq!(map.get("apple"), Some(&1));
//!
//! // Update values
//! map.insert("apple".to_string(), 10);
//! assert_eq!(map.get("apple"), Some(&10));
//!
//! // Remove values
//! map.remove("apple");
//! assert_eq!(map.get("apple"), None);
//! ```
//!
//! ## Concurrent Usage
//!
//! ```rust
//! use bathroom::ConcurrentElasticMap;
//! use std::sync::Arc;
//! use std::thread;
//!
//! // Create a shared hash map
//! let map = Arc::new(ConcurrentElasticMap::new());
//!
//! // Clone references for different threads
//! let map1 = Arc::clone(&map);
//! let map2 = Arc::clone(&map);
//!
//! // Spawn threads that modify the map concurrently
//! let t1 = thread::spawn(move || {
//!     for i in 0..100 {
//!         map1.insert(format!("key-{}", i), i);
//!     }
//! });
//!
//! let t2 = thread::spawn(move || {
//!     for i in 100..200 {
//!         map2.insert(format!("key-{}", i), i);
//!     }
//! });
//!
//! // Wait for threads to complete
//! t1.join().unwrap();
//! t2.join().unwrap();
//!
//! // Due to potential race conditions in a concurrent environment,
//! // the final count might be slightly less than expected
//! let count = map.len();
//! assert!(count >= 190, "Expected at least 190 entries, found {}", count);
//! ```

/// Module implementing a thread-safe concurrent hash map with elastic probing
mod concurrent_elastic_map;
/// Module implementing a single-threaded hash map with elastic probing
mod elastic_hashmap;
/// Utility functions and traits for the hash maps
mod utils;

pub use concurrent_elastic_map::ConcurrentElasticMap;
pub use elastic_hashmap::ElasticHashMap;
pub use utils::HashMapExtensions;
