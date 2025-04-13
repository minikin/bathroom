//! # Elastic Hash Map
//!
//! A Rust implementation of a hash table with adaptive probing strategy.
//!
//! This crate provides two hash map implementations:
//!
//! - `ElasticHashMap`: A single-threaded implementation optimized for performance
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
/// Module implementing a single-threaded hash map with elastic probing
mod elastic_hashmap;
/// Utility functions and traits for the hash maps
mod utils;

pub use elastic_hashmap::ElasticHashMap;
pub use utils::HashMapExtensions;
