//! Utility functions and traits for `ElasticHashMap` implementations

use crate::ElasticHashMap;
use std::hash::Hash;

/// Extension trait for map implementations that provides additional utility methods
pub trait HashMapExtensions<K, V> {
    /// Returns the keys of the hash map as a Vec
    fn keys(&self) -> Vec<K>;

    /// Returns the values of the hash map as a Vec
    fn values(&self) -> Vec<V>;

    /// Returns true if the hash map contains the given key
    fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: std::borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized;
}

impl<K, V> HashMapExtensions<K, V> for ElasticHashMap<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn keys(&self) -> Vec<K> {
        self.iter().map(|(k, _)| k.clone()).collect()
    }

    fn values(&self) -> Vec<V> {
        self.iter().map(|(_, v)| v.clone()).collect()
    }

    fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: std::borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get(key).is_some()
    }
}

/// Creates an `ElasticHashMap` from an iterator of key-value pairs
#[allow(dead_code)]
pub fn from_iter<K, V, I>(iter: I) -> ElasticHashMap<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
    I: IntoIterator<Item = (K, V)>,
{
    let iter = iter.into_iter();
    let mut map = ElasticHashMap::new();

    for (key, value) in iter {
        map.insert(key, value);
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ElasticHashMap;

    #[test]
    fn test_from_iter() {
        let data = vec![("a".to_string(), 1), ("b".to_string(), 2), ("c".to_string(), 3)];

        let map = from_iter(data);

        assert_eq!(map.get("a"), Some(&1));
        assert_eq!(map.get("b"), Some(&2));
        assert_eq!(map.get("c"), Some(&3));
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn test_keys_and_values() {
        let mut map = ElasticHashMap::new();
        map.insert("a".to_string(), 1);
        map.insert("b".to_string(), 2);
        map.insert("c".to_string(), 3);

        let mut keys = map.keys();
        keys.sort(); // Sort for predictable comparison

        let mut values = map.values();
        values.sort_unstable();

        assert_eq!(keys, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_contains_key() {
        let mut map = ElasticHashMap::new();
        map.insert("a".to_string(), 1);

        assert!(map.contains_key("a"));
        assert!(!map.contains_key("b"));
    }
}
