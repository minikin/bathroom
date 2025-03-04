# Bathroom

A Rust implementation of the Bathroom Model for hash map optimization, 
featuring adaptive probing strategies inspired by real-world bathroom stall selection behavior.

- [Bathroom](#bathroom)
  - [Overview](#overview)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Basic Usage](#basic-usage)
    - [Concurrent Usage](#concurrent-usage)
  - [How It Works](#how-it-works)
    - [Core Concepts](#core-concepts)
    - [The Bathroom Analogy](#the-bathroom-analogy)
    - [Theoretical Advantages](#theoretical-advantages)
  - [Performance Tuning](#performance-tuning)
  - [Implementation Details](#implementation-details)
  - [Future Work](#future-work)
  - [References](#references)
  - [Contributing](#contributing)
  - [License](#license)

## Overview

This implementation is based on two groundbreaking research papers:
- "The Bathroom Model: A Realistic Approach to Hash Table Algorithm Optimization" by Qiantong Wang
- "Optimal Bounds for Open Addressing Without Reordering" by Martín Farach-Colton, Andrew Krapivin, and William Kuszmaul

The Bathroom Model draws inspiration from human behavior when selecting bathroom stalls. 
Just as people dynamically adjust their search strategy based on occupancy patterns when looking for an available stall, 
this hash table implementation adaptively modifies its probing sequence based on observed data distribution patterns.

## Features

- **Adaptive Probing Strategy**: Dynamically adjusts step sizes during probing based on observed occupancy, avoiding clusters for better performance
- **Theoretical Guarantees**: Achieves O(1) amortized expected probe complexity and O(log δ⁻¹) worst-case expected probe complexity, where δ is the proportion of empty slots
- **Improved Performance**: Bypasses the "coupon collector" bottleneck that limits traditional uniform probing
- **Two Implementations**:
  - `ElasticHashMap`: Single-threaded implementation with optimal lookup performance
  - `ConcurrentElasticMap`: Thread-safe implementation using atomic operations

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
bathroom = "0.1.0"
```

## Usage

### Basic Usage

```rust
use bathroom::ElasticHashMap;

// Create a new hash map
let mut map = ElasticHashMap::new();

// Insert values
map.insert("apple".to_string(), 1);
map.insert("banana".to_string(), 2);
map.insert("cherry".to_string(), 3);

// Retrieve values
assert_eq!(map.get("apple"), Some(&1));

// Update values
map.insert("apple".to_string(), 10);
assert_eq!(map.get("apple"), Some(&10));

// Remove values
map.remove("apple");
assert_eq!(map.get("apple"), None);

// Check if key exists
assert!(map.contains_key("banana"));

// Get all keys or values
let keys = map.keys();
let values = map.values();
```

### Concurrent Usage

```rust
use bathroom::ConcurrentElasticMap;
use std::sync::Arc;
use std::thread;

// Create a shared hash map
let map = Arc::new(ConcurrentElasticMap::new());

// Clone references for different threads
let map1 = Arc::clone(&map);
let map2 = Arc::clone(&map);

// Spawn threads that modify the map concurrently
let t1 = thread::spawn(move || {
    for i in 0..100 {
        map1.insert(format!("key-{}", i), i);
    }
});

let t2 = thread::spawn(move || {
    for i in 100..200 {
        map2.insert(format!("key-{}", i), i);
    }
});

// Wait for threads to complete
t1.join().unwrap();
t2.join().unwrap();

// The map now contains all inserted values
assert_eq!(map.len(), 200);
```

## How It Works

The Bathroom Model introduces a novel approach to hash table probing that outperforms traditional methods:

### Core Concepts

1. **Dynamic Step Size Adjustment**: Unlike fixed-step probing methods (linear, quadratic), the step size dynamically changes based on what we observe during probing:
   - When encountering occupied slots, the step size increases exponentially to efficiently skip over clusters
   - When finding empty or deleted slots, the step size decreases to focus search in promising areas

2. **Occupancy-Based Decisions**: The algorithm tracks consecutive occupied slots and adjusts its strategy based on observed density, similar to how people might skip a crowded section of bathroom stalls.

3. **Tombstone Optimization**: Deleted slots (tombstones) are tracked and reused efficiently during insertion.

### The Bathroom Analogy

The name "Bathroom Model" comes from the real-world analogy of finding an empty bathroom stall:

- In a mostly empty bathroom, you simply take the first available stall (small step size)
- When encountering a section of occupied stalls, you tend to move further down to skip the cluster (increasing step size)
- If you see a recently vacated stall, you might consider using it (tombstone reuse)
- Your search strategy adapts based on observed occupancy patterns (dynamic adjustment)

This intuitive human behavior, when formalized and applied to hash table probing, results in significantly improved performance.

### Theoretical Advantages

This approach overcomes the "coupon collector" bottleneck that limits traditional uniform probing, which requires Ω(log δ⁻¹) probes on average. The Bathroom Model achieves:

- O(1) amortized expected probe complexity
- O(log δ⁻¹) worst-case expected probe complexity 

These bounds are optimal for hash tables that do not reorder elements.

## Performance Tuning

Both implementations offer configuration options to tune performance:

```rust
// For ElasticHashMap
let mut map = ElasticHashMap::new();
map.set_load_factor_threshold(80);  // Set maximum load factor to 80%
map.set_occupancy_threshold(3);     // Set consecutive occupancy threshold to 3

// For ConcurrentElasticMap
let map = ConcurrentElasticMap::new();
map.set_load_factor_threshold(0.8); // Set maximum load factor to 80%
map.set_occupancy_threshold(3);     // Set consecutive occupancy threshold to 3
```

## Implementation Details

- **Memory Efficiency**: Uses standard Rust vectors for storage with minimal overhead
- **Resize Strategy**: Automatically resizes when the load factor exceeds the threshold
- **Thread Safety**: The concurrent implementation uses atomic operations to ensure thread safety without traditional locks
- **Customizable Parameters**: Allows fine-tuning of occupancy thresholds and load factors

## Future Work

- [ ] Implement explicit segmentation as described in the original papers
- [ ] Add specialized batch insertion methods
- [ ] Develop comprehensive benchmarking suite
- [ ] Optimize memory usage patterns
- [ ] Add support for no-std environments

## References

1. Wang, Q. (2025). "The Bathroom Model: A Realistic Approach to Hash Table Algorithm Optimization." Vanderbilt University.
2. Farach-Colton, M., Krapivin, A., & Kuszmaul, W. (2025). "Optimal Bounds for Open Addressing Without Reordering." arXiv:2501.02305v1.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is dual-licensed under either:

- MIT License
- Apache License, Version 2.0

at your option.

