# Hash Table Implementation and Comparison

This project provides an implementation and visualization of four different hash table collision resolution methods. It compares their performance in terms of lookup efficiency, worst-case probing, and memory utilization.

- [Hash Table Implementation and Comparison](#hash-table-implementation-and-comparison)
  - [Collision Resolution Methods](#collision-resolution-methods)
  - [Requirements](#requirements)
  - [Usage](#usage)
  - [Understanding the Charts](#understanding-the-charts)
    - [Average Lookup Time](#average-lookup-time)
    - [Worst-Case Probing](#worst-case-probing)
    - [Memory Utilization](#memory-utilization)
  - [Performance Comparison Summary](#performance-comparison-summary)
  - [Customization](#customization)
  - [Troubleshooting](#troubleshooting)

## Collision Resolution Methods

1. **Random Probing**: Uses random steps to resolve collisions
2. **Elastic Hashing**: Dynamically adjusts probe step sizes based on table occupancy
3. **Funnel Hashing**: Uses a funnel-based approach to guide probe sequences
4. **Bathroom Model**: A novel approach optimizing memory usage and lookup efficiency

## Requirements

- Rust (stable version recommended, 1.50+)
- Cargo package manager
- Dependencies (automatically managed by Cargo):
  - `rand`: For random number generation
  - `plotters`: For visualization and chart generation

## Usage

Run the project to generate comparison charts:

```
cargo run --release --bin bath_hash_table
```

The program will:
1. Initialize hash tables with a range of load factors
2. Perform insertions and lookups for each method
3. Record performance metrics
4. Generate three visualization charts in the project directory:
   - `average_lookup_time.png`
   - `worst_case_probes.png`
   - `memory_utilization.png`

## Understanding the Charts

### Average Lookup Time
This chart shows the average number of probes needed to find an item in the hash table as the load factor increases. Lower values indicate better performance. Key observations:
- Random Probing and Funnel Hashing perform better at high load factors
- Performance degrades more rapidly after ~70% load factor for all methods

### Worst-Case Probing
This visualization shows the maximum number of probes needed to find an item. This represents the worst-case scenario for each method. Key insights:
- All methods eventually reach MAX_PROBES at very high load factors
- Some methods reach the worst-case threshold earlier than others

### Memory Utilization
This dual-view chart shows the memory usage of each method:
- The top chart shows overall memory usage across all load factors
- The bottom chart provides a detailed view at high load factors
- Memory efficiency differences between methods are relatively small (~1-2%)
- Elastic Hashing and Bathroom Model are slightly more memory-efficient

## Performance Comparison Summary

Based on the generated charts:

- **Best for Lookup Speed**: Random Probing and Funnel Hashing
- **Best for Memory Efficiency**: Elastic Hashing, followed closely by Bathroom Model
- **Most Balanced**: Funnel Hashing offers good lookup performance with moderate memory usage

## Customization

You can modify parameters in the source code to adjust:
- `TABLE_SIZE`: The size of the hash table
- `MAX_PROBES`: The maximum number of probes before giving up
- Test ranges and data distributions

## Troubleshooting

If you encounter errors related to the `plotters` library, you may need to install additional dependencies:

- On Ubuntu/Debian: `sudo apt-get install libfontconfig1-dev`
- On macOS: `brew install fontconfig`
- On Windows: No additional dependencies required
