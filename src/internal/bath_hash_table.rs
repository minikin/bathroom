#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::arithmetic_side_effects)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::pedantic)]
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::unnecessary_min_or_max)]
#![allow(warnings)]

use plotters::prelude::*;
use rand::Rng;
use std::mem::size_of_val;

// Constants matching the Python implementation
const TABLE_SIZE: usize = 1_000_000;
// Create load factors from 0.1 to 0.95 with 10 steps
const NUM_LOAD_FACTORS: usize = 10;

// Hash table methods to compare
const METHODS: [&str; 4] =
    ["Random Probing", "Elastic Hashing", "Funnel Hashing", "Bathroom Model"];
const MAX_PROBES: usize = 100; // Prevent infinite loops

// Simple hash function for simulation purposes
fn hash_function(key: usize, size: usize) -> usize {
    key % size
}

// Estimate memory usage of the hash table (in bytes)
fn get_memory_usage(table: &Vec<Option<usize>>) -> usize {
    // This is a simple approximation - in a real system we would need to account for
    // the Vec's capacity, alignment, etc.
    let vec_size = size_of_val(table);

    // Count both filled and empty slots differently to better show algorithmic differences
    let filled_slots = table.iter().filter(|item| item.is_some()).count();
    let empty_slots = table.len() - filled_slots;

    // Calculate total memory size with weighted accounting
    // Empty slots still take up some memory, but less than filled slots
    let filled_memory = filled_slots * std::mem::size_of::<usize>();
    let empty_memory = empty_slots * std::mem::size_of::<Option<()>>(); // Just the discriminant

    vec_size + filled_memory + empty_memory
}

// Traditional random probing method
fn random_probing(table: &mut Vec<Option<usize>>, key: usize) -> usize {
    let mut rng = rand::rng();
    let mut index = hash_function(key, TABLE_SIZE);
    let mut probes = 1; // Start with first probe attempt

    while table[index].is_some() && probes < MAX_PROBES {
        index = (index + rng.random_range(1..TABLE_SIZE)) % TABLE_SIZE;
        probes += 1;
    }

    if table[index].is_none() {
        table[index] = Some(key);
    }

    probes
}

// Elastic hashing method based on the paper
fn elastic_hashing(table: &mut Vec<Option<usize>>, key: usize) -> usize {
    let mut index = hash_function(key, TABLE_SIZE);
    let mut probes = 1; // Start with first probe attempt
    let mut jump = 1;

    while table[index].is_some() && probes < MAX_PROBES {
        index = (index + jump) % TABLE_SIZE;
        // Limit jump size to prevent overflow and ensure it doesn't exceed table size
        jump = (jump.saturating_mul(2)).min(TABLE_SIZE / 2);
        probes += 1;
    }

    if table[index].is_none() {
        table[index] = Some(key);
    }

    probes
}

// Funnel hashing method, dividing table into decreasing size regions
fn funnel_hashing(table: &mut Vec<Option<usize>>, key: usize) -> usize {
    let mut rng = rand::rng();
    let mut index = hash_function(key, TABLE_SIZE);
    let mut probes = 1; // Start with first probe attempt
    let mut sub_table_size = TABLE_SIZE;

    while table[index].is_some() && probes < MAX_PROBES {
        sub_table_size = sub_table_size / 2.max(2); // Ensure sub_table_size is at least 2
        // Ensure we have a valid range for random number generation
        let step = if sub_table_size <= 1 { 1 } else { rng.random_range(1..sub_table_size) };
        index = (index + step) % TABLE_SIZE;
        probes += 1;
    }

    if table[index].is_none() {
        table[index] = Some(key);
    }

    probes
}

// Bathroom Model: Adaptive dynamic probing
fn bathroom_model(table: &mut Vec<Option<usize>>, key: usize) -> usize {
    let mut index = hash_function(key, TABLE_SIZE);
    let mut probes = 1; // Start with first probe attempt
    let mut step_size = 1; // Start with a small step

    while table[index].is_some() && probes < MAX_PROBES {
        step_size = (step_size * 2).min(TABLE_SIZE / 4); // Adaptive step growth
        index = (index + step_size) % TABLE_SIZE;
        probes += 1;
    }

    if table[index].is_none() {
        table[index] = Some(key);
    }

    probes
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate load factors from 0.1 to 0.95
    let load_factors: Vec<f64> = (0..NUM_LOAD_FACTORS)
        .map(|i| 0.1 + (0.95 - 0.1) * (i as f64) / ((NUM_LOAD_FACTORS - 1) as f64))
        .collect();

    // Calculate number of keys for each load factor
    let num_keys: Vec<usize> =
        load_factors.iter().map(|&load| (TABLE_SIZE as f64 * load) as usize).collect();

    println!("Load factors: {:?}", load_factors);
    println!("Number of keys: {:?}", num_keys);

    // Results storage
    let mut average_lookup_time: Vec<Vec<f64>> = vec![Vec::new(); METHODS.len()];
    let mut worst_case_probes: Vec<Vec<usize>> = vec![Vec::new(); METHODS.len()];
    let mut memory_utilization: Vec<Vec<usize>> = vec![Vec::new(); METHODS.len()];

    // Generate random keys outside the loop to ensure fair comparison
    let mut rng = rand::rng();
    let max_keys_needed = *num_keys.iter().max().unwrap();
    let keys: Vec<usize> = (0..max_keys_needed).map(|_| rng.random_range(1..1_000_000)).collect();

    // Running experiments
    for &n_keys in &num_keys {
        println!("Testing with {} keys", n_keys);

        for (method_idx, &method) in METHODS.iter().enumerate() {
            let mut table: Vec<Option<usize>> = vec![None; TABLE_SIZE];
            let mut probes_list: Vec<usize> = Vec::with_capacity(n_keys);

            for &key in keys.iter().take(n_keys) {
                let probes = match method {
                    "Random Probing" => random_probing(&mut table, key),
                    "Elastic Hashing" => elastic_hashing(&mut table, key),
                    "Funnel Hashing" => funnel_hashing(&mut table, key),
                    "Bathroom Model" => bathroom_model(&mut table, key),
                    _ => panic!("Unknown method"),
                };
                probes_list.push(probes);
            }

            // Calculate statistics
            let avg_lookup = probes_list.iter().sum::<usize>() as f64 / probes_list.len() as f64;
            let worst_case = *probes_list.iter().max().unwrap_or(&0);
            let memory_usage = get_memory_usage(&table);

            // Store results
            average_lookup_time[method_idx].push(avg_lookup);
            worst_case_probes[method_idx].push(worst_case);
            memory_utilization[method_idx].push(memory_usage);

            println!(
                "  {}: Avg probes = {:.2}, Worst = {}, Memory = {} bytes",
                method, avg_lookup, worst_case, memory_usage
            );
        }
    }

    // Enhanced plot configuration
    let font_family = "sans-serif";

    // Enhanced colors with better contrast
    let colors = [
        RGBColor(220, 50, 50),  // Bright red
        RGBColor(50, 90, 220),  // Bright blue
        RGBColor(50, 180, 50),  // Bright green
        RGBColor(180, 50, 180), // Bright magenta
    ];

    // High-quality rendering settings
    let line_width = 2;
    let marker_size = 4;
    let text_size = 16;
    let title_size = 35;

    // Plot 1: Average Lookup Time - Higher resolution
    let root = BitMapBackend::new("average_lookup_time.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_avg = average_lookup_time
        .iter()
        .flat_map(|v| v.iter())
        .fold(0.0, |max, &x| if x > max { x } else { max }) *
        1.1; // Add 10% margin

    let mut chart = ChartBuilder::on(&root)
        .caption("Comparison of Hash Table Lookup Efficiency", (font_family, title_size))
        .margin(15)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .right_y_label_area_size(10)
        .build_cartesian_2d(0..(num_keys.len() - 1), 0.0..max_avg)?;

    // Create custom x-axis labels
    let x_labels: Vec<String> = num_keys.iter().map(|&n| n.to_string()).collect();

    chart
        .configure_mesh()
        .x_labels(num_keys.len() - 1)
        .x_label_formatter(&|x| {
            if *x < x_labels.len() { x_labels[*x].clone() } else { "".to_string() }
        })
        .x_desc("Number of Keys Inserted")
        .y_desc("Average Lookup Time (probes)")
        .axis_desc_style((font_family, text_size))
        .draw()?;

    // Add a vertical line at critical load factor (~70%)
    let critical_load_idx = num_keys.len() * 7 / 10;
    if critical_load_idx < num_keys.len() - 1 {
        // Create a thin dashed line with proper styling
        let reference_style = ShapeStyle::from(&BLACK.mix(0.3)).stroke_width(1);
        chart
            .draw_series(LineSeries::new(
                vec![(critical_load_idx, 0.0), (critical_load_idx, max_avg)],
                reference_style,
            ))?
            .label("~70% Load Factor")
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], reference_style));
    }

    // Draw lines for each method
    for (method_idx, &method) in METHODS.iter().enumerate() {
        let color = &colors[method_idx % colors.len()];
        // Create style with proper stroke width
        let line_style = ShapeStyle::from(color).stroke_width(line_width);

        // Draw the line with increased thickness
        chart
            .draw_series(LineSeries::new(
                (0..num_keys.len() - 1).map(|i| (i, average_lookup_time[method_idx][i])),
                line_style,
            ))?
            .label(method)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], line_style));

        // Add larger point markers
        chart.draw_series(
            (0..num_keys.len() - 1)
                .step_by(1) // Add marker at every point for better visibility
                .map(|i| {
                    Circle::new(
                        (i, average_lookup_time[method_idx][i]),
                        marker_size,
                        color.filled(),
                    )
                }),
        )?;
    }

    // Add annotation for performance degradation
    if num_keys.len() > 6 {
        let high_load_idx = num_keys.len() - 3;
        let max_method_idx = (0..METHODS.len())
            .max_by(|&a, &b| {
                average_lookup_time[a][high_load_idx]
                    .partial_cmp(&average_lookup_time[b][high_load_idx])
                    .unwrap()
            })
            .unwrap();

        chart.draw_series(std::iter::once(Text::new(
            "Performance drops at high load factors",
            (high_load_idx, average_lookup_time[max_method_idx][high_load_idx] * 0.9),
            (font_family, text_size),
        )))?;
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    // Plot 2: Worst-Case Probing - Higher resolution
    let root = BitMapBackend::new("worst_case_probes.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_worst = worst_case_probes
        .iter()
        .flat_map(|v| v.iter())
        .fold(0, |max, &x| if x > max { x } else { max }) as f64 *
        1.1; // Add 10% margin

    let mut chart = ChartBuilder::on(&root)
        .caption("Comparison of Worst-Case Probing", (font_family, title_size))
        .margin(15)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .right_y_label_area_size(10)
        .build_cartesian_2d(0..(num_keys.len() - 1), 0.0..max_worst)?;

    chart
        .configure_mesh()
        .x_labels(num_keys.len() - 1)
        .x_label_formatter(&|x| {
            if *x < x_labels.len() { x_labels[*x].clone() } else { "".to_string() }
        })
        .x_desc("Number of Keys Inserted")
        .y_desc("Worst-Case Probe Complexity")
        .axis_desc_style((font_family, text_size))
        .draw()?;

    // Add threshold line for acceptable probe count (MAX_PROBES/2)
    let threshold_style = ShapeStyle::from(&RED.mix(0.3)).stroke_width(1);
    chart
        .draw_series(LineSeries::new(
            vec![(0, MAX_PROBES as f64 / 2.0), (num_keys.len() - 1, MAX_PROBES as f64 / 2.0)],
            threshold_style,
        ))?
        .label("Warning Threshold")
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], threshold_style));

    // Draw lines for each method
    for (method_idx, &method) in METHODS.iter().enumerate() {
        let color = &colors[method_idx % colors.len()];
        let line_style = ShapeStyle::from(color).stroke_width(line_width);

        // Draw the line with increased thickness
        chart
            .draw_series(LineSeries::new(
                (0..num_keys.len() - 1).map(|i| (i, worst_case_probes[method_idx][i] as f64)),
                line_style,
            ))?
            .label(method)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], line_style));

        // Add larger point markers
        chart.draw_series(
            (0..num_keys.len() - 1)
                .step_by(1) // Add marker at every point for better visibility
                .map(|i| {
                    Circle::new(
                        (i, worst_case_probes[method_idx][i] as f64),
                        marker_size,
                        color.filled(),
                    )
                }),
        )?;
    }

    // Add annotation for hitting MAX_PROBES
    if num_keys.len() > 5 {
        let high_load_idx = num_keys.len() - 3;
        let max_method_idx = (0..METHODS.len())
            .filter(|&i| worst_case_probes[i][high_load_idx] >= MAX_PROBES / 2)
            .min_by_key(|&i| i)
            .unwrap_or(0);

        if worst_case_probes[max_method_idx][high_load_idx] >= MAX_PROBES / 2 {
            chart.draw_series(std::iter::once(Text::new(
                "Some methods reach MAX_PROBES",
                (high_load_idx, worst_case_probes[max_method_idx][high_load_idx] as f64 * 1.1),
                (font_family, text_size),
            )))?;
        }
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    // Plot 3: Memory Utilization - Higher resolution
    let root = BitMapBackend::new("memory_utilization.png", (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    // Create two different views - use vertical split
    let areas = root.split_evenly((2, 1));

    // Overall memory utilization plot
    let max_memory = memory_utilization
        .iter()
        .flat_map(|v| v.iter())
        .fold(0, |max, &x| if x > max { x } else { max }) as f64 *
        1.1; // Add 10% margin

    let mut overall_chart = ChartBuilder::on(&areas[0])
        .caption("Overall Memory Utilization", (font_family, title_size))
        .margin(15)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .right_y_label_area_size(10)
        .build_cartesian_2d(0..(num_keys.len() - 1), 0.0..max_memory)?;

    overall_chart
        .configure_mesh()
        .x_labels(num_keys.len() - 1)
        .x_label_formatter(&|x| {
            if *x < x_labels.len() { x_labels[*x].clone() } else { "".to_string() }
        })
        .x_desc("Number of Keys Inserted")
        .y_desc("Memory Utilization (bytes)")
        .axis_desc_style((font_family, text_size))
        .draw()?;

    // Draw lines for each method
    for (method_idx, &method) in METHODS.iter().enumerate() {
        let color = &colors[method_idx % colors.len()];
        let line_style = ShapeStyle::from(color).stroke_width(line_width);

        // Draw the line with increased thickness
        overall_chart
            .draw_series(LineSeries::new(
                (0..num_keys.len() - 1).map(|i| (i, memory_utilization[method_idx][i] as f64)),
                line_style,
            ))?
            .label(method)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], line_style));

        // Add larger point markers
        overall_chart.draw_series(
            (0..num_keys.len() - 1)
                .step_by(1) // Add marker at every point for better visibility
                .map(|i| {
                    Circle::new(
                        (i, memory_utilization[method_idx][i] as f64),
                        marker_size,
                        color.filled(),
                    )
                }),
        )?;
    }

    overall_chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    // Detailed view focusing on the high load factors (last 3-4 data points)
    // First calculate a reasonable y-axis range for the zoomed view
    let high_load_start = num_keys.len().saturating_sub(4); // Start index for high load

    // Find min/max for the detailed view with a safety buffer
    let min_memory_high_load = memory_utilization
        .iter()
        .flat_map(|v| v[high_load_start..].iter())
        .fold(usize::MAX, |min, &x| if x < min { x } else { min })
        as f64 *
        0.995; // Slightly below the minimum

    let max_memory_high_load = memory_utilization
        .iter()
        .flat_map(|v| v[high_load_start..].iter())
        .fold(0, |max, &x| if x > max { x } else { max }) as f64 *
        1.005; // Slightly above the maximum

    let mut detailed_chart = ChartBuilder::on(&areas[1])
        .caption(
            "Memory Utilization at High Load Factors (Detailed View)",
            (font_family, title_size),
        )
        .margin(15)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .right_y_label_area_size(10)
        .build_cartesian_2d(
            high_load_start..(num_keys.len() - 1),
            min_memory_high_load..max_memory_high_load,
        )?;

    detailed_chart
        .configure_mesh()
        .x_labels(num_keys.len() - high_load_start)
        .x_label_formatter(&|x| {
            if *x < x_labels.len() { x_labels[*x].clone() } else { "".to_string() }
        })
        .x_desc("Number of Keys Inserted")
        .y_desc("Memory Utilization (bytes)")
        .axis_desc_style((font_family, text_size))
        .draw()?;

    // Draw lines for each method in the detailed view
    for (method_idx, &method) in METHODS.iter().enumerate() {
        let color = &colors[method_idx % colors.len()];
        let detail_line_style = ShapeStyle::from(color).stroke_width(line_width + 1);

        // Draw the line with increased thickness
        detailed_chart
            .draw_series(LineSeries::new(
                (high_load_start..num_keys.len() - 1)
                    .map(|i| (i, memory_utilization[method_idx][i] as f64)),
                detail_line_style,
            ))?
            .label(method)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], detail_line_style));

        // Add larger point markers for every point in detailed view
        detailed_chart.draw_series((high_load_start..num_keys.len() - 1).map(|i| {
            Circle::new(
                (i, memory_utilization[method_idx][i] as f64),
                marker_size + 1,
                color.filled(),
            )
        }))?;
    }

    // Add annotations for memory efficiency insights
    if high_load_start + 2 < num_keys.len() {
        let last_idx = num_keys.len() - 2;

        // Find methods with best and worst memory usage
        let best_method_idx =
            (0..METHODS.len()).min_by_key(|&i| memory_utilization[i][last_idx]).unwrap_or(0);

        let worst_method_idx =
            (0..METHODS.len()).max_by_key(|&i| memory_utilization[i][last_idx]).unwrap_or(0);

        detailed_chart.draw_series(std::iter::once(Text::new(
            format!("{}: Most memory efficient", METHODS[best_method_idx]),
            (last_idx - 1, memory_utilization[best_method_idx][last_idx] as f64 * 0.998),
            (font_family, text_size),
        )))?;

        detailed_chart.draw_series(std::iter::once(Text::new(
            format!("{}: Least memory efficient", METHODS[worst_method_idx]),
            (last_idx - 1, memory_utilization[worst_method_idx][last_idx] as f64 * 1.002),
            (font_family, text_size),
        )))?;
    }

    detailed_chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    println!(
        "Generated high-quality plot images: average_lookup_time.png, worst_case_probes.png, memory_utilization.png"
    );

    Ok(())
}
