use pgrx::prelude::*;
pgrx::pg_module_magic!();
use serde_json::json;
use std::collections::HashMap;

pub mod bindings;
pub mod utils;

extern crate serde_derive;

/*
 * @param mini_batch: mini_batch of data. Assume all columns are string type in
 * libsvm codding
 */
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "profiling_filtering_phase")]
#[allow(unused_variables)]
pub fn profiling_filtering_phase(mini_batch: String, config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("mini_batch", mini_batch);
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::ms::profiling_filtering_phase(&task_json).to_string()
}

/*
 * @param mini_batch: training for one iteration.
 * libsvm codding
 */
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "profiling_refinement_phase")]
#[allow(unused_variables)]
pub fn profiling_refinement_phase(mini_batch: String, config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("mini_batch", mini_batch);
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::ms::profiling_refinement_phase(&task_json).to_string()
}

/*
 * @param mini_batch: training for one iteration.
 * libsvm codding
 */
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "coordinator")]
#[allow(unused_variables)]
pub fn coordinator(time_score: String, time_train: String, time_budget: String, only_phase1: bool,
                   config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("budget", time_budget);
    task_map.insert("score_time_per_model", time_score);
    task_map.insert("train_time_per_epoch", time_train);
    task_map.insert("only_phase1", only_phase1.to_string());
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::ms::coordinator(&task_json).to_string()
}


/*
 * @param mini_batch: mini_batch of data. Assume all columns are string type in
 * libsvm codding
 */
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "filtering_phase")]
#[allow(unused_variables)]
pub fn filtering_phase(mini_batch: String, n: i32, k: i32, config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("mini_batch", mini_batch);
    task_map.insert("n", n.to_string());
    task_map.insert("k", k.to_string());
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::ms::filtering_phase(&task_json).to_string()
}


#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "refinement_phase")]
#[allow(unused_variables)]
pub fn refinement_phase(config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::ms::refinement_phase().to_string()
}


/*
 End-2-End model selection, All in UDF runtime.
 */
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "model_selection")]
#[allow(unused_variables)]
pub fn model_selection(mini_batch: String, time_budget: String, config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("mini_batch", mini_batch);
    task_map.insert("budget", time_budget);
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::ms::model_selection(&task_json).to_string()
}

/*
 * @param mini_batch: mini_batch of data. Assume all columns are string type in
 * libsvm codding
 */
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "model_selection_workloads")]
#[allow(unused_variables)]
pub fn model_selection_workloads(mini_batch: String, n: i32, k: i32, config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("mini_batch", mini_batch);
    task_map.insert("n", n.to_string());
    task_map.insert("k", k.to_string());
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::ms::model_selection_workloads(&task_json).to_string()
}


// this two are filtering + refinement in GPU server
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "model_selection_trails")]
#[allow(unused_variables)]
pub fn model_selection_trails(mini_batch: String, time_budget: String, config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("mini_batch", mini_batch);
    task_map.insert("budget", time_budget);
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::ms::model_selection_trails(&task_json).to_string()
}


#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "model_selection_trails_workloads")]
#[allow(unused_variables)]
pub fn model_selection_trails_workloads(mini_batch: String, n: i32, k: i32, config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("mini_batch", mini_batch);
    task_map.insert("n", n.to_string());
    task_map.insert("k", k.to_string());
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::ms::model_selection_trails_workloads(&task_json).to_string()
}

// micro benchmarks
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "benchmark_filtering_phase_latency")]
#[allow(unused_variables)]
pub fn benchmark_filtering_phase_latency(explore_models: i32, config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("explore_models", explore_models.to_string());
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::ms::benchmark_filtering_phase_latency(&task_json).to_string()
}

#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "benchmark_filtering_latency_in_db")]
#[allow(unused_variables)]
pub fn benchmark_filtering_latency_in_db(
    explore_models: i32, dataset: String, batch_size_m: i32, config_file: String) -> String {
    crate::bindings::ms::benchmark_filtering_latency_in_db(explore_models, &dataset, batch_size_m, &config_file).to_string()
}


// Model Inference
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "inference")]
#[allow(unused_variables)]
pub fn run_inference(
    dataset: String,
    condition: String,
    config_file: String,
    col_cardinalities_file: String,
    model_path: String,
    sql: String,
    batch_size: i32,
) -> String {
    crate::bindings::inference::run_inference(
        &dataset,
        &condition,
        &config_file,
        &col_cardinalities_file,
        &model_path,
        &sql,
        batch_size).to_string()
}

// Model Inference
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "inference_shared")]
#[allow(unused_variables)]
pub fn run_inference_shared(
    dataset: String,
    condition: String,
    config_file: String,
    col_cardinalities_file: String,
    model_path: String,
    sql: String,
    batch_size: i32,
) -> String {
    crate::bindings::inference::run_inference_shared_memory(
        &dataset,
        &condition,
        &config_file,
        &col_cardinalities_file,
        &model_path,
        &sql,
        batch_size).to_string()
}

// Model Inference
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "inference_shared_write_once")]
#[allow(unused_variables)]
pub fn inference_shared_write_once(
    dataset: String,
    condition: String,
    config_file: String,
    col_cardinalities_file: String,
    model_path: String,
    sql: String,
    batch_size: i32,
) -> String {
    crate::bindings::inference::run_inference_shared_memory_write_once(
        &dataset,
        &condition,
        &config_file,
        &col_cardinalities_file,
        &model_path,
        &sql,
        batch_size).to_string()
}

// Model Inference
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "inference_shared_write_once_int")]
#[allow(unused_variables)]
pub fn inference_shared_write_once_int(
    dataset: String,
    condition: String,
    config_file: String,
    col_cardinalities_file: String,
    model_path: String,
    sql: String,
    batch_size: i32,
) -> String {
    crate::bindings::inference::run_inference_shared_memory_write_once_int(
        &dataset,
        &condition,
        &config_file,
        &col_cardinalities_file,
        &model_path,
        &sql,
        batch_size).to_string()
}

// Model Inference
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "run_inference_shared_memory_write_once_int_join")]
#[allow(unused_variables)]
pub fn run_inference_shared_memory_write_once_int_join(
    dataset: String,
    condition: String,
    config_file: String,
    col_cardinalities_file: String,
    model_path: String,
    sql: String,
    batch_size: i32,
) -> String {
    crate::bindings::inference::run_inference_shared_memory_write_once_int_join(
        &dataset,
        &condition,
        &config_file,
        &col_cardinalities_file,
        &model_path,
        &sql,
        batch_size).to_string()
}


// Model Inference
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "model_init")]
#[allow(unused_variables)]
pub fn model_init(
    condition: String,
    config_file: String,
    col_cardinalities_file: String,
    model_path: String,
) -> String {
    crate::bindings::inference::init_model(
        &condition,
        &config_file,
        &col_cardinalities_file,
        &model_path).to_string()
}


// Model Inference
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "run_inference_profiling")]
#[allow(unused_variables)]
pub fn run_inference_profiling(
    func_num: i32,
    dataset: String,
    condition: String,
    config_file: String,
    col_cardinalities_file: String,
    model_path: String,
    sql: String,
    batch_size: i32,
) -> String {
    match func_num {
        1 => match crate::bindings::inference::run_inference_w_all_opt_workloads(
            &dataset,
            &condition,
            &config_file,
            &col_cardinalities_file,
            &model_path,
            &sql,
            batch_size,
        ) {
            Ok(_) => serde_json::json!("ok").to_string(),
            Err(e) => serde_json::json!({ "error": format!("Error: {}", e) }).to_string(),
        },

        // without model caching
        2 => match crate::bindings::inference::run_inference_wo_cache_workloads(
            &dataset,
            &condition,
            &config_file,
            &col_cardinalities_file,
            &model_path,
            &sql,
            batch_size,
        ) {
            Ok(_) => serde_json::json!("ok").to_string(),
            Err(e) => serde_json::json!({ "error": format!("Error: {}", e) }).to_string(),
        },

        // without memory cache
        3 => match crate::bindings::inference::run_inference_wo_memoryshare_workloads(
            &dataset,
            &condition,
            &config_file,
            &col_cardinalities_file,
            &model_path,
            &sql,
            batch_size,
        ) {
            Ok(_) => serde_json::json!("ok").to_string(),
            Err(e) => serde_json::json!({ "error": format!("Error: {}", e) }).to_string(),
        },

        // invesgate memory usage
        4 => match crate::bindings::inference::invesgate_memory_usage(
            &dataset,
            &condition,
            &config_file,
            &col_cardinalities_file,
            &model_path,
            &sql,
            batch_size,
        ) {
            Ok(_) => serde_json::json!("ok").to_string(),
            Err(e) => serde_json::json!({ "error": format!("Error: {}", e) }).to_string(),
        },

        // invesgate memory usage
        5 => match crate::bindings::inference::invesgate_memory_usage_record_only(
            &dataset,
            &condition,
            &config_file,
            &col_cardinalities_file,
            &model_path,
            &sql,
            batch_size,
        ) {
            Ok(_) => serde_json::json!("ok").to_string(),
            Err(e) => serde_json::json!({ "error": format!("Error: {}", e) }).to_string(),
        },

        _ => serde_json::json!({ "error": "Invalid function number" }).to_string(),
    }
}
