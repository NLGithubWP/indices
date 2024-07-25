use serde_json::json;
use std::collections::HashMap;
use std::ffi::c_long;
use pgrx::prelude::*;
use crate::bindings::ml_register::PY_MODULE_INFERENCE;
use crate::bindings::ml_register::run_python_function;
use crate::utils::monitor::{start_memory_monitoring, record_memory_usage, start_memory_monitoring_handler};
use shared_memory::{ShmemConf};
use std::time::{Duration, Instant};
use std::thread::sleep;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::thread::{self, JoinHandle};
use sysinfo::{System, SystemExt};


pub fn run_inference_shared_memory(
    dataset: &String,
    condition: &String,
    config_file: &String,
    col_cardinalities_file: &String,
    model_path: &String,
    sql: &String,
    batch_size: i32,
) -> serde_json::Value {
    let mut response = HashMap::new();

    let overall_start_time = Instant::now();

    let mut last_id = 0;

    // Step 1: load model and columns etc
    let mut task_map = HashMap::new();
    task_map.insert("where_cond", condition.clone());
    task_map.insert("config_file", config_file.clone());
    task_map.insert("col_cardinalities_file", col_cardinalities_file.clone());
    task_map.insert("model_path", model_path.clone());
    let task_json = json!(task_map).to_string();
    // here it cache a state
    run_python_function(
        &PY_MODULE_INFERENCE,
        &task_json,
        "model_inference_load_model");

    let _end_time = Instant::now();
    let model_init_time = _end_time.duration_since(overall_start_time).as_secs_f64();
    response.insert("model_init_time", model_init_time.clone());

    // Step 2: query data via SPI
    let start_time = Instant::now();
    let results: Result<Vec<Vec<String>>, String> = Spi::connect(|client| {
        let query = format!("SELECT * FROM {}_train {} LIMIT {}",
                            dataset, sql, batch_size);
        let mut cursor = client.open_cursor(&query, None);
        let table = match cursor.fetch(batch_size as c_long) {
            Ok(table) => table,
            Err(e) => return Err(e.to_string()), // Convert the error to a string and return
        };

        let mut mini_batch = Vec::new();

        for row in table.into_iter() {
            let mut each_row = Vec::new();
            // add primary key
            let col0 = match row.get::<i32>(1) {
                Ok(Some(val)) => {
                    // Update last_id with the retrieved value
                    if val > 100000 {
                        last_id = 0;
                    } else {
                        last_id = val
                    }
                    val.to_string()
                }
                Ok(None) => "".to_string(), // Handle the case when there's no valid value
                Err(e) => e.to_string(),
            };
            each_row.push(col0);
            // add label
            let col1 = match row.get::<i32>(2) {
                Ok(val) => val.map(|i| i.to_string()).unwrap_or_default(),
                Err(e) => e.to_string(),
            };
            each_row.push(col1);
            // add fields
            let texts: Vec<String> = (3..row.columns() + 1)
                .filter_map(|i| {
                    match row.get::<&str>(i) {
                        Ok(Some(s)) => Some(s.to_string()),
                        Ok(None) => None,
                        Err(e) => Some(e.to_string()),  // Convert error to string
                    }
                }).collect();
            each_row.extend(texts);
            mini_batch.push(each_row)
        }
        // return
        Ok(mini_batch)
    });
    // serialize the mini-batch data
    let tup_table = match results {
        Ok(data) => {
            serde_json::json!({
                        "status": "success",
                        "data": data
                    })
        }
        Err(e) => {
            serde_json::json!({
                    "status": "error",
                    "message": format!("Error while connecting: {}", e)
                })
        }
    };
    let mini_batch_json = tup_table.to_string();

    let end_time = Instant::now();
    let data_query_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("data_query_time", data_query_time.clone());


    let start_time = Instant::now();
    // Set an identifier for the shared memory
    let shmem_name = "my_shared_memory";
    let my_shmem = ShmemConf::new()
        .size(tup_table.to_string().len())
        .os_id(shmem_name)
        .create()
        .unwrap();

    // Use unsafe to access and write to the raw memory
    let data_to_write = mini_batch_json.as_bytes();
    unsafe {
        // Get the raw pointer to the shared memory
        let shmem_ptr = my_shmem.as_ptr() as *mut u8;
        // Copy data into the shared memory
        std::ptr::copy_nonoverlapping(
            data_to_write.as_ptr(), shmem_ptr, data_to_write.len());
    }

    let end_time = Instant::now();
    let data_copy_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("data_copy", data_copy_time.clone());

    let start_time = Instant::now();
    // Step 3: model evaluate in Python
    let mut eva_task_map = HashMap::new();
    eva_task_map.insert("config_file", config_file.clone());
    eva_task_map.insert("spi_seconds", data_query_time.to_string());

    let eva_task_json = json!(eva_task_map).to_string(); // Corrected this line

    run_python_function(
        &PY_MODULE_INFERENCE,
        &eva_task_json,
        "model_inference_compute_shared_memory");

    let end_time = Instant::now();
    let python_compute_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("python_compute_time", python_compute_time.clone());

    let overall_end_time = Instant::now();
    let overall_elapsed_time = overall_end_time.duration_since(overall_start_time).as_secs_f64();
    let diff_time = model_init_time + data_query_time + data_copy_time + python_compute_time - overall_elapsed_time;

    response.insert("overall_query_latency", overall_elapsed_time.clone());
    response.insert("diff", diff_time.clone());

    // Step 4: Return to PostgresSQL
    return serde_json::json!(response);
}


/*
    This is infernce without all opts
 */
pub fn run_inference(
    dataset: &String,
    condition: &String,
    config_file: &String,
    col_cardinalities_file: &String,
    model_path: &String,
    sql: &String,
    batch_size: i32,
) -> serde_json::Value {
    let mut response = HashMap::new();

    let overall_start_time = Instant::now();

//     let mut last_id = 0;

    // Step 1: load model and columns etc
    let mut task_map = HashMap::new();
    task_map.insert("where_cond", condition.clone());
    task_map.insert("config_file", config_file.clone());
    task_map.insert("col_cardinalities_file", col_cardinalities_file.clone());
    task_map.insert("model_path", model_path.clone());
    let task_json = json!(task_map).to_string();
    // here it cache a state
    run_python_function(
        &PY_MODULE_INFERENCE,
        &task_json,
        "model_inference_load_model");

    let _end_time = Instant::now();
    let model_init_time = _end_time.duration_since(overall_start_time).as_secs_f64();
    response.insert("model_init_time", model_init_time.clone());

    // Step 2: query data via SPI
    let start_time = Instant::now();
    let mut all_rows = Vec::new();
    let _ = Spi::connect(|client| {
        let query = format!("SELECT * FROM {}_int_train {} LIMIT {}", dataset, sql, batch_size);
        let mut cursor = client.open_cursor(&query, None);
        let table = match cursor.fetch(batch_size as c_long) {
            Ok(table) => table,
            Err(e) => return Err(e.to_string()),
        };

        let end_time = Instant::now();
        let data_query_time_spi = end_time.duration_since(start_time).as_secs_f64();
        response.insert("data_query_time_spi", data_query_time_spi);

        // todo: nl: this part can must be optimized, since i go through all of those staff.
        for row in table.into_iter() {
            for i in 3..=row.columns() {
                match row.get::<i32>(i) {
                    Ok(Some(val)) => all_rows.push(val), // Handle the case when a valid i32 is obtained
                    Ok(None) => {
                        // Handle the case when the value is missing or erroneous
                        // For example, you can add a default value, like -1
                        all_rows.push(-1);
                    }
                    Err(e) => {
                        // Handle the error, e.g., log it or handle it in some way
                        eprintln!("Error fetching value: {:?}", e);
                    }
                }
            }
        }
        // Return OK or some status
        Ok(())
    });

    let mini_batch_json = serde_json::to_string(&all_rows).unwrap();

    let end_time = Instant::now();
    let data_query_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("data_query_time", data_query_time.clone());

    let start_time = Instant::now();
    // Step 3: model evaluate in Python
    let mut eva_task_map = HashMap::new();
    eva_task_map.insert("config_file", config_file.clone());
    eva_task_map.insert("mini_batch", mini_batch_json);
    eva_task_map.insert("spi_seconds", data_query_time.to_string());

    let eva_task_json = json!(eva_task_map).to_string(); // Corrected this line

    run_python_function(
        &PY_MODULE_INFERENCE,
        &eva_task_json,
        "model_inference_compute");

    let end_time = Instant::now();
    let python_compute_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("python_compute_time", python_compute_time.clone());

    let overall_end_time = Instant::now();
    let overall_elapsed_time = overall_end_time.duration_since(overall_start_time).as_secs_f64();
    let diff_time = model_init_time + data_query_time + python_compute_time - overall_elapsed_time;

    response.insert("overall_query_latency", overall_elapsed_time.clone());
    response.insert("diff", diff_time.clone());

    let response_json = json!(response).to_string();
    run_python_function(
        &PY_MODULE_INFERENCE,
        &response_json,
        "records_results");

    // Step 4: Return to PostgresSQL
    return serde_json::json!(response);
}


/*
    This is infernce with shared memory but on id:value pair
 */
pub fn run_inference_shared_memory_write_once(
    dataset: &String,
    condition: &String,
    config_file: &String,
    col_cardinalities_file: &String,
    model_path: &String,
    sql: &String,
    batch_size: i32,
) -> serde_json::Value {
    let mut response = HashMap::new();

    let overall_start_time = Instant::now();

    let mut last_id = 0;

    // Step 1: load model and columns etc
    let mut task_map = HashMap::new();
    task_map.insert("where_cond", condition.clone());
    task_map.insert("config_file", config_file.clone());
    task_map.insert("col_cardinalities_file", col_cardinalities_file.clone());
    task_map.insert("model_path", model_path.clone());
    let task_json = json!(task_map).to_string();
    // here it cache a state
    run_python_function(
        &PY_MODULE_INFERENCE,
        &task_json,
        "model_inference_load_model");

    let _end_time = Instant::now();
    let model_init_time = _end_time.duration_since(overall_start_time).as_secs_f64();
    response.insert("model_init_time", model_init_time.clone());

    // Step 2: query data via SPI
    let start_time = Instant::now();
    // Allocate shared memory in advance
    // Set an identifier for the shared memory
    let shmem_name = "my_shared_memory";

    // Pre-allocate a size for shared memory (this might need some logic to determine a reasonable size)
    let avg_row_size = 120;
    let shmem_size = (1.5 * (avg_row_size * batch_size as usize) as f64) as usize;
    let my_shmem = ShmemConf::new()
        .size(shmem_size)
        .os_id(shmem_name)
        .create()
        .unwrap();

    let shmem_ptr = my_shmem.as_ptr() as *mut u8;

    let end_time = Instant::now();
    let mem_allocate_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("mem_allocate_time", mem_allocate_time.clone());

    let start_time = Instant::now();
    // Use unsafe to access and write to the raw memory
    unsafe {
        let _ = Spi::connect(|client| {
            let query = format!("SELECT * FROM {}_train {} LIMIT {}", dataset, sql, batch_size);
            let mut cursor = client.open_cursor(&query, None);
            let table = match cursor.fetch(batch_size as c_long) {
                Ok(table) => table,
                Err(e) => return Err(e.to_string()),
            };

            let end_time = Instant::now();
            let data_query_time_spi = end_time.duration_since(start_time).as_secs_f64();
            response.insert("data_query_time_spi", data_query_time_spi.clone());

            let mut offset = 0;  // Keep track of how much we've written to shared memory

            // Write the opening square bracket
            shmem_ptr.offset(offset as isize).write(b"["[0]);
            offset += 1;

            let mut is_first_row = true;
            for row in table.into_iter() {

                // If not the first row, write a comma before the next row's data
                if !is_first_row {
                    shmem_ptr.offset(offset as isize).write(b","[0]);
                    offset += 1;
                } else {
                    is_first_row = false;
                }

                let mut each_row = Vec::new();
                // add primary key
                let col0 = match row.get::<i32>(1) {
                    Ok(Some(val)) => {
                        // Update last_id with the retrieved value
                        if val > 100000 {
                            last_id = 0;
                        } else {
                            last_id = val
                        }
                        val.to_string()
                    }
                    Ok(None) => "".to_string(), // Handle the case when there's no valid value
                    Err(e) => e.to_string(),
                };
                each_row.push(col0);
                // add label
                let col1 = match row.get::<i32>(2) {
                    Ok(val) => val.map(|i| i.to_string()).unwrap_or_default(),
                    Err(e) => e.to_string(),
                };
                each_row.push(col1);
                // add fields
                let texts: Vec<String> = (3..row.columns() + 1)
                    .filter_map(|i| {
                        match row.get::<&str>(i) {
                            Ok(Some(s)) => Some(s.to_string()),
                            Ok(None) => None,
                            Err(e) => Some(e.to_string()),  // Convert error to string
                        }
                    }).collect();
                each_row.extend(texts);

                // Serialize each row into shared memory
                let serialized_row = serde_json::to_string(&each_row).unwrap();
                let bytes = serialized_row.as_bytes();

                // Check if there's enough space left in shared memory
                if offset + bytes.len() > shmem_size {
                    // Handle error: not enough space in shared memory
                    return Err("Shared memory exceeded estimated size.".to_string());
                }

                // Copy the serialized row into shared memory
                std::ptr::copy_nonoverlapping(bytes.as_ptr(),
                                              shmem_ptr.offset(offset as isize),
                                              bytes.len());
                offset += bytes.len();
            }
            // Write the closing square bracket after all rows
            shmem_ptr.offset(offset as isize).write(b"]"[0]);

            // Return OK or some status
            Ok(())
        });
    }

    let end_time = Instant::now();
    let data_query_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("data_query_time", data_query_time.clone());

    let start_time = Instant::now();
    // Step 3: model evaluate in Python
    let mut eva_task_map = HashMap::new();
    eva_task_map.insert("config_file", config_file.clone());
    eva_task_map.insert("spi_seconds", data_query_time.to_string());

    let eva_task_json = json!(eva_task_map).to_string(); // Corrected this line

    run_python_function(
        &PY_MODULE_INFERENCE,
        &eva_task_json,
        "model_inference_compute_shared_memory_write_once");

    let end_time = Instant::now();
    let python_compute_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("python_compute_time", python_compute_time.clone());

    let overall_end_time = Instant::now();
    let overall_elapsed_time = overall_end_time.duration_since(overall_start_time).as_secs_f64();
    let diff_time = model_init_time + data_query_time + python_compute_time - overall_elapsed_time;

    response.insert("overall_query_latency", overall_elapsed_time.clone());
    response.insert("diff", diff_time.clone());


    let response_json = json!(response).to_string();
    run_python_function(
        &PY_MODULE_INFERENCE,
        &response_json,
        "records_results");

    // Step 4: Return to PostgresSQL
    return serde_json::json!(response);
}


/*
    This is infernce with shared memory but on id only
 */
pub fn run_inference_shared_memory_write_once_int(
    dataset: &String,
    condition: &String,
    config_file: &String,
    col_cardinalities_file: &String,
    model_path: &String,
    sql: &String,
    batch_size: i32,
) -> serde_json::Value {
    let mut response = HashMap::new();

    let mut num_columns: i32 = 0;
    match dataset.as_str() {  // assuming dataset is a String
        "frappe" => num_columns = 12,
        "adult" => num_columns = 15,
        "cvd" => num_columns = 13,
        "bank" => num_columns = 18,
        "census" => num_columns = 41 + 2,
        "credit" => num_columns = 23 + 2,
        "diabetes" => num_columns = 48 + 2,
        "hcdr" => num_columns = 69 + 2,
        "avazu" => num_columns = 22 + 2,
        _ => {}
    }

    let overall_start_time = Instant::now();

    // Step 1: load model and columns etc
    let mut task_map = HashMap::new();
    task_map.insert("where_cond", condition.clone());
    task_map.insert("config_file", config_file.clone());
    task_map.insert("col_cardinalities_file", col_cardinalities_file.clone());
    task_map.insert("model_path", model_path.clone());
    let task_json = json!(task_map).to_string();
    // here it cache a state
    run_python_function(
        &PY_MODULE_INFERENCE,
        &task_json,
        "model_inference_load_model");

    let _end_time = Instant::now();
    let model_init_time = _end_time.duration_since(overall_start_time).as_secs_f64();
    response.insert("model_init_time", model_init_time.clone());


    // Step 1: query data
    let start_time = Instant::now();
    let mut all_rows = Vec::new();
    let _ = Spi::connect(|client| {
        let query = format!("SELECT * FROM {}_int_train {} LIMIT {}", dataset, sql, batch_size);
        let mut cursor = client.open_cursor(&query, None);
        let table = match cursor.fetch(batch_size as c_long) {
            Ok(table) => table,
            Err(e) => return Err(e.to_string()),
        };
        let end_time = Instant::now();
        let data_query_time_spi = end_time.duration_since(start_time).as_secs_f64();
        response.insert("data_query_time_spi", data_query_time_spi);

        // todo: nl: this part can must be optimized, since i go through all of those staff.
        let start_time_3 = Instant::now();
        for row in table.into_iter() {
            for i in 3..=num_columns as usize {
                if let Ok(Some(val)) = row.get::<i32>(i) {
                    all_rows.push(val);
                }
            }
        }
        let end_time_min3 = Instant::now();
        let data_query_time_min3 = end_time_min3.duration_since(start_time_3).as_secs_f64();
        response.insert("data_type_convert_time", data_query_time_min3.clone());

        // Return OK or some status
        Ok(())
    });
    let end_time = Instant::now();
    let data_query_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("data_query_time", data_query_time.clone());


    // log the query datas
    // let serialized_row = serde_json::to_string(&all_rows).unwrap();
    // response_log.insert("query_data", serialized_row);

    // Step 3: Putting all data to he shared memory
    let start_time = Instant::now();
    let shmem_name = "my_shared_memory";
    let my_shmem = ShmemConf::new()
        .size(4 * all_rows.len())
        .os_id(shmem_name)
        .create()
        .unwrap();
    let shmem_ptr = my_shmem.as_ptr() as *mut i32;

    unsafe {
        // Copy data into shared memory
        std::ptr::copy_nonoverlapping(
            all_rows.as_ptr(),
            shmem_ptr as *mut i32,
            all_rows.len(),
        );
    }
    let end_time = Instant::now();
    let mem_allocate_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("mem_allocate_time", mem_allocate_time.clone());


    let start_time = Instant::now();
    // Step 3: model evaluate in Python
    let mut eva_task_map = HashMap::new();
    eva_task_map.insert("config_file", config_file.clone());
    eva_task_map.insert("spi_seconds", data_query_time.to_string());
    eva_task_map.insert("rows", batch_size.to_string());

    let eva_task_json = json!(eva_task_map).to_string(); // Corrected this line

    run_python_function(
        &PY_MODULE_INFERENCE,
        &eva_task_json,
        "model_inference_compute_shared_memory_write_once_int");

    let end_time = Instant::now();
    let python_compute_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("python_compute_time", python_compute_time.clone());

    let overall_end_time = Instant::now();
    let overall_elapsed_time = overall_end_time.duration_since(overall_start_time).as_secs_f64();
    let diff_time = model_init_time + data_query_time + python_compute_time - overall_elapsed_time;

    response.insert("overall_query_latency", overall_elapsed_time.clone());
    response.insert("diff", diff_time.clone());

    let response_json = json!(response).to_string();
    run_python_function(
        &PY_MODULE_INFERENCE,
        &response_json,
        "records_results");

    // Step 4: Return to PostgresSQL
    return serde_json::json!(response);
}


/*
    Init model
 */
pub fn init_model(
    condition: &String,
    config_file: &String,
    col_cardinalities_file: &String,
    model_path: &String,
) -> serde_json::Value {
    let overall_start_time = Instant::now();
    // Step 1: load model and columns etc
    let mut task_map = HashMap::new();
    task_map.insert("where_cond", condition.clone());
    task_map.insert("config_file", config_file.clone());
    task_map.insert("col_cardinalities_file", col_cardinalities_file.clone());
    task_map.insert("model_path", model_path.clone());
    let task_json = json!(task_map).to_string();
    // here it cache a state
    run_python_function(
        &PY_MODULE_INFERENCE,
        &task_json,
        "model_inference_load_model");
    let _end_time = Instant::now();
    let model_init_time = _end_time.duration_since(overall_start_time).as_secs_f64();
    return serde_json::json!(model_init_time);
}


/*
    Only for SPJ queries
 */
pub fn run_inference_shared_memory_write_once_int_join(
    dataset: &String,
    condition: &String,
    config_file: &String,
    col_cardinalities_file: &String,
    model_path: &String,
    sql: &String,
    batch_size: i32,
) -> serde_json::Value {
    let mut response = HashMap::new();

    let mut num_columns: i32 = 0;
    match dataset.as_str() {  // assuming dataset is a String
        "frappe" => num_columns = 12,
        "adult" => num_columns = 15,
        "cvd" => num_columns = 13,
        "bank" => num_columns = 18,
        "census" => num_columns = 41 + 2,
        "credit" => num_columns = 23 + 2,
        "diabetes" => num_columns = 48 + 2,
        "hcdr" => num_columns = 69 + 2,
        "avazu" => num_columns = 22 + 2,
        _ => {}
    }

    let mut spi_sql = String::new();
    match dataset.as_str() {
        "census" => {
            spi_sql = format!(
                "SELECT l.id, l.label, l.col1, l.col2, l.col3, l.col4, l.col5, l.col6, l.col7, l.col8, l.col9, l.col10,
             l.col11, l.col12, l.col13, l.col14, l.col15, l.col16, l.col17, l.col18, l.col19, l.col20,
             r.col21, r.col22, r.col23, r.col24, r.col25, r.col26, r.col27, r.col28, r.col29, r.col30,
             r.col31, r.col32, r.col33, r.col34, r.col35, r.col36, r.col37, r.col38, r.col39, r.col40, r.col41
             FROM {}_int_train_left l
             JOIN {}_int_train_right r ON l.id = r.id
             {} limit {};", dataset, dataset, sql, batch_size
            );
        }
        "credit" => {
            spi_sql = format!(
                "SELECT l.id, l.label, l.col1, l.col2, l.col3, l.col4, l.col5, l.col6, l.col7, l.col8, l.col9, l.col10, l.col11, l.col12,
             r.col13, r.col14, r.col15, r.col16, r.col17, r.col18, r.col19, r.col20, r.col21, r.col22, r.col23
             FROM {}_int_train_left l
             JOIN {}_int_train_right r ON l.id = r.id
             {} limit {};", dataset, dataset, sql, batch_size
            );
        }
        "diabetes" => {
            spi_sql = format!(
                "SELECT l.id, l.label, l.col1, l.col2, l.col3, l.col4, l.col5, l.col6, l.col7, l.col8, l.col9, l.col10, l.col11, l.col12, l.col13, l.col14, l.col15, l.col16, l.col17, l.col18, l.col19, l.col20, l.col21, l.col22, l.col23, l.col24,
             r.col25, r.col26, r.col27, r.col28, r.col29, r.col30, r.col31, r.col32, r.col33, r.col34, r.col35, r.col36, r.col37, r.col38, r.col39, r.col40, r.col41, r.col42, r.col43, r.col44, r.col45, r.col46, r.col47, r.col48
             FROM {}_int_train_left l
             JOIN {}_int_train_right r ON l.id = r.id
             {} limit {};", dataset, dataset, sql, batch_size
            );
        }
        "hcdr" => {
            spi_sql = format!(
                "SELECT l.id, l.label, l.col1, l.col2, l.col3, l.col4, l.col5, l.col6, l.col7, l.col8, l.col9, l.col10, l.col11, l.col12, l.col13, l.col14, l.col15, l.col16, l.col17, l.col18, l.col19, l.col20, l.col21, l.col22, l.col23, l.col24, l.col25, l.col26, l.col27, l.col28, l.col29, l.col30, l.col31, l.col32, l.col33, l.col34,
             r.col35, r.col36, r.col37, r.col38, r.col39, r.col40, r.col41, r.col42, r.col43, r.col44, r.col45, r.col46, r.col47, r.col48, r.col49, r.col50, r.col51, r.col52, r.col53, r.col54, r.col55, r.col56, r.col57, r.col58, r.col59, r.col60, r.col61, r.col62, r.col63, r.col64, r.col65, r.col66, r.col67, r.col68, r.col69
             FROM {}_int_train_left l
             JOIN {}_int_train_right r ON l.id = r.id
             {} limit {};", dataset, dataset, sql, batch_size
            );
        }
        _ => {}
    }

    let overall_start_time = Instant::now();

    // Step 1: load model and columns etc
    let mut task_map = HashMap::new();
    task_map.insert("where_cond", condition.clone());
    task_map.insert("config_file", config_file.clone());
    task_map.insert("col_cardinalities_file", col_cardinalities_file.clone());
    task_map.insert("model_path", model_path.clone());
    let task_json = json!(task_map).to_string();
    // here it cache a state
    run_python_function(
        &PY_MODULE_INFERENCE,
        &task_json,
        "model_inference_load_model");

    let _end_time = Instant::now();
    let model_init_time = _end_time.duration_since(overall_start_time).as_secs_f64();
    response.insert("model_init_time", model_init_time.clone());


    // Step 1: query data
    let start_time = Instant::now();
    let mut all_rows = Vec::new();
    let _ = Spi::connect(|client| {
        let query = spi_sql;
        let mut cursor = client.open_cursor(&query, None);
        let table = match cursor.fetch(batch_size as c_long) {
            Ok(table) => table,
            Err(e) => return Err(e.to_string()),
        };
        let end_time = Instant::now();
        let data_query_time_spi = end_time.duration_since(start_time).as_secs_f64();
        response.insert("data_query_time_spi", data_query_time_spi);

        // todo: nl: this part can must be optimized, since i go through all of those staff.
        let start_time_3 = Instant::now();
        for row in table.into_iter() {
            for i in 3..=num_columns as usize {
                if let Ok(Some(val)) = row.get::<i32>(i) {
                    all_rows.push(val);
                }
            }
        }
        let end_time_min3 = Instant::now();
        let data_query_time_min3 = end_time_min3.duration_since(start_time_3).as_secs_f64();
        response.insert("data_type_convert_time", data_query_time_min3.clone());

        // Return OK or some status
        Ok(())
    });
    let end_time = Instant::now();
    let data_query_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("data_query_time", data_query_time.clone());


    // log the query datas
    // let serialized_row = serde_json::to_string(&all_rows).unwrap();
    // response_log.insert("query_data", serialized_row);

    // Step 3: Putting all data to he shared memory
    let start_time = Instant::now();
    let shmem_name = "my_shared_memory";
    let my_shmem = ShmemConf::new()
        .size(4 * all_rows.len())
        .os_id(shmem_name)
        .create()
        .unwrap();
    let shmem_ptr = my_shmem.as_ptr() as *mut i32;

    unsafe {
        // Copy data into shared memory
        std::ptr::copy_nonoverlapping(
            all_rows.as_ptr(),
            shmem_ptr as *mut i32,
            all_rows.len(),
        );
    }
    let end_time = Instant::now();
    let mem_allocate_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("mem_allocate_time", mem_allocate_time.clone());


    let start_time = Instant::now();
    // Step 3: model evaluate in Python
    let mut eva_task_map = HashMap::new();
    eva_task_map.insert("config_file", config_file.clone());
    eva_task_map.insert("spi_seconds", data_query_time.to_string());
    eva_task_map.insert("rows", batch_size.to_string());

    let eva_task_json = json!(eva_task_map).to_string(); // Corrected this line

    run_python_function(
        &PY_MODULE_INFERENCE,
        &eva_task_json,
        "model_inference_compute_shared_memory_write_once_int");

    let end_time = Instant::now();
    let python_compute_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("python_compute_time", python_compute_time.clone());

    let overall_end_time = Instant::now();
    let overall_elapsed_time = overall_end_time.duration_since(overall_start_time).as_secs_f64();
    let diff_time = model_init_time + data_query_time + python_compute_time - overall_elapsed_time;

    response.insert("overall_query_latency", overall_elapsed_time.clone());
    response.insert("diff", diff_time.clone());

    let response_json = json!(response).to_string();
    run_python_function(
        &PY_MODULE_INFERENCE,
        &response_json,
        "records_results");

    // Step 4: Return to PostgresSQL
    return serde_json::json!(response);
}

/*
 How does the database performance, memory usage, and query processing times changes
 */
pub fn run_inference_w_all_opt_workloads(
    dataset: &String,
    condition: &String,
    config_file: &String,
    col_cardinalities_file: &String,
    model_path: &String,
    sql: &String,
    batch_size: i32,
) -> Result<(), String> {
    let mut overall_response = HashMap::new();

    let monitor_log = Arc::new(Mutex::new(Vec::new()));
    let overall_start_time = Instant::now();

    // Pass the Arc directly to the function
    start_memory_monitoring(Duration::from_millis(200), Arc::clone(&monitor_log), overall_start_time);

    // let stop_flag = Arc::new(AtomicBool::new(false));
    // let monitor_handle = start_memory_monitoring_handler(Duration::from_millis(200), Arc::clone(&monitor_log), overall_start_time, Arc::clone(&stop_flag));

    let num_columns: i32 = match dataset.as_str() {
        "frappe" => 12,
        "adult" => 15,
        "cvd" => 13,
        "bank" => 18,
        "census" => 41 + 2,
        "credit" => 23 + 2,
        "diabetes" => 48 + 2,
        "hcdr" => 69 + 2,
        "avazu" => 22 + 2,
        _ => return Err(format!("Unknown dataset: {}", dataset)),
    };

    // Step 1: load model and columns etc
    let mut task_map = HashMap::new();
    task_map.insert("where_cond", condition.clone());
    task_map.insert("config_file", config_file.clone());
    task_map.insert("col_cardinalities_file", col_cardinalities_file.clone());
    task_map.insert("model_path", model_path.clone());
    let task_json = json!(task_map).to_string();

    // Allocate shared memory once, here is not primary key and id
    let shmem_size = 4 * batch_size as usize * (num_columns - 2) as usize;
    let shmem_name = "my_shared_memory";
    let my_shmem = ShmemConf::new()
        .size(shmem_size)
        .os_id(shmem_name)
        .create()
        .map_err(|e| e.to_string())?;
    let shmem_ptr = my_shmem.as_ptr() as *mut i32;

    // run_python_function(
    //     &PY_MODULE_INFERENCE,
    //     &task_json,
    //     "init_log",
    // );
    // Here it cache a state once
    run_python_function(
        &PY_MODULE_INFERENCE,
        &task_json,
        "model_inference_load_model",
    );

    // log_memory_usage(&mut memory_log, overall_start_time, "load model done", pid);

    // Execute workloads
    let mut nquery = 0;
    let mut response = HashMap::new();
    while nquery < 300 {
        pgrx::log!("{}", "started");

        let model_init_time = Instant::now().duration_since(overall_start_time).as_secs_f64();
        response.insert("model_init_time", model_init_time.clone());

        // Step 1: query data
        let start_time = Instant::now();
        Spi::connect(|client| {
            let query = format!(
                "SELECT * FROM {}_int_train {} LIMIT {}",
                dataset, sql, batch_size
            );
            let mut cursor = client.open_cursor(&query, None);
            let table = cursor.fetch(batch_size as c_long)
                .map_err(|e| e.to_string())?;

            let end_time = Instant::now();
            let data_query_time_spi = end_time.duration_since(start_time).as_secs_f64();
            response.insert("data_query_time_spi", data_query_time_spi);

            let start_time_3 = Instant::now();
            let mut idx = 0;
            for row in table.into_iter() {
                for i in 3..=num_columns as usize {
                    if let Ok(Some(val)) = row.get::<i32>(i) {
                        unsafe {
                            std::ptr::write(shmem_ptr.add(idx), val);
                            idx += 1;
                        }
                    }
                }
            }
            let end_time_min3 = Instant::now();
            let data_query_time_min3 = end_time_min3.duration_since(start_time_3).as_secs_f64();
            response.insert("data_type_convert_time", data_query_time_min3.clone());

            Ok::<(), String>(()) // Specify the type explicitly
        })?;
        let data_query_time = Instant::now().duration_since(start_time).as_secs_f64();
        response.insert("data_query_time", data_query_time.clone());

        let mem_allocate_time = Instant::now().duration_since(start_time).as_secs_f64();
        response.insert("mem_allocate_time", mem_allocate_time.clone());

        // Step 4: model evaluate in Python
        let start_time = Instant::now();
        let mut eva_task_map = HashMap::new();
        eva_task_map.insert("config_file", config_file.clone());
        eva_task_map.insert("spi_seconds", data_query_time.to_string());
        eva_task_map.insert("rows", batch_size.to_string());
        let eva_task_json = json!(eva_task_map).to_string();

        run_python_function(
            &PY_MODULE_INFERENCE,
            &eva_task_json,
            "model_inference_compute_shared_memory_write_once_int",
        );

        // Step 4: simulate model evaluate in Python by sleeping
        // sleep(Duration::from_millis(10));

        let python_compute_time = Instant::now().duration_since(start_time).as_secs_f64();
        response.insert("python_compute_time", python_compute_time.clone());

        let overall_end_time = Instant::now();
        let overall_elapsed_time = overall_end_time.duration_since(overall_start_time).as_secs_f64();
        let diff_time = model_init_time + data_query_time + python_compute_time - overall_elapsed_time;
        response.insert("diff_time", diff_time.clone());

        nquery += 1;
        response.clear(); // Clear the response hash map/**/
        // log_memory_usage(&mut memory_log, overall_start_time, &format!("batch {} done", nquery), pid);
    }

    // Log memory usage after processing each batch
    // log_memory_usage(&mut memory_log, overall_start_time, "all batch done", pid);

    // Signal the monitoring thread to stop
    // stop_flag.store(true, Ordering::SeqCst);
    // monitor_handle.join().expect("Monitoring thread panicked");


    let overall_time_usage = Instant::now().duration_since(overall_start_time).as_secs_f64();
    overall_response.insert("overall_time_usage".to_string(), overall_time_usage.to_string());

    let monitor_log_rep = monitor_log.lock().unwrap();
    overall_response.insert("memory_log".to_string(), serde_json::to_string(&json!(*monitor_log_rep)).unwrap());

    let overall_response_json = serde_json::to_string(&json!(overall_response)).map_err(|e| e.to_string())?;

    run_python_function(
        &PY_MODULE_INFERENCE,
        &overall_response_json,
        "records_results",
    );

    Ok(())
}


pub fn run_inference_wo_cache_workloads(
    dataset: &String,
    condition: &String,
    config_file: &String,
    col_cardinalities_file: &String,
    model_path: &String,
    sql: &String,
    batch_size: i32,
) -> Result<(), String> {
    let mut overall_response = HashMap::new();

    let monitor_log = Arc::new(Mutex::new(Vec::new()));
    let overall_start_time = Instant::now();

    // Pass the Arc directly to the function
    // start_memory_monitoring(Duration::from_secs(1), Arc::clone(&monitor_log), overall_start_time);

    start_memory_monitoring(Duration::from_millis(200), Arc::clone(&monitor_log), overall_start_time);

    let num_columns: i32 = match dataset.as_str() {
        "frappe" => 12,
        "adult" => 15,
        "cvd" => 13,
        "bank" => 18,
        "census" => 41 + 2,
        "credit" => 23 + 2,
        "diabetes" => 48 + 2,
        "hcdr" => 69 + 2,
        "avazu" => 22 + 2,
        _ => return Err(format!("Unknown dataset: {}", dataset)),
    };

    // Allocate shared memory once, here is not primary key and id
    let shmem_size = 4 * batch_size as usize * (num_columns - 2) as usize;
    let shmem_name = "my_shared_memory";
    let my_shmem = ShmemConf::new()
        .size(shmem_size)
        .os_id(shmem_name)
        .create()
        .map_err(|e| e.to_string())?;
    let shmem_ptr = my_shmem.as_ptr() as *mut i32;

    // log_memory_usage(&mut memory_log, overall_start_time, "load model done", pid);

    // Execute workloads
    let mut nquery = 0;
    let mut response = HashMap::new();
    while nquery < 300 {
        pgrx::log!("{}", "started");

        let model_init_time = Instant::now().duration_since(overall_start_time).as_secs_f64();
        response.insert("model_init_time", model_init_time.clone());

        // Step 1: query data
        let start_time = Instant::now();
        Spi::connect(|client| {
            let query = format!(
                "SELECT * FROM {}_int_train {} LIMIT {}",
                dataset, sql, batch_size
            );
            let mut cursor = client.open_cursor(&query, None);
            let table = cursor.fetch(batch_size as c_long)
                .map_err(|e| e.to_string())?;

            let end_time = Instant::now();
            let data_query_time_spi = end_time.duration_since(start_time).as_secs_f64();
            response.insert("data_query_time_spi", data_query_time_spi);

            let start_time_3 = Instant::now();
            let mut idx = 0;
            for row in table.into_iter() {
                for i in 3..=num_columns as usize {
                    if let Ok(Some(val)) = row.get::<i32>(i) {
                        unsafe {
                            std::ptr::write(shmem_ptr.add(idx), val);
                            idx += 1;
                        }
                    }
                }
            }
            let end_time_min3 = Instant::now();
            let data_query_time_min3 = end_time_min3.duration_since(start_time_3).as_secs_f64();
            response.insert("data_type_convert_time", data_query_time_min3.clone());

            Ok::<(), String>(()) // Specify the type explicitly
        })?;
        let data_query_time = Instant::now().duration_since(start_time).as_secs_f64();
        response.insert("data_query_time", data_query_time.clone());

        let mem_allocate_time = Instant::now().duration_since(start_time).as_secs_f64();
        response.insert("mem_allocate_time", mem_allocate_time.clone());

        // Step 4: model evaluate in Python
        let start_time = Instant::now();
        let mut eva_task_map = HashMap::new();
        eva_task_map.insert("config_file", config_file.clone());
        eva_task_map.insert("spi_seconds", data_query_time.to_string());
        eva_task_map.insert("rows", batch_size.to_string());

        eva_task_map.insert("where_cond", condition.clone());
        eva_task_map.insert("col_cardinalities_file", col_cardinalities_file.clone());
        eva_task_map.insert("model_path", model_path.clone());
        let eva_task_json = json!(eva_task_map).to_string();

        run_python_function(
            &PY_MODULE_INFERENCE,
            &eva_task_json,
            "load_model_inference",
        );

        // Step 4: simulate model evaluate in Python by sleeping
        // sleep(Duration::from_millis(10));

        let python_compute_time = Instant::now().duration_since(start_time).as_secs_f64();
        response.insert("python_compute_time", python_compute_time.clone());

        let overall_end_time = Instant::now();
        let overall_elapsed_time = overall_end_time.duration_since(overall_start_time).as_secs_f64();
        let diff_time = model_init_time + data_query_time + python_compute_time - overall_elapsed_time;
        response.insert("diff_time", diff_time.clone());

        nquery += 1;
        response.clear(); // Clear the response hash map/**/
        // log_memory_usage(&mut memory_log, overall_start_time, &format!("batch {} done", nquery), pid);
    }

    // Log memory usage after processing each batch
    // log_memory_usage(&mut memory_log, overall_start_time, "all batch done", pid);

    let _end_time = Instant::now();
    let overall_time_usage = _end_time.duration_since(overall_start_time).as_secs_f64();
    overall_response.insert("overall_time_usage".to_string(), overall_time_usage.to_string());

    let monitor_log_rep = monitor_log.lock().unwrap();
    overall_response.insert("memory_log".to_string(), serde_json::to_string(&json!(*monitor_log_rep)).unwrap());


    let overall_response_json = serde_json::to_string(&json!(overall_response)).map_err(|e| e.to_string())?;

    run_python_function(
        &PY_MODULE_INFERENCE,
        &overall_response_json,
        "records_results",
    );

    Ok(())
}


pub fn run_inference_wo_memoryshare_workloads(
    dataset: &String,
    condition: &String,
    config_file: &String,
    col_cardinalities_file: &String,
    model_path: &String,
    sql: &String,
    batch_size: i32,
) -> Result<(), String> {
    let mut overall_response = HashMap::new();

    let monitor_log = Arc::new(Mutex::new(Vec::new()));
    let overall_start_time = Instant::now();

    // Pass the Arc directly to the function
    start_memory_monitoring(Duration::from_millis(200), Arc::clone(&monitor_log), overall_start_time);


    let num_columns: i32 = match dataset.as_str() {
        "frappe" => 12,
        "adult" => 15,
        "cvd" => 13,
        "bank" => 18,
        "census" => 41 + 2,
        "credit" => 23 + 2,
        "diabetes" => 48 + 2,
        "hcdr" => 69 + 2,
        "avazu" => 22 + 2,
        _ => return Err(format!("Unknown dataset: {}", dataset)),
    };

    // Step 1: load model and columns etc
    let mut task_map = HashMap::new();
    task_map.insert("where_cond", condition.clone());
    task_map.insert("config_file", config_file.clone());
    task_map.insert("col_cardinalities_file", col_cardinalities_file.clone());
    task_map.insert("model_path", model_path.clone());
    let task_json = json!(task_map).to_string();

    run_python_function(
        &PY_MODULE_INFERENCE,
        &task_json,
        "model_inference_load_model",
    );

    // log_memory_usage(&mut memory_log, overall_start_time, "load model done", pid);

    // Execute workloads
    let mut nquery = 0;
    let mut response = HashMap::new();
    while nquery < 300 {

        // pgx::elog!(pgx::PgLogLevel::NOTICE, &format!("batch {} done", nquery));

        let model_init_time = Instant::now().duration_since(overall_start_time).as_secs_f64();
        response.insert("model_init_time", model_init_time.clone());

        // Step 1: query data
        let start_time = Instant::now();
        let mut all_rows = Vec::new();
        Spi::connect(|client| {
            let query = format!(
                "SELECT * FROM {}_int_train {} LIMIT {}",
                dataset, sql, batch_size
            );
            let mut cursor = client.open_cursor(&query, None);
            let table = cursor.fetch(batch_size as c_long)
                .map_err(|e| e.to_string())?;

            let end_time = Instant::now();
            let data_query_time_spi = end_time.duration_since(start_time).as_secs_f64();
            response.insert("data_query_time_spi", data_query_time_spi);

            let start_time_3 = Instant::now();
            for row in table.into_iter() {
                for i in 3..=num_columns as usize {
                    if let Ok(Some(val)) = row.get::<i32>(i) {
                        all_rows.push(val);
                    }
                }
            }
            let end_time_min3 = Instant::now();
            let data_query_time_min3 = end_time_min3.duration_since(start_time_3).as_secs_f64();
            response.insert("data_type_convert_time", data_query_time_min3.clone());

            Ok::<(), String>(()) // Specify the type explicitly
        })?;
        let data_query_time = Instant::now().duration_since(start_time).as_secs_f64();
        response.insert("data_query_time", data_query_time.clone());

        let mem_allocate_time = Instant::now().duration_since(start_time).as_secs_f64();
        response.insert("mem_allocate_time", mem_allocate_time.clone());

        let serialized_row = serde_json::to_string(&all_rows).unwrap();
        // Step 4: model evaluate in Python
        let start_time = Instant::now();
        let mut eva_task_map = HashMap::new();
        eva_task_map.insert("config_file", config_file.clone());
        eva_task_map.insert("mini_batch", serialized_row);
        eva_task_map.insert("spi_seconds", data_query_time.to_string());
        let eva_task_json = json!(eva_task_map).to_string();

        run_python_function(
            &PY_MODULE_INFERENCE,
            &eva_task_json,
            "model_inference_compute",
        );

        // Step 4: simulate model evaluate in Python by sleeping
        // sleep(Duration::from_millis(10));

        let python_compute_time = Instant::now().duration_since(start_time).as_secs_f64();
        response.insert("python_compute_time", python_compute_time.clone());

        let overall_end_time = Instant::now();
        let overall_elapsed_time = overall_end_time.duration_since(overall_start_time).as_secs_f64();
        let diff_time = model_init_time + data_query_time + python_compute_time - overall_elapsed_time;
        response.insert("diff_time", diff_time.clone());

        nquery += 1;
        response.clear(); // Clear the response hash map/**/
        // log_memory_usage(&mut memory_log, overall_start_time, &format!("batch {} done", nquery), pid);
    }

    // Log memory usage after processing each batch
    // log_memory_usage(&mut memory_log, overall_start_time, "all batch done", pid);

    let _end_time = Instant::now();
    let overall_time_usage = _end_time.duration_since(overall_start_time).as_secs_f64();
    overall_response.insert("overall_time_usage".to_string(), overall_time_usage.to_string());

    let monitor_log_rep = monitor_log.lock().unwrap();
    overall_response.insert("memory_log".to_string(), serde_json::to_string(&json!(*monitor_log_rep)).unwrap());

    let overall_response_json = serde_json::to_string(&json!(overall_response)).map_err(|e| e.to_string())?;

    run_python_function(
        &PY_MODULE_INFERENCE,
        &overall_response_json,
        "records_results",
    );

    Ok(())
}

pub fn invesgate_memory_usage(
    dataset: &String,
    condition: &String,
    config_file: &String,
    col_cardinalities_file: &String,
    model_path: &String,
    sql: &String,
    batch_size: i32,
) -> Result<(), String> {
    let mut overall_response = HashMap::new();
    let overall_start_time = Instant::now();

    // Pass the Arc directly to the function
    let monitor_log = Arc::new(Mutex::new(Vec::new()));
    // start_memory_monitoring(Duration::from_millis(200), Arc::clone(&monitor_log), overall_start_time);

    let stop_flag = Arc::new(AtomicBool::new(false));
    let monitor_handle = start_memory_monitoring_handler(Duration::from_millis(200), Arc::clone(&monitor_log), overall_start_time, Arc::clone(&stop_flag));


    let num_columns: i32 = match dataset.as_str() {
        "frappe" => 12,
        "adult" => 15,
        "cvd" => 13,
        "bank" => 18,
        "census" => 41 + 2,
        "credit" => 23 + 2,
        "diabetes" => 48 + 2,
        "hcdr" => 69 + 2,
        "avazu" => 22 + 2,
        _ => return Err(format!("Unknown dataset: {}", dataset)),
    };

    // Execute workloads
    sleep(Duration::from_millis(210));

    // Signal the monitoring thread to stop
    stop_flag.store(true, Ordering::SeqCst);
    // Wait for the monitoring thread to finish
    monitor_handle.join().expect("Monitoring thread panicked");

    // let mut monitor_log = Vec::new();
    // record_memory_usage(&mut monitor_log, overall_start_time);
    // overall_response.insert("memory_log".to_string(), serde_json::to_string(&json!(monitor_log)).unwrap());

    let overall_time_usage = Instant::now().duration_since(overall_start_time).as_secs_f64();
    overall_response.insert("overall_time_usage".to_string(), overall_time_usage.to_string());

    let monitor_log_rep = monitor_log.lock().unwrap();
    overall_response.insert("memory_log".to_string(), serde_json::to_string(&json!(*monitor_log_rep)).unwrap());
    let overall_response_json = serde_json::to_string(&json!(overall_response)).map_err(|e| e.to_string())?;

    let mut file = OpenOptions::new()
        .append(true)  // Open in append mode
        .create(true)  // Create the file if it doesn't exist
        .open("/home/postgres/.pgrx/data-14/trails_log_folder/rust_res.json")
        .map_err(|e| e.to_string())?;

    // Write the JSON string to the file, followed by a newline
    writeln!(file, "{}", overall_response_json).map_err(|e| e.to_string())?;
    Ok(())
}


pub fn invesgate_memory_usage_record_only(
    dataset: &String,
    condition: &String,
    config_file: &String,
    col_cardinalities_file: &String,
    model_path: &String,
    sql: &String,
    batch_size: i32,
) -> Result<(), String> {
    let mut overall_response = HashMap::new();
    let overall_start_time = Instant::now();


    let mut monitor_log = Vec::new();

    let num_columns: i32 = match dataset.as_str() {
        "frappe" => 12,
        "adult" => 15,
        "cvd" => 13,
        "bank" => 18,
        "census" => 41 + 2,
        "credit" => 23 + 2,
        "diabetes" => 48 + 2,
        "hcdr" => 69 + 2,
        "avazu" => 22 + 2,
        _ => return Err(format!("Unknown dataset: {}", dataset)),
    };

    // Step 1: load model and columns etc
    let mut task_map = HashMap::new();
    task_map.insert("where_cond", condition.clone());
    task_map.insert("config_file", config_file.clone());
    task_map.insert("col_cardinalities_file", col_cardinalities_file.clone());
    task_map.insert("model_path", model_path.clone());
    let task_json = json!(task_map).to_string();

    record_memory_usage(&mut monitor_log, overall_start_time);

    // Allocate shared memory once, here is not primary key and id
    let shmem_size = 4 * batch_size as usize * (num_columns - 2) as usize;
    let shmem_name = "my_shared_memory";
    let my_shmem = ShmemConf::new()
        .size(shmem_size)
        .os_id(shmem_name)
        .create()
        .map_err(|e| e.to_string())?;
    let shmem_ptr = my_shmem.as_ptr() as *mut i32;

    // run_python_function(
    //     &PY_MODULE_INFERENCE,
    //     &task_json,
    //     "init_log",
    // );
    // Here it cache a state once

    record_memory_usage(&mut monitor_log, overall_start_time);

    run_python_function(
        &PY_MODULE_INFERENCE,
        &task_json,
        "model_inference_load_model",
    );

    record_memory_usage(&mut monitor_log, overall_start_time);
    // log_memory_usage(&mut memory_log, overall_start_time, "load model done", pid);

    // Execute workloads
    let mut nquery = 0;
    let mut response = HashMap::new();
    while nquery < 300 {
        pgrx::log!("{}", "started");

        let model_init_time = Instant::now().duration_since(overall_start_time).as_secs_f64();
        response.insert("model_init_time", model_init_time.clone());

        // Step 1: query data
        let start_time = Instant::now();
        Spi::connect(|client| {
            let query = format!(
                "SELECT * FROM {}_int_train {} LIMIT {}",
                dataset, sql, batch_size
            );
            let mut cursor = client.open_cursor(&query, None);
            let table = cursor.fetch(batch_size as c_long)
                .map_err(|e| e.to_string())?;

            let end_time = Instant::now();
            let data_query_time_spi = end_time.duration_since(start_time).as_secs_f64();
            response.insert("data_query_time_spi", data_query_time_spi);

            let start_time_3 = Instant::now();
            let mut idx = 0;
            for row in table.into_iter() {
                for i in 3..=num_columns as usize {
                    if let Ok(Some(val)) = row.get::<i32>(i) {
                        unsafe {
                            std::ptr::write(shmem_ptr.add(idx), val);
                            idx += 1;
                        }
                    }
                }
            }
            let end_time_min3 = Instant::now();
            let data_query_time_min3 = end_time_min3.duration_since(start_time_3).as_secs_f64();
            response.insert("data_type_convert_time", data_query_time_min3.clone());

            Ok::<(), String>(()) // Specify the type explicitly
        })?;
        let data_query_time = Instant::now().duration_since(start_time).as_secs_f64();
        response.insert("data_query_time", data_query_time.clone());

        let mem_allocate_time = Instant::now().duration_since(start_time).as_secs_f64();
        response.insert("mem_allocate_time", mem_allocate_time.clone());

        // Step 4: model evaluate in Python
        let start_time = Instant::now();
        let mut eva_task_map = HashMap::new();
        eva_task_map.insert("config_file", config_file.clone());
        eva_task_map.insert("spi_seconds", data_query_time.to_string());
        eva_task_map.insert("rows", batch_size.to_string());
        let eva_task_json = json!(eva_task_map).to_string();

        run_python_function(
            &PY_MODULE_INFERENCE,
            &eva_task_json,
            "model_inference_compute_shared_memory_write_once_int",
        );

        // Step 4: simulate model evaluate in Python by sleeping
        // sleep(Duration::from_millis(10));

        let python_compute_time = Instant::now().duration_since(start_time).as_secs_f64();
        response.insert("python_compute_time", python_compute_time.clone());

        let overall_end_time = Instant::now();
        let overall_elapsed_time = overall_end_time.duration_since(overall_start_time).as_secs_f64();
        let diff_time = model_init_time + data_query_time + python_compute_time - overall_elapsed_time;
        response.insert("diff_time", diff_time.clone());

        nquery += 1;
        response.clear(); // Clear the response hash map/**/

        record_memory_usage(&mut monitor_log, overall_start_time);
    }

    let overall_time_usage = Instant::now().duration_since(overall_start_time).as_secs_f64();
    overall_response.insert("overall_time_usage".to_string(), overall_time_usage.to_string());

    // let monitor_log_rep = monitor_log.lock().unwrap();
    overall_response.insert("memory_log".to_string(), serde_json::to_string(&json!(monitor_log)).unwrap());

    let overall_response_json = serde_json::to_string(&json!(overall_response)).map_err(|e| e.to_string())?;

    record_memory_usage(&mut monitor_log, overall_start_time);
    run_python_function(
        &PY_MODULE_INFERENCE,
        &overall_response_json,
        "records_results",
    );

    Ok(())
}

