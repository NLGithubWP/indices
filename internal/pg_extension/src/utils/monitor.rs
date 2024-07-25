use std::thread;
use std::time::{Duration, Instant};
use sysinfo::{System, SystemExt, ProcessExt};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};

pub fn start_memory_monitoring(interval: Duration,
                               memory_log: Arc<Mutex<Vec<(i32, f64, u64)>>>,
                               start_time: Instant) {
    let pid = std::process::id() as i32;
    let mut system = System::new_all();

    thread::spawn(move || {
        loop {
            system.refresh_all();
            if let Some(process) = system.process(pid) {
                let memory_usage = process.memory();
                let timestamp = start_time.elapsed().as_secs_f64();
                let mut log = memory_log.lock().unwrap();
                log.push((pid, timestamp, memory_usage));
            }
            thread::sleep(interval);
        }
    });
}

pub fn record_memory_usage(monitor_log: &mut Vec<(i32, f64, u64)>, start_time: Instant) {
    let pid = std::process::id() as i32;
    let mut system = System::new_all();
    system.refresh_all();
    if let Some(process) = system.process(pid) {
        let memory_usage = process.memory();
        let timestamp = start_time.elapsed().as_secs_f64();
        monitor_log.push((pid, timestamp, memory_usage));
    }
}


pub fn log_memory_usage(memory_log: &mut Vec<(String, f64, u64)>,
                        start_time: Instant, label: &str, pid: i32) {
    let mut system = System::new_all();
    system.refresh_memory();
    if let Some(process) = system.process(pid) {
        let memory_usage = process.memory();
        let timestamp = start_time.elapsed().as_secs_f64();
        memory_log.push((label.to_string(), timestamp, memory_usage));
    }
}


pub fn start_memory_monitoring_handler(interval: Duration,
                                       memory_log: Arc<Mutex<Vec<(i32, f64, u64)>>>,
                                       start_time: Instant,
                                       stop_flag: Arc<AtomicBool>) -> thread::JoinHandle<()> {
    let pid = std::process::id() as i32;
    let mut system = System::new_all();

    thread::spawn(move || {
        while !stop_flag.load(Ordering::SeqCst) {
            system.refresh_all();
            if let Some(process) = system.process(pid) {
                let memory_usage = process.memory();
                let timestamp = start_time.elapsed().as_secs_f64();
                let mut log = memory_log.lock().unwrap();
                log.push((pid, timestamp, memory_usage));
            }
            thread::sleep(interval);
        }
    })
}