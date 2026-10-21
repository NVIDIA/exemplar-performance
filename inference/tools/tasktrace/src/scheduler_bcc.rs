// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native runtime-loaded BCC scheduler tracing.
//!
//! This module deliberately has no link-time dependency on libbcc. The native
//! collector resolves its required libbcc C ABI with `dlopen(3)`/`dlsym(3)`
//! only when preflight or tracing starts.

#![allow(dead_code)]

use crate::scheduler_perf::{summarize_scheduler_events, write_scheduler_events, SchedulerEvent};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::error::Error;
use std::ffi::{CStr, CString};
use std::fmt;
use std::fs;
use std::io;
use std::mem;
use std::os::fd::RawFd;
use std::os::raw::{c_char, c_int, c_uint, c_ulonglong, c_void};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::ptr;

pub(crate) const BPF_PROGRAM: &str = r#"
#include <uapi/linux/ptrace.h>

struct event_t {
    u64 timestamp_ns;
    u32 cpu;
    u32 tid;
    u8 entering;
};

struct cpu_state_t {
    u32 tid;
    u8 active;
};

BPF_ARRAY(target_cgroup_id, u64, 1);
BPF_ARRAY(target_cgroup_level, u32, 1);
BPF_PERCPU_ARRAY(cpu_state, struct cpu_state_t, 1);
BPF_PERF_OUTPUT(events);

static __always_inline void emit_transition(
    void *ctx,
    u64 timestamp_ns,
    u32 tid,
    u8 entering
)
{
    struct event_t event = {};
    event.timestamp_ns = timestamp_ns;
    event.cpu = bpf_get_smp_processor_id();
    event.tid = tid;
    event.entering = entering;
    events.perf_submit(ctx, &event, sizeof(event));
}

int trace_finish_task_switch(struct pt_regs *ctx)
{
    u64 timestamp_ns = bpf_ktime_get_ns();
    u32 key = 0;
    u64 *target = target_cgroup_id.lookup(&key);
    u32 *target_level = target_cgroup_level.lookup(&key);
    struct cpu_state_t *state = cpu_state.lookup(&key);
    if (!target || !target_level || !state)
        return 0;

    u64 current_cgroup_id = bpf_get_current_ancestor_cgroup_id(*target_level);
    u32 current_tid = (u32)bpf_get_current_pid_tgid();
    u8 active = current_cgroup_id == *target;

    if (state->active && !active) {
        emit_transition(ctx, timestamp_ns, state->tid, 0);
    } else if (!state->active && active) {
        emit_transition(ctx, timestamp_ns, current_tid, 1);
    }

    state->active = active;
    state->tid = active ? current_tid : 0;
    return 0;
}
"#;

const DEFAULT_PERF_BUFFER_PAGES: u64 = 8;
pub(crate) const EVENT_BOUNDARY_TOLERANCE_SEC: f64 = 0.000_002;
const REQUIRED_SYMBOLS: &[&str] = &[
    "bpf_module_create_c_from_string",
    "bpf_module_destroy",
    "bpf_module_license",
    "bpf_module_kern_version",
    "bpf_function_start",
    "bpf_function_size",
    "bpf_table_fd",
    "bpf_update_elem",
    "bpf_delete_elem",
    "bcc_func_load",
    "bpf_attach_kprobe",
    "bpf_detach_kprobe",
    "bpf_close_perf_event_fd",
    "bpf_open_perf_buffer_opts",
    "perf_reader_fd",
    "perf_reader_poll",
    "perf_reader_free",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BccSchedulerError(String);

impl BccSchedulerError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for BccSchedulerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for BccSchedulerError {}

impl From<io::Error> for BccSchedulerError {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}

type RawCallback = unsafe extern "C" fn(*mut c_void, *mut c_void, c_int);
type LostCallback = unsafe extern "C" fn(*mut c_void, c_ulonglong);
type ModuleCreate = unsafe extern "C" fn(
    *const c_char,
    c_uint,
    *const *const c_char,
    c_int,
    bool,
    *const c_char,
) -> *mut c_void;
type ModuleDestroy = unsafe extern "C" fn(*mut c_void);
type ModuleLicense = unsafe extern "C" fn(*mut c_void) -> *const c_char;
type ModuleKernelVersion = unsafe extern "C" fn(*mut c_void) -> c_uint;
type FunctionStart = unsafe extern "C" fn(*mut c_void, *const c_char) -> *mut c_void;
type FunctionSize = unsafe extern "C" fn(*mut c_void, *const c_char) -> usize;
type TableFd = unsafe extern "C" fn(*mut c_void, *const c_char) -> c_int;
type UpdateElement =
    unsafe extern "C" fn(c_int, *const c_void, *const c_void, c_ulonglong) -> c_int;
type DeleteElement = unsafe extern "C" fn(c_int, *const c_void) -> c_int;
type FunctionLoad = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    *const c_char,
    *mut c_void,
    usize,
    *const c_char,
    c_uint,
    c_int,
    *mut c_char,
    c_uint,
    *const c_char,
    c_uint,
) -> c_int;
type AttachKprobe =
    unsafe extern "C" fn(c_int, c_int, *const c_char, *const c_char, c_ulonglong, c_int) -> c_int;
type DetachKprobe = unsafe extern "C" fn(*const c_char) -> c_int;
type ClosePerfEventFd = unsafe extern "C" fn(c_int) -> c_int;
type OpenPerfBuffer = unsafe extern "C" fn(
    RawCallback,
    LostCallback,
    *mut c_void,
    c_int,
    *const PerfBufferOptions,
) -> *mut c_void;
type PerfReaderFd = unsafe extern "C" fn(*mut c_void) -> c_int;
type PerfReaderPoll = unsafe extern "C" fn(c_int, *mut *mut c_void, c_int) -> c_int;
type PerfReaderFree = unsafe extern "C" fn(*mut c_void);

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct PerfBufferOptions {
    pid: c_int,
    cpu: c_int,
    wakeup_events: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawSchedulerEvent {
    timestamp_ns: u64,
    cpu: u32,
    tid: u32,
    entering: u8,
}

#[derive(Default)]
struct CallbackState {
    events: Vec<SchedulerEvent>,
    lost_events: u64,
    error: Option<String>,
}

unsafe extern "C" fn record_event_callback(context: *mut c_void, data: *mut c_void, size: c_int) {
    if context.is_null() {
        return;
    }
    let state = &mut *context.cast::<CallbackState>();
    if data.is_null() || size < mem::size_of::<RawSchedulerEvent>() as c_int {
        if state.error.is_none() {
            state.error = Some(format!(
                "BCC scheduler perf event has invalid size {size}; expected at least {}",
                mem::size_of::<RawSchedulerEvent>()
            ));
        }
        return;
    }
    let raw = ptr::read_unaligned(data.cast::<RawSchedulerEvent>());
    state.events.push(SchedulerEvent {
        timestamp_ns: raw.timestamp_ns,
        cpu: raw.cpu,
        tid: i64::from(raw.tid),
        entering: raw.entering != 0,
    });
}

unsafe extern "C" fn record_lost_callback(context: *mut c_void, lost: c_ulonglong) {
    if context.is_null() {
        return;
    }
    let state = &mut *context.cast::<CallbackState>();
    match state.lost_events.checked_add(lost) {
        Some(total) => state.lost_events = total,
        None if state.error.is_none() => {
            state.error = Some("BCC scheduler lost-event count overflow".to_owned())
        }
        None => {}
    }
}

struct DynamicLibrary {
    handle: *mut c_void,
    loaded_name: String,
}

impl DynamicLibrary {
    fn load(explicit: Option<&Path>) -> Result<Self, BccSchedulerError> {
        let candidates = explicit.map_or_else(
            || vec![PathBuf::from("libbcc.so.0"), PathBuf::from("libbcc.so")],
            |path| vec![path.to_path_buf()],
        );
        let mut failures = Vec::new();
        for candidate in candidates {
            let name = CString::new(candidate.as_os_str().as_bytes())
                .map_err(|_| BccSchedulerError::new("libbcc path contains an embedded NUL byte"))?;
            unsafe {
                libc::dlerror();
            }
            let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            if !handle.is_null() {
                return Ok(Self {
                    handle,
                    loaded_name: candidate.to_string_lossy().into_owned(),
                });
            }
            failures.push(format!(
                "{}: {}",
                candidate.display(),
                dynamic_error("unknown dynamic-loader error")
            ));
        }
        Err(BccSchedulerError::new(format!(
            "unable to load libbcc at runtime: {}",
            failures.join("; ")
        )))
    }

    fn symbol(&self, name: &str) -> Result<*mut c_void, BccSchedulerError> {
        let name_c = CString::new(name).expect("static symbol has no NUL");
        unsafe {
            libc::dlerror();
        }
        let symbol = unsafe { libc::dlsym(self.handle, name_c.as_ptr()) };
        let error = unsafe { libc::dlerror() };
        if symbol.is_null() || !error.is_null() {
            let detail = if error.is_null() {
                "symbol address is null".to_owned()
            } else {
                unsafe { CStr::from_ptr(error) }
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(BccSchedulerError::new(format!(
                "libbcc runtime {} is missing required symbol {name}: {detail}",
                self.loaded_name
            )));
        }
        Ok(symbol)
    }
}

impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                libc::dlclose(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}

fn dynamic_error(fallback: &str) -> String {
    let error = unsafe { libc::dlerror() };
    if error.is_null() {
        fallback.to_owned()
    } else {
        unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned()
    }
}

struct BccApi {
    library: DynamicLibrary,
    module_create: ModuleCreate,
    module_destroy: ModuleDestroy,
    module_license: ModuleLicense,
    module_kernel_version: ModuleKernelVersion,
    function_start: FunctionStart,
    function_size: FunctionSize,
    table_fd: TableFd,
    update_element: UpdateElement,
    delete_element: DeleteElement,
    function_load: FunctionLoad,
    attach_kprobe: AttachKprobe,
    detach_kprobe: DetachKprobe,
    close_perf_event_fd: ClosePerfEventFd,
    open_perf_buffer: OpenPerfBuffer,
    perf_reader_fd: PerfReaderFd,
    perf_reader_poll: PerfReaderPoll,
    perf_reader_free: PerfReaderFree,
}

impl BccApi {
    fn load(path: Option<&Path>) -> Result<Self, BccSchedulerError> {
        let library = DynamicLibrary::load(path)?;
        macro_rules! function {
            ($name:literal, $kind:ty) => {{
                let address = library.symbol($name)?;
                unsafe { mem::transmute::<*mut c_void, $kind>(address) }
            }};
        }
        Ok(Self {
            module_create: function!("bpf_module_create_c_from_string", ModuleCreate),
            module_destroy: function!("bpf_module_destroy", ModuleDestroy),
            module_license: function!("bpf_module_license", ModuleLicense),
            module_kernel_version: function!("bpf_module_kern_version", ModuleKernelVersion),
            function_start: function!("bpf_function_start", FunctionStart),
            function_size: function!("bpf_function_size", FunctionSize),
            table_fd: function!("bpf_table_fd", TableFd),
            update_element: function!("bpf_update_elem", UpdateElement),
            delete_element: function!("bpf_delete_elem", DeleteElement),
            function_load: function!("bcc_func_load", FunctionLoad),
            attach_kprobe: function!("bpf_attach_kprobe", AttachKprobe),
            detach_kprobe: function!("bpf_detach_kprobe", DetachKprobe),
            close_perf_event_fd: function!("bpf_close_perf_event_fd", ClosePerfEventFd),
            open_perf_buffer: function!("bpf_open_perf_buffer_opts", OpenPerfBuffer),
            perf_reader_fd: function!("perf_reader_fd", PerfReaderFd),
            perf_reader_poll: function!("perf_reader_poll", PerfReaderPoll),
            perf_reader_free: function!("perf_reader_free", PerfReaderFree),
            library,
        })
    }
}

#[derive(Clone, Debug)]
struct BccHostPaths {
    cgroup_root: PathBuf,
    online_cpus: PathBuf,
    kallsyms: PathBuf,
    available_filter_functions: PathBuf,
    kprobe_blacklist: PathBuf,
}

impl BccHostPaths {
    fn system() -> Self {
        let debug_tracefs = PathBuf::from("/sys/kernel/debug/tracing");
        let tracefs = if debug_tracefs.exists() {
            debug_tracefs
        } else {
            PathBuf::from("/sys/kernel/tracing")
        };
        Self {
            cgroup_root: PathBuf::from("/sys/fs/cgroup"),
            online_cpus: PathBuf::from("/sys/devices/system/cpu/online"),
            kallsyms: PathBuf::from("/proc/kallsyms"),
            available_filter_functions: tracefs.join("available_filter_functions"),
            kprobe_blacklist: PathBuf::from("/sys/kernel/debug/kprobes/blacklist"),
        }
    }
}

struct PerfReader {
    cpu: u32,
    reader: *mut c_void,
}

struct ActiveSession {
    module: *mut c_void,
    program_fd: RawFd,
    attach_fd: RawFd,
    event_name: Option<CString>,
    events_map_fd: RawFd,
    readers: Vec<PerfReader>,
}

impl ActiveSession {
    fn new(module: *mut c_void) -> Self {
        Self {
            module,
            program_fd: -1,
            attach_fd: -1,
            event_name: None,
            events_map_fd: -1,
            readers: Vec::new(),
        }
    }
}

pub(crate) struct BccSchedulerTracer {
    library_path: Option<PathBuf>,
    buffer_pages: Option<u64>,
    callback: Box<CallbackState>,
    api: Option<BccApi>,
    active: Option<ActiveSession>,
    ever_started: bool,
    pub(crate) root_tid: Option<u32>,
    pub(crate) target_cgroup_id: Option<u64>,
    pub(crate) target_cgroup_level: Option<u32>,
    pub(crate) kprobe_symbol: Option<String>,
}

impl BccSchedulerTracer {
    pub(crate) fn new(buffer_pages: Option<u64>) -> Result<Self, BccSchedulerError> {
        Self::with_library(None, buffer_pages)
    }

    pub(crate) fn with_library(
        library_path: Option<PathBuf>,
        buffer_pages: Option<u64>,
    ) -> Result<Self, BccSchedulerError> {
        validate_buffer_pages(buffer_pages)?;
        Ok(Self {
            library_path,
            buffer_pages,
            callback: Box::<CallbackState>::default(),
            api: None,
            active: None,
            ever_started: false,
            root_tid: None,
            target_cgroup_id: None,
            target_cgroup_level: None,
            kprobe_symbol: None,
        })
    }

    fn effective_buffer_pages(&self) -> u64 {
        self.buffer_pages.unwrap_or(DEFAULT_PERF_BUFFER_PAGES)
    }

    pub(crate) fn preflight(
        library_path: Option<&Path>,
        buffer_pages: Option<u64>,
    ) -> Result<Value, BccSchedulerError> {
        preflight_with_paths(library_path, buffer_pages, &BccHostPaths::system(), true)
    }

    pub(crate) fn inspect_library(library_path: Option<&Path>) -> Result<Value, BccSchedulerError> {
        let api = BccApi::load(library_path)?;
        Ok(json!({
            "schema_version": "scheduler-bcc-library-probe.v1",
            "ready": true,
            "runtime_loading": "dlopen-dlsym",
            "library": api.library.loaded_name,
            "required_symbols": REQUIRED_SYMBOLS,
        }))
    }

    pub(crate) fn start(
        &mut self,
        root_tid: u32,
        cgroup_path: &Path,
    ) -> Result<(), BccSchedulerError> {
        self.start_with_paths(root_tid, cgroup_path, &BccHostPaths::system(), true)
    }

    fn start_with_paths(
        &mut self,
        root_tid: u32,
        cgroup_path: &Path,
        paths: &BccHostPaths,
        require_root: bool,
    ) -> Result<(), BccSchedulerError> {
        if self.ever_started {
            return Err(BccSchedulerError::new(
                "BCC scheduler tracer cannot be started more than once",
            ));
        }
        self.ever_started = true;
        let result = self.start_inner(root_tid, cgroup_path, paths, require_root);
        if let Err(error) = result {
            let cleanup = self.close_checked();
            return Err(match cleanup {
                Ok(()) => error,
                Err(cleanup) => BccSchedulerError::new(format!(
                    "{error}; BCC scheduler cleanup also failed: {cleanup}"
                )),
            });
        }
        Ok(())
    }

    fn start_inner(
        &mut self,
        root_tid: u32,
        cgroup_path: &Path,
        paths: &BccHostPaths,
        require_root: bool,
    ) -> Result<(), BccSchedulerError> {
        require_privilege(require_root)?;
        validate_cgroup_v2(&paths.cgroup_root)?;
        let target_level = cgroup_level(cgroup_path, &paths.cgroup_root)?;
        let target_id = cgroup_path
            .metadata()
            .map_err(|error| {
                BccSchedulerError::new(format!(
                    "unable to inspect scheduler target cgroup {}: {error}",
                    cgroup_path.display()
                ))
            })?
            .ino();
        let cpus = online_cpus(&paths.online_cpus)?;
        let symbol = discover_kprobe_symbol(paths)?;
        let api = BccApi::load(self.library_path.as_deref())?;
        let module = create_module(&api)?;
        self.api = Some(api);
        self.active = Some(ActiveSession::new(module));

        let pages = c_int::try_from(self.effective_buffer_pages())
            .map_err(|_| BccSchedulerError::new("BCC perf-buffer page count exceeds c_int"))?;
        let api = self.api.as_ref().expect("BCC API installed");
        let active = self.active.as_mut().expect("BCC session installed");
        let target_map = table_fd(api, module, "target_cgroup_id")?;
        update_map(api, target_map, &0_u32, &target_id, "target_cgroup_id")?;
        let level_map = table_fd(api, module, "target_cgroup_level")?;
        update_map(api, level_map, &0_u32, &target_level, "target_cgroup_level")?;
        active.events_map_fd = table_fd(api, module, "events")?;

        let callback_context = (&mut *self.callback as *mut CallbackState).cast::<c_void>();
        for cpu in cpus {
            let cpu_int = c_int::try_from(cpu)
                .map_err(|_| BccSchedulerError::new("online logical CPU ID exceeds c_int"))?;
            let options = PerfBufferOptions {
                pid: -1,
                cpu: cpu_int,
                wakeup_events: 1,
            };
            let reader = unsafe {
                (api.open_perf_buffer)(
                    record_event_callback,
                    record_lost_callback,
                    callback_context,
                    pages,
                    &options,
                )
            };
            if reader.is_null() {
                return Err(BccSchedulerError::new(format!(
                    "unable to open BCC scheduler perf buffer for logical CPU {cpu}: {}",
                    io::Error::last_os_error()
                )));
            }
            active.readers.push(PerfReader { cpu, reader });
            let reader_fd = unsafe { (api.perf_reader_fd)(reader) };
            if reader_fd < 0 {
                return Err(BccSchedulerError::new(format!(
                    "BCC scheduler perf reader for logical CPU {cpu} has no valid fd"
                )));
            }
            update_map(api, active.events_map_fd, &cpu, &reader_fd, "events")?;
        }

        active.program_fd = load_program(api, module)?;
        let event_name = kprobe_event_name(&symbol)?;
        let symbol_c = CString::new(symbol.as_bytes()).expect("kernel symbol has no NUL");
        active.attach_fd = unsafe {
            (api.attach_kprobe)(
                active.program_fd,
                0,
                event_name.as_ptr(),
                symbol_c.as_ptr(),
                0,
                0,
            )
        };
        if active.attach_fd < 0 {
            return Err(BccSchedulerError::new(format!(
                "unable to attach scheduler BPF program to kprobe {symbol}: {}",
                io::Error::last_os_error()
            )));
        }
        active.event_name = Some(event_name);
        self.root_tid = Some(root_tid);
        self.target_cgroup_id = Some(target_id);
        self.target_cgroup_level = Some(target_level);
        self.kprobe_symbol = Some(symbol);
        Ok(())
    }

    pub(crate) fn poll(&mut self, timeout_ms: i32) -> Result<(), BccSchedulerError> {
        if timeout_ms < 0 {
            return Err(BccSchedulerError::new(
                "BCC scheduler poll timeout must be nonnegative",
            ));
        }
        let api = self
            .api
            .as_ref()
            .ok_or_else(|| BccSchedulerError::new("scheduler tracer has not been started"))?;
        let active = self
            .active
            .as_ref()
            .ok_or_else(|| BccSchedulerError::new("scheduler tracer has not been started"))?;
        let mut readers = active
            .readers
            .iter()
            .map(|reader| reader.reader)
            .collect::<Vec<_>>();
        let count = c_int::try_from(readers.len())
            .map_err(|_| BccSchedulerError::new("too many BCC perf readers"))?;
        let result = unsafe { (api.perf_reader_poll)(count, readers.as_mut_ptr(), timeout_ms) };
        if result < 0 {
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(BccSchedulerError::new(format!(
                    "BCC scheduler perf-buffer poll failed: {error}"
                )));
            }
        }
        if let Some(error) = &self.callback.error {
            return Err(BccSchedulerError::new(error.clone()));
        }
        Ok(())
    }

    pub(crate) fn timestamp_ns() -> Result<u64, BccSchedulerError> {
        let mut value = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut value) } != 0 {
            return Err(io::Error::last_os_error().into());
        }
        let seconds = u64::try_from(value.tv_sec)
            .map_err(|_| BccSchedulerError::new("CLOCK_MONOTONIC returned negative seconds"))?;
        let nanoseconds = u64::try_from(value.tv_nsec)
            .map_err(|_| BccSchedulerError::new("CLOCK_MONOTONIC returned negative nanoseconds"))?;
        seconds
            .checked_mul(1_000_000_000)
            .and_then(|current| current.checked_add(nanoseconds))
            .ok_or_else(|| BccSchedulerError::new("CLOCK_MONOTONIC nanoseconds overflow"))
    }

    pub(crate) fn finish(
        &mut self,
        window_start_ns: u64,
        window_end_ns: u64,
    ) -> Result<Value, BccSchedulerError> {
        let report = self.poll(0).and_then(|()| {
            summarize_bcc_events(
                &self.callback.events,
                window_start_ns,
                window_end_ns,
                self.callback.lost_events,
            )
            .map(|mut report| {
                let object = report.as_object_mut().expect("scheduler report object");
                object.insert("root_tid".to_owned(), json!(self.root_tid));
                object.insert("backend".to_owned(), Value::String("bcc".to_owned()));
                object.insert(
                    "configured_perf_buffer_pages".to_owned(),
                    json!(self.buffer_pages),
                );
                object.insert(
                    "perf_buffer_pages".to_owned(),
                    json!(self.effective_buffer_pages()),
                );
                object.insert("target_cgroup_id".to_owned(), json!(self.target_cgroup_id));
                object.insert(
                    "target_cgroup_level".to_owned(),
                    json!(self.target_cgroup_level),
                );
                object.insert("kprobe_symbol".to_owned(), json!(self.kprobe_symbol));
                report
            })
        });
        let cleanup = self.close_checked();
        match (report, cleanup) {
            (Ok(report), Ok(())) => Ok(report),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Err(error), Err(cleanup)) => Err(BccSchedulerError::new(format!(
                "{error}; BCC scheduler cleanup also failed: {cleanup}"
            ))),
        }
    }

    pub(crate) fn write_events(&self, path: &Path) -> Result<(), BccSchedulerError> {
        write_scheduler_events(path, &self.callback.events)
            .map_err(|error| BccSchedulerError::new(error.to_string()))
    }

    pub(crate) fn close(&mut self) -> Result<(), BccSchedulerError> {
        self.close_checked()
    }

    fn close_checked(&mut self) -> Result<(), BccSchedulerError> {
        let Some(api) = self.api.as_ref() else {
            self.active = None;
            return Ok(());
        };
        let mut failures = Vec::new();
        if let Some(mut active) = self.active.take() {
            if active.attach_fd >= 0 {
                if unsafe { (api.close_perf_event_fd)(active.attach_fd) } < 0 {
                    failures.push(format!(
                        "unable to close scheduler kprobe perf-event fd: {}",
                        io::Error::last_os_error()
                    ));
                }
                active.attach_fd = -1;
            }
            if let Some(event_name) = active.event_name.take() {
                if unsafe { (api.detach_kprobe)(event_name.as_ptr()) } < 0 {
                    failures.push(format!(
                        "unable to detach scheduler kprobe: {}",
                        io::Error::last_os_error()
                    ));
                }
            }
            for reader in active.readers.drain(..) {
                if active.events_map_fd >= 0
                    && unsafe {
                        (api.delete_element)(
                            active.events_map_fd,
                            (&reader.cpu as *const u32).cast(),
                        )
                    } < 0
                {
                    let error = io::Error::last_os_error();
                    if error.raw_os_error() != Some(libc::ENOENT) {
                        failures.push(format!(
                            "unable to clear scheduler perf map for logical CPU {}: {error}",
                            reader.cpu
                        ));
                    }
                }
                unsafe {
                    (api.perf_reader_free)(reader.reader);
                }
            }
            if active.program_fd >= 0 {
                if unsafe { libc::close(active.program_fd) } != 0 {
                    failures.push(format!(
                        "unable to close scheduler BPF program fd: {}",
                        io::Error::last_os_error()
                    ));
                }
                active.program_fd = -1;
            }
            if !active.module.is_null() {
                unsafe {
                    (api.module_destroy)(active.module);
                }
                active.module = ptr::null_mut();
            }
        }
        self.api = None;
        if failures.is_empty() {
            Ok(())
        } else {
            Err(BccSchedulerError::new(failures.join("; ")))
        }
    }
}

impl Drop for BccSchedulerTracer {
    fn drop(&mut self) {
        let _ = self.close_checked();
    }
}

pub(crate) fn summarize_bcc_events(
    events: &[SchedulerEvent],
    window_start_ns: u64,
    window_end_ns: u64,
    lost_events: u64,
) -> Result<Value, BccSchedulerError> {
    summarize_scheduler_events(
        events,
        window_start_ns,
        window_end_ns,
        lost_events,
        "ebpf-bcc-finish-task-switch-cgroup-subtree",
        "all tasks scheduled from the measured container cgroup or descendant cgroups",
    )
    .map_err(|error| BccSchedulerError::new(error.to_string()))
}

fn validate_buffer_pages(buffer_pages: Option<u64>) -> Result<(), BccSchedulerError> {
    if buffer_pages
        .is_some_and(|pages| pages == 0 || !pages.is_power_of_two() || pages > c_int::MAX as u64)
    {
        return Err(BccSchedulerError::new(
            "BCC perf-buffer pages must be a positive power of two representable as c_int",
        ));
    }
    Ok(())
}

fn require_privilege(required: bool) -> Result<(), BccSchedulerError> {
    if required && unsafe { libc::geteuid() } != 0 {
        return Err(BccSchedulerError::new(
            "--scheduler-concurrency requires root privileges to load eBPF programs",
        ));
    }
    Ok(())
}

fn validate_cgroup_v2(root: &Path) -> Result<(), BccSchedulerError> {
    let controllers = root.join("cgroup.controllers");
    if !controllers.is_file() {
        return Err(BccSchedulerError::new(format!(
            "BCC scheduler preflight requires a cgroup v2 hierarchy at {}",
            root.display()
        )));
    }
    Ok(())
}

fn cgroup_level(path: &Path, root: &Path) -> Result<u32, BccSchedulerError> {
    if !path.is_dir() {
        return Err(BccSchedulerError::new(format!(
            "scheduler target cgroup is not a directory: {}",
            path.display()
        )));
    }
    let relative = path.strip_prefix(root).map_err(|_| {
        BccSchedulerError::new(format!(
            "cgroup path is outside the cgroup v2 hierarchy: {}",
            path.display()
        ))
    })?;
    u32::try_from(relative.components().count())
        .map_err(|_| BccSchedulerError::new("scheduler target cgroup level overflow"))
}

fn online_cpus(path: &Path) -> Result<Vec<u32>, BccSchedulerError> {
    let text = fs::read_to_string(path).map_err(|error| {
        BccSchedulerError::new(format!(
            "unable to read online logical CPUs from {}: {error}",
            path.display()
        ))
    })?;
    parse_cpu_ranges(&text)
}

fn parse_cpu_ranges(text: &str) -> Result<Vec<u32>, BccSchedulerError> {
    let mut cpus = BTreeSet::new();
    for raw in text.trim().split(',') {
        let component = raw.trim();
        if component.is_empty() {
            return Err(BccSchedulerError::new(
                "online logical CPU list contains an empty component",
            ));
        }
        if let Some((start, end)) = component.split_once('-') {
            if end.contains('-') {
                return Err(BccSchedulerError::new(format!(
                    "invalid online logical CPU range: {component:?}"
                )));
            }
            let start = start.parse::<u32>().map_err(|_| {
                BccSchedulerError::new(format!("invalid online logical CPU range: {component:?}"))
            })?;
            let end = end.parse::<u32>().map_err(|_| {
                BccSchedulerError::new(format!("invalid online logical CPU range: {component:?}"))
            })?;
            if end < start {
                return Err(BccSchedulerError::new(format!(
                    "invalid descending online logical CPU range: {component:?}"
                )));
            }
            cpus.extend(start..=end);
        } else {
            cpus.insert(component.parse::<u32>().map_err(|_| {
                BccSchedulerError::new(format!("invalid online logical CPU: {component:?}"))
            })?);
        }
    }
    if cpus.is_empty() {
        return Err(BccSchedulerError::new(
            "host reports no online logical CPUs",
        ));
    }
    Ok(cpus.into_iter().collect())
}

fn discover_kprobe_symbol(paths: &BccHostPaths) -> Result<String, BccSchedulerError> {
    let kallsyms = fs::read_to_string(&paths.kallsyms).map_err(|error| {
        BccSchedulerError::new(format!(
            "unable to read kernel symbols from {}: {error}",
            paths.kallsyms.display()
        ))
    })?;
    let available = fs::read_to_string(&paths.available_filter_functions).map_err(|error| {
        BccSchedulerError::new(format!(
            "unable to read traceable kernel functions from {}: {error}",
            paths.available_filter_functions.display()
        ))
    })?;
    let blacklist = match fs::read_to_string(&paths.kprobe_blacklist) {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => String::new(),
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => String::new(),
        Err(error) => {
            return Err(BccSchedulerError::new(format!(
                "unable to read kprobe blacklist from {}: {error}",
                paths.kprobe_blacklist.display()
            )))
        }
    };
    let available = available
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .collect::<BTreeSet<_>>();
    let blacklisted = blacklist
        .lines()
        .filter_map(|line| line.split_whitespace().nth(1))
        .collect::<BTreeSet<_>>();
    let mut matches = BTreeSet::new();
    for line in kallsyms.lines() {
        let columns = line.split_whitespace().collect::<Vec<_>>();
        if columns.len() < 3 || !matches!(columns[1], "t" | "T" | "w" | "W") {
            continue;
        }
        let symbol = columns[2];
        if finish_task_switch_symbol(symbol)
            && available.contains(symbol)
            && !blacklisted.contains(symbol)
        {
            matches.insert(symbol.to_owned());
        }
    }
    matches.into_iter().next().ok_or_else(|| {
        BccSchedulerError::new("finish_task_switch kprobe symbol is unavailable or not traceable")
    })
}

fn finish_task_switch_symbol(symbol: &str) -> bool {
    if symbol == "finish_task_switch" {
        return true;
    }
    let Some(suffix) = symbol.strip_prefix("finish_task_switch.isra.") else {
        return false;
    };
    !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
}

fn kprobe_event_name(symbol: &str) -> Result<CString, BccSchedulerError> {
    let sanitized = symbol.replace(['+', '.'], "_");
    CString::new(format!("p_{sanitized}"))
        .map_err(|_| BccSchedulerError::new("kprobe symbol contains an embedded NUL byte"))
}

fn create_module(api: &BccApi) -> Result<*mut c_void, BccSchedulerError> {
    let program = CString::new(BPF_PROGRAM).expect("static BPF source has no NUL");
    let module =
        unsafe { (api.module_create)(program.as_ptr(), 0, ptr::null(), 0, true, ptr::null()) };
    if module.is_null() {
        return Err(BccSchedulerError::new(
            "unable to compile scheduler BPF program with libbcc",
        ));
    }
    Ok(module)
}

fn table_fd(api: &BccApi, module: *mut c_void, name: &str) -> Result<RawFd, BccSchedulerError> {
    let name_c = CString::new(name).expect("static map name has no NUL");
    let fd = unsafe { (api.table_fd)(module, name_c.as_ptr()) };
    if fd < 0 {
        return Err(BccSchedulerError::new(format!(
            "compiled scheduler BPF module has no {name} map"
        )));
    }
    Ok(fd)
}

fn update_map<K, V>(
    api: &BccApi,
    fd: RawFd,
    key: &K,
    value: &V,
    name: &str,
) -> Result<(), BccSchedulerError> {
    if unsafe { (api.update_element)(fd, (key as *const K).cast(), (value as *const V).cast(), 0) }
        < 0
    {
        return Err(BccSchedulerError::new(format!(
            "unable to update scheduler BPF map {name}: {}",
            io::Error::last_os_error()
        )));
    }
    Ok(())
}

fn load_program(api: &BccApi, module: *mut c_void) -> Result<RawFd, BccSchedulerError> {
    let function = CString::new("trace_finish_task_switch").expect("static function name");
    let start = unsafe { (api.function_start)(module, function.as_ptr()) };
    let size = unsafe { (api.function_size)(module, function.as_ptr()) };
    if start.is_null() || size == 0 {
        return Err(BccSchedulerError::new(
            "compiled scheduler BPF module omitted trace_finish_task_switch",
        ));
    }
    let license = unsafe { (api.module_license)(module) };
    if license.is_null() {
        return Err(BccSchedulerError::new(
            "compiled scheduler BPF module has no license",
        ));
    }
    let kernel_version = unsafe { (api.module_kernel_version)(module) };
    let fd = unsafe {
        (api.function_load)(
            module,
            2,
            function.as_ptr(),
            start,
            size,
            license,
            kernel_version,
            0,
            ptr::null_mut(),
            0,
            ptr::null(),
            c_uint::MAX,
        )
    };
    if fd < 0 {
        return Err(BccSchedulerError::new(format!(
            "unable to load scheduler BPF program; root privilege and kernel helper bpf_get_current_ancestor_cgroup_id are required: {}",
            io::Error::last_os_error()
        )));
    }
    Ok(fd)
}

fn preflight_with_paths(
    library_path: Option<&Path>,
    buffer_pages: Option<u64>,
    paths: &BccHostPaths,
    require_root: bool,
) -> Result<Value, BccSchedulerError> {
    validate_buffer_pages(buffer_pages)?;
    require_privilege(require_root)?;
    validate_cgroup_v2(&paths.cgroup_root)?;
    let cpus = online_cpus(&paths.online_cpus)?;
    let symbol = discover_kprobe_symbol(paths)?;
    let api = BccApi::load(library_path)?;
    let module = create_module(&api)?;
    let loaded = load_program(&api, module);
    let mut close_error = None;
    let program_fd = match loaded {
        Ok(fd) => fd,
        Err(error) => {
            unsafe {
                (api.module_destroy)(module);
            }
            return Err(error);
        }
    };
    if unsafe { libc::close(program_fd) } != 0 {
        close_error = Some(io::Error::last_os_error());
    }
    unsafe {
        (api.module_destroy)(module);
    }
    if let Some(error) = close_error {
        return Err(BccSchedulerError::new(format!(
            "BCC scheduler preflight could not close its probe program: {error}"
        )));
    }
    Ok(json!({
        "schema_version": "scheduler-bcc-preflight.v1",
        "ready": true,
        "backend": "bcc",
        "runtime_loading": "dlopen-dlsym",
        "library": api.library.loaded_name,
        "required_symbols": REQUIRED_SYMBOLS,
        "effective_uid": unsafe { libc::geteuid() },
        "cgroup_v2_root": paths.cgroup_root,
        "online_logical_cpus": cpus,
        "kprobe_symbol": symbol,
        "kernel_helper": "bpf_get_current_ancestor_cgroup_id",
        "kernel_program_load": "passed",
        "configured_perf_buffer_pages": buffer_pages,
        "perf_buffer_pages": buffer_pages.unwrap_or(DEFAULT_PERF_BUFFER_PAGES),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command;
    use tempfile::tempdir;

    fn write_host_fixtures(root: &Path) -> (BccHostPaths, PathBuf) {
        let cgroup_root = root.join("cgroup");
        let target = cgroup_root.join("parent/target");
        fs::create_dir_all(&target).unwrap();
        fs::write(cgroup_root.join("cgroup.controllers"), "cpu memory\n").unwrap();
        let online = root.join("online");
        fs::write(&online, "0-1\n").unwrap();
        let kallsyms = root.join("kallsyms");
        fs::write(
            &kallsyms,
            concat!(
                "0000000000000000 t finish_task_switch.isra.7\n",
                "0000000000000000 t finish_task_switch.isra.7.cold\n",
            ),
        )
        .unwrap();
        let available = root.join("available_filter_functions");
        fs::write(&available, "finish_task_switch.isra.7\n").unwrap();
        let blacklist = root.join("blacklist");
        fs::write(&blacklist, "").unwrap();
        (
            BccHostPaths {
                cgroup_root,
                online_cpus: online,
                kallsyms,
                available_filter_functions: available,
                kprobe_blacklist: blacklist,
            },
            target,
        )
    }

    fn compile_shared(directory: &Path, name: &str, source: &str) -> PathBuf {
        let source_path = directory.join(format!("{name}.c"));
        let library_path = directory.join(format!("lib{name}.so"));
        fs::write(&source_path, source).unwrap();
        let output = Command::new("cc")
            .args(["-shared", "-fPIC", "-O0", "-o"])
            .arg(&library_path)
            .arg(&source_path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "fake libbcc compile failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let mut permissions = fs::metadata(&library_path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&library_path, permissions).unwrap();
        library_path
    }

    const FAKE_BCC: &str = r#"
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

typedef void (*raw_cb_t)(void *, void *, int);
typedef void (*lost_cb_t)(void *, unsigned long long);
struct perf_opts { int pid; int cpu; int wakeup_events; };
struct raw_event { uint64_t timestamp_ns; uint32_t cpu; uint32_t tid; uint8_t entering; };
struct reader { raw_cb_t raw; lost_cb_t lost; void *ctx; int cpu; int emitted; };

void *bpf_module_create_c_from_string(const char *text, unsigned int debug,
    const char **flags, int count, bool allow_rlimit, const char *device) {
    return strstr(text, "bpf_get_current_ancestor_cgroup_id") ? (void *)0x1000 : NULL;
}
void bpf_module_destroy(void *module) {}
const char *bpf_module_license(void *module) { return "GPL"; }
unsigned int bpf_module_kern_version(void *module) { return 0; }
void *bpf_function_start(void *module, const char *name) { return (void *)0x2000; }
size_t bpf_function_size(void *module, const char *name) { return 8; }
int bpf_table_fd(void *module, const char *name) {
    if (!strcmp(name, "target_cgroup_id")) return 101;
    if (!strcmp(name, "target_cgroup_level")) return 102;
    if (!strcmp(name, "events")) return 103;
    return -1;
}
int bpf_update_elem(int fd, const void *key, const void *value, unsigned long long flags) {
    return *(const uint32_t *)key <= 1 ? 0 : -1;
}
int bpf_delete_elem(int fd, const void *key) { return 0; }
int bcc_func_load(void *module, int type, const char *name, void *start,
    size_t size, const char *license, unsigned int version, int log_level,
    char *log, unsigned int log_size, const char *device, unsigned int attach_type) {
    return open("/dev/null", O_RDONLY);
}
int bpf_attach_kprobe(int fd, int retprobe, const char *event_name,
    const char *event, unsigned long long offset, int maxactive) {
    return open("/dev/null", O_RDONLY);
}
int bpf_detach_kprobe(const char *event_name) { return 0; }
int bpf_close_perf_event_fd(int fd) { return close(fd); }
void *bpf_open_perf_buffer_opts(raw_cb_t raw, lost_cb_t lost, void *ctx,
    int pages, const struct perf_opts *opts) {
    struct reader *reader = calloc(1, sizeof(*reader));
    reader->raw = raw; reader->lost = lost; reader->ctx = ctx; reader->cpu = opts->cpu;
    return reader;
}
int perf_reader_fd(void *opaque) { return 200 + ((struct reader *)opaque)->cpu; }
int perf_reader_poll(int count, void **opaque, int timeout) {
    for (int index = 0; index < count; ++index) {
        struct reader *reader = opaque[index];
        if (reader->emitted) continue;
        struct raw_event event = {0};
        event.cpu = (uint32_t)reader->cpu;
        event.tid = (uint32_t)(40 + reader->cpu);
        event.timestamp_ns = reader->cpu == 0 ? 100 : 120;
        event.entering = 1;
        reader->raw(reader->ctx, &event, sizeof(event));
        event.timestamp_ns = reader->cpu == 0 ? 200 : 180;
        event.entering = 0;
        reader->raw(reader->ctx, &event, sizeof(event));
        reader->emitted = 1;
    }
    return 0;
}
void perf_reader_free(void *reader) { free(reader); }
"#;

    #[test]
    fn cpu_ranges_and_buffer_pages_fail_closed() {
        assert_eq!(parse_cpu_ranges("0-2,4,2\n").unwrap(), [0, 1, 2, 4]);
        for invalid in ["", "1-0", "1--2", "x", "0,"] {
            assert!(parse_cpu_ranges(invalid).is_err(), "{invalid:?}");
        }
        assert!(BccSchedulerTracer::new(Some(1)).is_ok());
        assert!(BccSchedulerTracer::new(Some(64)).is_ok());
        assert!(BccSchedulerTracer::new(Some(0)).is_err());
        assert!(BccSchedulerTracer::new(Some(3)).is_err());
    }

    #[test]
    fn kprobe_discovery_matches_only_traceable_finish_task_switch_variants() {
        let directory = tempdir().unwrap();
        let (paths, _) = write_host_fixtures(directory.path());
        assert_eq!(
            discover_kprobe_symbol(&paths).unwrap(),
            "finish_task_switch.isra.7"
        );
        fs::write(
            &paths.kprobe_blacklist,
            "0000000000000000 finish_task_switch.isra.7\n",
        )
        .unwrap();
        assert!(discover_kprobe_symbol(&paths).is_err());
    }

    #[test]
    fn dynamic_loader_names_every_missing_required_symbol() {
        let directory = tempdir().unwrap();
        let library = compile_shared(
            directory.path(),
            "incomplete_bcc",
            "void *bpf_module_create_c_from_string(void) { return (void *)1; }\n",
        );
        let error = match BccApi::load(Some(&library)) {
            Ok(_) => panic!("incomplete fake libbcc unexpectedly loaded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("bpf_module_destroy"));
        assert!(error.to_string().contains("missing required symbol"));
    }

    #[test]
    fn fake_libbcc_drives_maps_per_cpu_buffers_callbacks_and_cleanup() {
        let directory = tempdir().unwrap();
        let library = compile_shared(directory.path(), "fake_bcc", FAKE_BCC);
        let (paths, target) = write_host_fixtures(directory.path());
        let target_id = target.metadata().unwrap().ino();
        let mut tracer = BccSchedulerTracer::with_library(Some(library), Some(16)).unwrap();
        tracer
            .start_with_paths(1234, &target, &paths, false)
            .unwrap();
        tracer.poll(0).unwrap();
        let report = tracer.finish(0, 300).unwrap();
        assert_eq!(report["backend"], "bcc");
        assert_eq!(report["root_tid"], 1234);
        assert_eq!(report["configured_perf_buffer_pages"], 16);
        assert_eq!(report["perf_buffer_pages"], 16);
        assert_eq!(report["target_cgroup_id"], target_id);
        assert_eq!(report["target_cgroup_level"], 2);
        assert_eq!(report["kprobe_symbol"], "finish_task_switch.isra.7");
        assert_eq!(report["events_observed"], 4);
        assert_eq!(report["lost_perf_events"], 0);
        assert_eq!(report["peak_simultaneous_logical_cores"], 2);
        assert_eq!(report["trace_complete"], true);
        assert!(tracer.api.is_none());
        assert!(tracer.active.is_none());

        let events = directory.path().join("events.jsonl");
        tracer.write_events(&events).unwrap();
        assert_eq!(fs::read_to_string(events).unwrap().lines().count(), 4);
    }

    #[test]
    fn finish_combines_primary_and_cleanup_errors_and_closes_once() {
        let directory = tempdir().unwrap();
        let failing_cleanup_source = FAKE_BCC.replace(
            concat!(
                "int bpf_detach_kprobe(const char *event_name) { return 0; }\n",
                "int bpf_close_perf_event_fd(int fd) { return close(fd); }",
            ),
            r#"static int fake_detach_calls = 0;
static int fake_close_calls = 0;
int bpf_detach_kprobe(const char *event_name) {
    ++fake_detach_calls;
    return 0;
}
int bpf_close_perf_event_fd(int fd) {
    ++fake_close_calls;
    (void)close(fd);
    errno = EIO;
    return -1;
}
int fake_bcc_detach_calls(void) { return fake_detach_calls; }
int fake_bcc_close_calls(void) { return fake_close_calls; }"#,
        );
        assert_ne!(failing_cleanup_source, FAKE_BCC);
        let library = compile_shared(
            directory.path(),
            "failing_cleanup_bcc",
            &failing_cleanup_source,
        );
        // Keep one independent dlopen reference alive so the test-only
        // counters remain queryable after the tracer releases its BCC API.
        let audit_library = DynamicLibrary::load(Some(&library)).unwrap();
        let close_calls = unsafe {
            mem::transmute::<*mut c_void, unsafe extern "C" fn() -> c_int>(
                audit_library.symbol("fake_bcc_close_calls").unwrap(),
            )
        };
        let detach_calls = unsafe {
            mem::transmute::<*mut c_void, unsafe extern "C" fn() -> c_int>(
                audit_library.symbol("fake_bcc_detach_calls").unwrap(),
            )
        };
        let (paths, target) = write_host_fixtures(directory.path());
        let mut tracer = BccSchedulerTracer::with_library(Some(library), None).unwrap();
        tracer
            .start_with_paths(1234, &target, &paths, false)
            .unwrap();

        let error = tracer.finish(300, 100).unwrap_err().to_string();
        assert!(error.contains("scheduler measurement window"));
        assert!(error.contains("BCC scheduler cleanup also failed"));
        assert!(error.contains("unable to close scheduler kprobe perf-event fd"));
        assert_eq!(unsafe { close_calls() }, 1);
        assert_eq!(unsafe { detach_calls() }, 1);

        // A failed checked close still consumes the active session. Neither
        // an explicit retry nor Drop may repeat an ownership-sensitive close.
        tracer.close().unwrap();
        drop(tracer);
        assert_eq!(unsafe { close_calls() }, 1);
        assert_eq!(unsafe { detach_calls() }, 1);
    }

    #[test]
    fn callbacks_retain_loss_and_reject_short_records_without_panicking() {
        let mut state = Box::<CallbackState>::default();
        let context = (&mut *state as *mut CallbackState).cast::<c_void>();
        unsafe {
            record_lost_callback(context, 7);
            record_lost_callback(context, 5);
            let mut byte = 0_u8;
            record_event_callback(context, (&mut byte as *mut u8).cast(), 1);
        }
        assert_eq!(state.lost_events, 12);
        assert!(state.events.is_empty());
        assert!(state.error.as_deref().unwrap().contains("invalid size"));
    }

    #[test]
    fn perf_buffer_lost_callback_invalidates_the_integrated_report() {
        let directory = tempdir().unwrap();
        let lossy_source = FAKE_BCC.replace(
            "reader->emitted = 1;",
            "if (reader->cpu == 1) reader->lost(reader->ctx, 3); reader->emitted = 1;",
        );
        let library = compile_shared(directory.path(), "lossy_bcc", &lossy_source);
        let (paths, target) = write_host_fixtures(directory.path());
        let mut tracer = BccSchedulerTracer::with_library(Some(library), None).unwrap();
        tracer
            .start_with_paths(1234, &target, &paths, false)
            .unwrap();
        let report = tracer.finish(0, 300).unwrap();
        assert_eq!(report["lost_perf_events"], 3);
        assert_eq!(report["trace_complete"], false);
    }

    #[test]
    fn preflight_classifies_kernel_program_or_helper_load_failure() {
        let directory = tempdir().unwrap();
        let failing_source = FAKE_BCC.replacen(
            "return open(\"/dev/null\", O_RDONLY);",
            "errno = EINVAL; return -1;",
            1,
        );
        let library = compile_shared(directory.path(), "helper_fail_bcc", &failing_source);
        let (paths, _) = write_host_fixtures(directory.path());
        let error = preflight_with_paths(Some(&library), None, &paths, false).unwrap_err();
        assert!(error
            .to_string()
            .contains("bpf_get_current_ancestor_cgroup_id"));
        assert!(error.to_string().contains("Invalid argument"));
    }

    #[test]
    fn fake_preflight_loads_program_without_attaching_kprobe() {
        let directory = tempdir().unwrap();
        let library = compile_shared(directory.path(), "preflight_bcc", FAKE_BCC);
        let (paths, _) = write_host_fixtures(directory.path());
        let report = preflight_with_paths(Some(&library), Some(8), &paths, false).unwrap();
        assert_eq!(report["ready"], true);
        assert_eq!(report["kernel_program_load"], "passed");
        assert_eq!(report["kprobe_symbol"], "finish_task_switch.isra.7");
        assert_eq!(report["online_logical_cpus"], json!([0, 1]));
        assert_eq!(report["configured_perf_buffer_pages"], 8);
        assert_eq!(report["perf_buffer_pages"], 8);
    }
}
