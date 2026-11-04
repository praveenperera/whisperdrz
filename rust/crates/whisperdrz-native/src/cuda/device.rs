//! One CUDA context and stream, buffers from a private memory pool, and launches of the
//! embedded cuda-oxide PTX
//!
//! Everything runs on one stream, so buffers carry no event tracking and every step can
//! be captured into a CUDA graph. Each buffer holds an `Arc` of the shared [`Device`],
//! so the stream, pool and context outlive every allocation whatever order owners drop in

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::sync::Arc;

use bytemuck::Pod;
use cudarc::driver::sys::{self as cu, CUdeviceptr, CUfunction, CUmemoryPool, CUmodule, CUstream};
use cudarc::driver::{CudaContext, CudaStream, result as cur};
use eyre::{Result, WrapErr, bail, ensure, eyre};

/// Kernels from the cuda-oxide crate `rust/kernels` as compute_80 PTX, the first tier with
/// FP16 `mma.sync` at m16n8k16. The driver JIT-compiles it for the GPU it runs on;
/// `rust/scripts/build-ptx.sh` regenerates it and `check-ptx.sh` pins its inputs
static PTX_SM80: &str = concat!(
    include_str!("../../../../kernels/ptx/whisperdrz.sm80.ptx"),
    "\0"
);

/// Oldest driver API that cuBLAS 12.8 runs on, as `cuDriverGetVersion` reports it
const MIN_DRIVER: i32 = 12080;

/// Host bytes of in-flight uploads kept alive before the stream is drained, so loading
/// 3.2 GB of weights never holds a second host copy of all of them
const PENDING_LIMIT: usize = 256 << 20;

/// The stream and memory pool every buffer frees into
struct Device {
    stream: Arc<CudaStream>,
    pool: CUmemoryPool,
}

// the pool is a driver handle, usable from any thread once the context is current there
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    fn bind(&self) -> Result<()> {
        self.stream.context().bind_to_thread()?;
        Ok(())
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // the last buffer is gone, so destroying the pool hands its memory back to the driver
        if self.bind().is_ok() {
            let _ = self.stream.synchronize();
            unsafe {
                let _ = cur::mem_pool::destroy(self.pool);
            }
        }
    }
}

/// A device allocation, freed in stream order on drop
pub struct Buf {
    ptr: CUdeviceptr,
    bytes: usize,
    device: Arc<Device>,
}

impl Buf {
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Device address `offset` bytes in
    pub fn at(&self, offset: usize) -> u64 {
        debug_assert!(offset <= self.bytes);
        self.ptr + offset as u64
    }

    /// The whole buffer as a kernel slice of `T`, the element type the kernel declares
    /// for it (`u16` for FP16)
    pub fn slice<T>(&self) -> Arg {
        debug_assert!(self.bytes.is_multiple_of(size_of::<T>()));
        Arg::Slice(self.ptr, (self.bytes / size_of::<T>()) as u64)
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        // a failed free leaves the block in the pool, which Device::drop destroys
        if self.bytes > 0 && self.device.bind().is_ok() {
            unsafe {
                let _ = cur::free_async(self.ptr, self.device.stream.cu_stream());
            }
        }
    }
}

/// A kernel argument
///
/// Each `&[T]` or `DisjointSlice<T>` parameter of a cuda-oxide kernel lowers to two PTX
/// params, the address then the length in elements of `T`, which is [`Arg::Slice`]. The
/// rest are 32-bit scalars
#[derive(Clone, Copy, Debug)]
pub enum Arg {
    /// Address and length in elements of the kernel's slice type
    Slice(u64, u64),
    U32(u32),
    I32(i32),
    F32(f32),
}

impl Arg {
    /// An empty slice for an optional input the kernel skips; the address is never read,
    /// but a Rust slice's must be non-null and aligned even when empty
    pub const EMPTY: Arg = Arg::Slice(256, 0);

    /// `len` elements of the kernel's slice type at `ptr`
    pub fn slice(ptr: u64, len: usize) -> Self {
        Arg::Slice(ptr, len as u64)
    }

    fn params(&self) -> &'static [Param] {
        match self {
            Arg::Slice(..) => &[Param::Ptr, Param::U64],
            // the PTX declares i32 params as .u32 too
            Arg::U32(_) | Arg::I32(_) => &[Param::U32],
            Arg::F32(_) => &[Param::F32],
        }
    }
}

impl From<i32> for Arg {
    fn from(v: i32) -> Self {
        Arg::I32(v)
    }
}

impl From<u32> for Arg {
    fn from(v: u32) -> Self {
        Arg::U32(v)
    }
}

impl From<usize> for Arg {
    fn from(v: usize) -> Self {
        Arg::U32(u32::try_from(v).expect("kernel size argument overflows u32"))
    }
}

impl From<f32> for Arg {
    fn from(v: f32) -> Self {
        Arg::F32(v)
    }
}

/// PTX param kinds, read from the `.entry` declarations
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Param {
    Ptr,
    U64,
    U32,
    F32,
}

struct Kernel {
    f: CUfunction,
    params: Vec<Param>,
}

impl Kernel {
    /// Checks that `args` fill the kernel's params exactly: `cuLaunchKernel` reads each
    /// param at the size the PTX declares, so a dropped slice length would silently shift
    /// every later argument
    fn check(&self, name: &str, args: &[Arg]) -> Result<()> {
        let host: Vec<Param> = args.iter().flat_map(Arg::params).copied().collect();
        ensure!(
            host == self.params,
            "kernel {name}: the PTX declares {:?}, the host passes {host:?}",
            self.params
        );
        Ok(())
    }
}

/// Where a [`Gpu`]'s stream sits in the block scheduler, which hands free SMs to
/// pending blocks of higher-priority streams first. Blocks already running are never
/// preempted, so a background stream still delays the others by up to one of its blocks
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Priority {
    /// The CUDA default, which is the lowest priority
    Default,
    /// The greatest priority the context offers, for latency-bound work
    High,
    /// The lowest priority, with host waits that sleep rather than spin, so a helper
    /// thread waiting on the GPU leaves its core to the host work beside it
    Background,
}

/// The GPU: context, stream, pool and the kernel module
pub struct Gpu {
    device: Arc<Device>,
    module: CUmodule,
    functions: RefCell<HashMap<&'static str, Kernel>>,
    /// Streaming multiprocessors, for grids sized to whole waves
    sms: usize,
    /// Compute capability, major and minor
    cc: (i32, i32),
    /// Dynamic shared memory a block may opt into, in bytes
    smem_optin: usize,
    /// Host sources of asynchronous uploads, kept until the stream next synchronizes
    pending: RefCell<Vec<Box<dyn Any + Send>>>,
    pending_bytes: Cell<usize>,
    /// A blocking-sync event that [`Gpu::synchronize`] waits on instead of the stream,
    /// for [`Priority::Background`]
    sleep_event: Option<cu::CUevent>,
}

// CUDA handles work from any thread once the context is current there, and Gpu isn't
// Sync, so one thread uses it at a time
unsafe impl Send for Gpu {}

impl Gpu {
    pub fn new(ordinal: usize) -> Result<Self> {
        Self::with_priority(ordinal, Priority::Default)
    }

    /// A GPU handle with its own stream at `priority`, pool and module, on the device's
    /// primary context, so handles made on several threads share one context
    pub fn with_priority(ordinal: usize, priority: Priority) -> Result<Self> {
        check_driver()?;
        let ctx =
            CudaContext::new(ordinal).wrap_err_with(|| format!("open CUDA device {ordinal}"))?;
        let (major, minor) = ctx.compute_capability()?;
        ensure!(
            major >= 8,
            "GPU {ordinal} has compute capability {major}.{minor}; whisperdrz needs 8.0 or newer"
        );
        let sms = ctx.attribute(cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?
            as usize;
        let smem_optin = ctx.attribute(
            cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )? as usize;

        // a single stream, so cudarc's cross-stream event tracking is pure overhead
        unsafe { ctx.disable_event_tracking() };
        // the context is current on this thread from `CudaContext::new`
        let (least, greatest) = cur::stream::get_priority_range()?;
        let stream = match priority {
            Priority::Default => ctx.new_stream()?,
            Priority::High => ctx.new_stream_with_priority(greatest)?,
            Priority::Background => ctx.new_stream_with_priority(least)?,
        };
        let sleep_event = match priority {
            Priority::Background => Some(cur::event::create(
                cu::CUevent_flags::CU_EVENT_BLOCKING_SYNC,
            )?),
            _ => None,
        };
        let device = Arc::new(Device {
            pool: create_pool(&ctx)?,
            stream,
        });
        let module = unsafe { cur::module::load_data(PTX_SM80.as_ptr().cast()) }
            .wrap_err("JIT-compile the whisperdrz kernels")?;
        Ok(Self {
            device,
            module,
            functions: RefCell::default(),
            sms,
            cc: (major, minor),
            smem_optin,
            pending: RefCell::default(),
            pending_bytes: Cell::new(0),
            sleep_event,
        })
    }

    pub fn sms(&self) -> usize {
        self.sms
    }

    /// Shared memory a block may use with the opt-in launch attribute, in bytes
    pub fn max_shared_per_block(&self) -> usize {
        self.smem_optin
    }

    /// Compute capability as `(major, minor)`
    pub fn compute_capability(&self) -> (i32, i32) {
        self.cc
    }

    pub fn cu_stream(&self) -> CUstream {
        self.device.stream.cu_stream()
    }

    pub fn synchronize(&self) -> Result<()> {
        match self.sleep_event {
            // SAFETY: the event belongs to this handle and lives until it drops
            Some(event) => unsafe {
                cur::event::record(event, self.cu_stream())?;
                cu::cuEventSynchronize(event).result()?;
            },
            None => self.device.stream.synchronize()?,
        }
        self.pending.borrow_mut().clear();
        self.pending_bytes.set(0);
        Ok(())
    }

    /// GPU time in milliseconds of the work `enqueue` puts on the stream, from events
    /// recorded around it. Synchronizes, so call it outside a graph capture
    pub fn time_ms(&self, enqueue: impl FnOnce() -> Result<()>) -> Result<f32> {
        let flags = cu::CUevent_flags::CU_EVENT_DEFAULT;
        let start = cur::event::create(flags)?;
        let end = match cur::event::create(flags) {
            Ok(end) => end,
            Err(e) => {
                // SAFETY: `start` was created above and nothing has recorded it
                unsafe {
                    let _ = cur::event::destroy(start);
                }
                return Err(e.into());
            }
        };
        let timed = (|| {
            unsafe { cur::event::record(start, self.cu_stream())? };
            enqueue()?;
            unsafe { cur::event::record(end, self.cu_stream())? };
            self.synchronize()?;
            Ok(unsafe { cur::event::elapsed(start, end)? })
        })();
        unsafe {
            let _ = cur::event::destroy(start);
            let _ = cur::event::destroy(end);
        }
        timed
    }

    pub fn alloc(&self, bytes: usize) -> Result<Buf> {
        let mut ptr = 0;
        if bytes > 0 {
            unsafe {
                cu::cuMemAllocFromPoolAsync(&mut ptr, bytes, self.device.pool, self.cu_stream())
                    .result()
            }
            .wrap_err_with(|| format!("allocate {bytes} bytes on the GPU"))?;
        }
        Ok(Buf {
            ptr,
            bytes,
            device: self.device.clone(),
        })
    }

    pub fn zeros(&self, bytes: usize) -> Result<Buf> {
        let buf = self.alloc(bytes)?;
        self.memset_zero(buf.ptr, bytes)?;
        Ok(buf)
    }

    pub fn memset_zero(&self, ptr: u64, bytes: usize) -> Result<()> {
        if bytes > 0 {
            unsafe { cur::memset_d8_async(ptr, 0, bytes, self.cu_stream())? };
        }
        Ok(())
    }

    /// Copies `host` into a new buffer. The copy is asynchronous, so `host` stays alive
    /// until the stream next synchronizes
    pub fn upload<T: Pod + Send>(&self, host: Vec<T>) -> Result<Buf> {
        let bytes: &[u8] = bytemuck::cast_slice(&host);
        let buf = self.alloc(bytes.len())?;
        if bytes.is_empty() {
            return Ok(buf);
        }

        unsafe { cur::memcpy_htod_async(buf.ptr, bytes, self.cu_stream())? };
        self.pending_bytes
            .set(self.pending_bytes.get() + bytes.len());
        self.pending.borrow_mut().push(Box::new(host));
        if self.pending_bytes.get() > PENDING_LIMIT {
            self.synchronize()?;
        }

        Ok(buf)
    }

    /// Copies borrowed host bytes to `dst` and waits, since the caller may reuse them
    pub fn upload_to(&self, dst: u64, host: &[u8]) -> Result<()> {
        if !host.is_empty() {
            unsafe { cur::memcpy_htod_async(dst, host, self.cu_stream())? };
            self.synchronize()?;
        }

        Ok(())
    }

    pub fn copy_dtod(&self, src: u64, dst: u64, bytes: usize) -> Result<()> {
        if bytes > 0 {
            unsafe { cur::memcpy_dtod_async(dst, src, bytes, self.cu_stream())? };
        }

        Ok(())
    }

    /// Copies `host.len()` elements starting `offset` bytes into `src` and waits
    pub fn download_into<T: Pod>(&self, src: &Buf, offset: usize, host: &mut [T]) -> Result<()> {
        let host: &mut [u8] = bytemuck::cast_slice_mut(host);
        ensure!(
            offset
                .checked_add(host.len())
                .is_some_and(|end| end <= src.bytes),
            "download of {} bytes at {offset} from a {} byte buffer",
            host.len(),
            src.bytes
        );
        // a copy to pageable memory returns only once the stream drains, and while it
        // waits it holds up other threads' pageable copies (nsys: the decoder's logits
        // download waited out whole background embeds). A background stream drains first
        // on its sleeping event, so its copy finds the stream idle
        if self.sleep_event.is_some() {
            self.synchronize()?;
        }

        if !host.is_empty() {
            unsafe { cur::memcpy_dtoh_async(host, src.ptr + offset as u64, self.cu_stream())? };
        }

        self.synchronize()
    }

    /// The whole buffer as `T`s
    pub fn download<T: Pod + Default>(&self, src: &Buf) -> Result<Vec<T>> {
        let mut host = vec![T::default(); src.bytes / size_of::<T>()];
        self.download_into(src, 0, &mut host)?;
        Ok(host)
    }

    fn function(&self, name: &'static str, args: &[Arg]) -> Result<CUfunction> {
        let mut functions = self.functions.borrow_mut();
        if let Some(k) = functions.get(name) {
            k.check(name, args)?;
            return Ok(k.f);
        }

        let cname = CString::new(name)?;
        let f = unsafe { cur::module::get_function(self.module, cname) }
            .wrap_err_with(|| format!("kernel {name} missing from the PTX module"))?;
        let kernel = Kernel {
            f,
            params: entry_params(PTX_SM80, name)?,
        };
        kernel.check(name, args)?;
        functions.insert(name, kernel);
        Ok(f)
    }

    /// Launches kernel `name` from the embedded PTX
    pub fn launch(
        &self,
        name: &'static str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        smem: u32,
        args: &[Arg],
    ) -> Result<()> {
        let f = self.function(name, args)?;
        // the elementwise kernels' 32-bit thread index is exact only up to here
        ensure!(
            grid.0 as u64 * block.0 as u64 <= 1 << 32,
            "launch {name}: more than 2^32 threads along x"
        );
        if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 {
            return Ok(());
        }

        let mut params: Vec<*mut c_void> = Vec::with_capacity(2 * args.len());
        for a in args {
            match a {
                Arg::Slice(ptr, len) => {
                    params.extend([
                        ptr as *const u64 as *mut c_void,
                        len as *const u64 as *mut c_void,
                    ]);
                }
                Arg::U32(v) => params.push(v as *const u32 as *mut c_void),
                Arg::I32(v) => params.push(v as *const i32 as *mut c_void),
                Arg::F32(v) => params.push(v as *const f32 as *mut c_void),
            }
        }

        if smem > 48 * 1024 {
            unsafe {
                cu::cuFuncSetAttribute(
                    f,
                    cu::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    smem as i32,
                )
                .result()?;
            }
        }

        unsafe { cur::launch_kernel(f, grid, block, smem, self.cu_stream(), &mut params) }
            .wrap_err_with(|| format!("launch {name}"))?;
        Ok(())
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        if self.device.bind().is_ok() {
            let _ = self.device.stream.synchronize();
            unsafe {
                let _ = cur::module::unload(self.module);
                if let Some(event) = self.sleep_event {
                    let _ = cur::event::destroy(event);
                }
            }
        }
    }
}

/// The params of kernel `name` in `ptx`, in order, read from the PTX text
fn entry_params(ptx: &str, name: &str) -> Result<Vec<Param>> {
    let entry = format!(".entry {name}(");
    let start = ptx
        .find(&entry)
        .ok_or_else(|| eyre!("kernel {name} missing from the PTX"))?
        + entry.len();
    let end = ptx[start..]
        .find(')')
        .ok_or_else(|| eyre!("kernel {name}: unterminated PTX param list"))?;
    ptx[start..start + end]
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| {
            let ty = line
                .strip_prefix(".param ")
                .and_then(|rest| rest.split_whitespace().next());
            match (ty, line.contains(" .ptr ")) {
                (Some(".u64"), true) => Ok(Param::Ptr),
                (Some(".u64"), false) => Ok(Param::U64),
                (Some(".u32"), false) => Ok(Param::U32),
                (Some(".f32"), false) => Ok(Param::F32),
                _ => Err(eyre!("kernel {name}: unexpected PTX param `{line}`")),
            }
        })
        .collect()
}

/// A pool of our own, so caching freed blocks doesn't change the default pool for other
/// CUDA users in the process
fn create_pool(ctx: &CudaContext) -> Result<CUmemoryPool> {
    let pool = unsafe {
        let mut props: cu::CUmemPoolProps = std::mem::zeroed();
        props.allocType = cu::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        props.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.id = ctx.cu_device();
        cur::mem_pool::create(&props).wrap_err("create the CUDA memory pool")?
    };

    // keep freed blocks rather than returning them to the driver at every synchronize
    let mut threshold = u64::MAX;
    unsafe {
        cur::mem_pool::set_attribute(
            pool,
            cu::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut threshold as *mut u64).cast(),
        )?;
    }

    Ok(pool)
}

fn check_driver() -> Result<()> {
    unsafe {
        if !cu::is_culib_present() {
            bail!("no NVIDIA driver found (libcuda.so.1)");
        }

        let mut version = 0;
        cu::cuDriverGetVersion(&mut version).result()?;
        ensure!(
            version >= MIN_DRIVER,
            "the NVIDIA driver supports CUDA {}.{}; whisperdrz needs 12.8 or newer",
            version / 1000,
            version % 1000 / 10
        );
    }

    Ok(())
}

/// Page-locked host memory from [`Gpu::alloc_host`]. The copy engine writes it
/// directly, so a download into it is queued like a kernel and lands while the host
/// works, where a pageable download returns only once its stream drains
pub struct HostBuf {
    ptr: *mut u8,
    bytes: usize,
    device: Arc<Device>,
}

// the pointer is page-locked memory owned by this buffer alone
unsafe impl Send for HostBuf {}

impl HostBuf {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// `len` `T`s starting `offset` bytes in
    ///
    /// # Safety
    ///
    /// No copy into those bytes may be in flight, or start while the slice lives
    pub unsafe fn slice<T: Pod>(&self, offset: usize, len: usize) -> Result<&[T]> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .and_then(|b| b.checked_add(offset))
            .filter(|&end| end <= self.bytes);
        ensure!(
            bytes.is_some() && offset.is_multiple_of(align_of::<T>()),
            "{len} elements at {offset} of a {} byte host buffer",
            self.bytes
        );
        // SAFETY: in bounds and aligned (the allocation is page aligned), and the caller
        // rules out concurrent device writes
        Ok(unsafe { std::slice::from_raw_parts(self.ptr.add(offset).cast::<T>(), len) })
    }
}

impl Drop for HostBuf {
    fn drop(&mut self) {
        if self.device.bind().is_ok() {
            // SAFETY: allocated by `cuMemHostAlloc`; a copy into it can't outlive the
            // owner that queued it, which syncs or drops the stream first
            unsafe {
                let _ = self.device.stream.synchronize();
                let _ = cur::free_host(self.ptr.cast());
            }
        }
    }
}

/// A CUDA event without timing, recorded on a [`Gpu`]'s stream
pub struct Event {
    event: cu::CUevent,
    device: Arc<Device>,
}

// a driver handle, usable from any thread with the context current
unsafe impl Send for Event {}

impl Event {
    /// Waits until the work queued before the last record has finished
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: the event belongs to this handle and lives until it drops
        unsafe { cur::event::synchronize(self.event)? };
        Ok(())
    }

    /// Whether the work queued before the last record has finished, without waiting
    pub fn query(&self) -> Result<bool> {
        // SAFETY: as in `synchronize`
        match unsafe { cur::event::query(self.event) } {
            Ok(()) => Ok(true),
            Err(e) if e.0 == cu::CUresult::CUDA_ERROR_NOT_READY => Ok(false),
            Err(e) => Err(e.into()),
        }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if self.device.bind().is_ok() {
            // SAFETY: created by `Gpu::event` and not used after this
            unsafe {
                let _ = cur::event::destroy(self.event);
            }
        }
    }
}

impl Gpu {
    /// `bytes` of page-locked host memory
    pub fn alloc_host(&self, bytes: usize) -> Result<HostBuf> {
        ensure!(bytes > 0, "empty host buffer");
        // SAFETY: the context is current on this thread and the size is non-zero
        let ptr = unsafe { cur::malloc_host(bytes, 0) }
            .wrap_err_with(|| format!("allocate {bytes} bytes of page-locked host memory"))?;
        Ok(HostBuf {
            ptr: ptr.cast(),
            bytes,
            device: self.device.clone(),
        })
    }

    pub fn event(&self) -> Result<Event> {
        let event = cur::event::create(cu::CUevent_flags::CU_EVENT_DISABLE_TIMING)?;
        Ok(Event {
            event,
            device: self.device.clone(),
        })
    }

    /// Queues a copy of `bytes` bytes from `offset` in `src` to `dst_offset` in `dst`,
    /// then records `done`; nothing waits. `dst` must not be read there until `done`
    /// completes
    pub fn download_async(
        &self,
        src: &Buf,
        offset: usize,
        dst: &HostBuf,
        dst_offset: usize,
        bytes: usize,
        done: &Event,
    ) -> Result<()> {
        let fits = |at: usize, len: usize| at.checked_add(bytes).is_some_and(|end| end <= len);
        ensure!(
            fits(offset, src.bytes) && fits(dst_offset, dst.bytes),
            "async download of {bytes} bytes from {offset} of {} to {dst_offset} of {}",
            src.bytes,
            dst.bytes
        );
        // SAFETY: both ranges are in bounds, the destination is page-locked, and the
        // event belongs to this context
        unsafe {
            if bytes > 0 {
                cu::cuMemcpyDtoHAsync_v2(
                    dst.ptr.add(dst_offset).cast::<c_void>(),
                    src.ptr + offset as u64,
                    bytes,
                    self.cu_stream(),
                )
                .result()?;
            }

            cur::event::record(done.event, self.cu_stream())?;
        }

        Ok(())
    }
}

/// Grid size for `n` items with `block` threads each
pub fn blocks(n: usize, block: u32) -> u32 {
    u32::try_from(n.div_ceil(block as usize)).expect("grid too large")
}

#[cfg(test)]
mod tests {
    use super::{PTX_SM80, Param, entry_params};

    /// The launch guard reads every kernel's params from the PTX text, so a cuda-oxide or
    /// LLVM change to the param syntax must fail here rather than at a launch
    #[test]
    fn ptx_params_parse() {
        let names: Vec<&str> = PTX_SM80
            .match_indices(".entry ")
            .map(|(i, m)| {
                let rest = &PTX_SM80[i + m.len()..];
                &rest[..rest.find('(').unwrap()]
            })
            .collect();
        assert!(!names.is_empty());
        for name in names {
            let params = entry_params(PTX_SM80, name).unwrap();
            assert!(params.contains(&Param::Ptr), "{name} has no slice params");
        }
    }
}
