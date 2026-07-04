pub mod data;
pub mod export;
pub mod model;
pub mod spec;
pub mod train;

/// The training backend: autodiff over the compiled-in backend.
#[cfg(feature = "wgpu-vulkan")]
pub type TrainBackend = burn::backend::Autodiff<burn::backend::Vulkan>;

#[cfg(any(feature = "torch-cpu", feature = "torch-cuda"))]
pub type TrainBackend = burn::backend::Autodiff<burn::backend::libtorch::LibTorch>;

/// The default device for [`TrainBackend`]: the default GPU adapter.
#[cfg(feature = "wgpu-vulkan")]
pub fn default_device() -> burn::backend::wgpu::WgpuDevice {
    burn::backend::wgpu::WgpuDevice::default()
}

/// The default device for [`TrainBackend`]: the CPU.
#[cfg(feature = "torch-cpu")]
pub fn default_device() -> burn::backend::libtorch::LibTorchDevice {
    burn::backend::libtorch::LibTorchDevice::Cpu
}

/// The default device for [`TrainBackend`]: the first CUDA device.
#[cfg(feature = "torch-cuda")]
pub fn default_device() -> burn::backend::libtorch::LibTorchDevice {
    burn::backend::libtorch::LibTorchDevice::Cuda(0)
}
