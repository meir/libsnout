use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, PaddingConfig2d};
use burn::tensor::{Tensor, backend::Backend};

/// A `Conv2d -> BatchNorm` block (no activation).
/// Activation, when present, is applied by the caller.
#[derive(Module, Debug)]
pub struct ConvNorm<B: Backend> {
    pub conv: Conv2d<B>,
    pub bn: BatchNorm<B>,
}

#[derive(Config, Debug)]
pub struct ConvNormConfig {
    pub in_channels: usize,
    pub out_channels: usize,

    #[config(default = "3")]
    pub kernel_size: usize,

    #[config(default = "1")]
    pub stride: usize,

    /// Convolution groups. For depthwise, set `groups = in_channels = out_channels`.
    #[config(default = "1")]
    pub groups: usize,
}

impl ConvNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvNorm<B> {
        let padding = (self.kernel_size - 1) / 2;

        ConvNorm {
            conv: Conv2dConfig::new(
                [self.in_channels, self.out_channels],
                [self.kernel_size, self.kernel_size],
            )
            .with_stride([self.stride, self.stride])
            .with_padding(PaddingConfig2d::Explicit(padding, padding, padding, padding))
            .with_groups(self.groups)
            .with_bias(false)
            .init(device),
            bn: BatchNormConfig::new(self.out_channels).init(device),
        }
    }
}

impl<B: Backend> ConvNorm<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.bn.forward(self.conv.forward(x))
    }
}
