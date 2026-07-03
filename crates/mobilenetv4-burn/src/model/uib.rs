use burn::config::Config;
use burn::module::Module;
use burn::tensor::activation::relu;
use burn::tensor::{Tensor, backend::Backend};

use super::conv_norm::{ConvNorm, ConvNormConfig};

/// Universal Inverted Bottleneck (MobileNetV4).
///
/// `dw_start` (optional, no act) -> `pw_exp` (act) -> `dw_mid` (optional, act) -> `pw_proj` (no act).
/// A residual connection is added when the input and output shapes match (stride 1, in == out).
#[derive(Module, Debug)]
pub struct Uib<B: Backend> {
    pub dw_start: Option<ConvNorm<B>>,
    pub pw_exp: ConvNorm<B>,
    pub dw_mid: Option<ConvNorm<B>>,
    pub pw_proj: ConvNorm<B>,
    /// Whether the residual connection applies (static: stride 1 and in == out).
    pub use_residual: bool,
}

#[derive(Config, Debug)]
pub struct UibConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub exp_channels: usize,

    /// Depthwise kernel before the expansion (None = absent).
    #[config(default = "None")]
    pub dw_start_kernel: Option<usize>,

    /// Depthwise kernel after the expansion (None = absent).
    #[config(default = "None")]
    pub dw_mid_kernel: Option<usize>,

    #[config(default = "1")]
    pub stride: usize,
}

impl UibConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Uib<B> {
        // Stride is applied at the first depthwise conv that exists.
        let (start_stride, mid_stride) = if self.dw_start_kernel.is_some() {
            (self.stride, 1)
        } else {
            (1, self.stride)
        };

        let dw_start = self.dw_start_kernel.map(|k| {
            ConvNormConfig::new(self.in_channels, self.in_channels)
                .with_kernel_size(k)
                .with_stride(start_stride)
                .with_groups(self.in_channels)
                .init(device)
        });

        let pw_exp = ConvNormConfig::new(self.in_channels, self.exp_channels)
            .with_kernel_size(1)
            .init(device);

        let dw_mid = self.dw_mid_kernel.map(|k| {
            ConvNormConfig::new(self.exp_channels, self.exp_channels)
                .with_kernel_size(k)
                .with_stride(mid_stride)
                .with_groups(self.exp_channels)
                .init(device)
        });

        let pw_proj = ConvNormConfig::new(self.exp_channels, self.out_channels)
            .with_kernel_size(1)
            .init(device);

        // Residual applies only when input and output shapes match; this is fixed by
        // the topology, so decide it once here rather than inspecting `dims()` per
        // forward pass.
        let use_residual = self.stride == 1 && self.in_channels == self.out_channels;

        Uib {
            dw_start,
            pw_exp,
            dw_mid,
            pw_proj,
            use_residual,
        }
    }
}

impl<B: Backend> Uib<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Keep the input only if it will be added back as a residual.
        let residual = self.use_residual.then(|| x.clone());

        let mut out = x;
        if let Some(dw) = &self.dw_start {
            out = dw.forward(out);
        }
        out = relu(self.pw_exp.forward(out));
        if let Some(dw) = &self.dw_mid {
            out = relu(dw.forward(out));
        }
        out = self.pw_proj.forward(out);

        if let Some(identity) = residual {
            out = out + identity;
        }
        out
    }
}

/// A network block: either a `Conv -> BN -> ReLU` (ConvBnAct) or a UIB.
#[derive(Module, Debug)]
pub enum Block<B: Backend> {
    ConvBnAct(ConvNorm<B>),
    Uib(Uib<B>),
}

impl<B: Backend> Block<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Block::ConvBnAct(conv) => relu(conv.forward(x)),
            Block::Uib(uib) => uib.forward(x),
        }
    }
}
