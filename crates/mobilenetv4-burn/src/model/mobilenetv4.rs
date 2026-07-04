use burn::config::Config;
use burn::module::Module;
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::activation::relu;
use burn::tensor::{Tensor, backend::Backend};

use super::conv_norm::{ConvNorm, ConvNormConfig};
use super::uib::{Block, UibConfig};

/// MobileNetV4 `conv_small` at width 0.5, as shipped in the Babble expression checkpoint.
/// Input `[B, in_channels, 128, 128]` -> `[B, num_classes]`.
#[derive(Module, Debug)]
pub struct MobileNetV4<B: Backend> {
    pub stem: ConvNorm<B>,
    pub blocks: Vec<Block<B>>,
    pub head_conv: ConvNorm<B>,
    pub pool: AdaptiveAvgPool2d,
    pub dropout: Dropout,
    pub classifier: Linear<B>,
}

impl<B: Backend> MobileNetV4<B> {
    /// Full forward pass: `[B, C, H, W]` -> `[B, num_classes]`.
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        self.forward_head(self.forward_features(input))
    }

    /// Classifier head applied to pooled features: `[B, 1280]` -> `[B, num_classes]`.
    /// Useful for training the head on precomputed (e.g. frozen-backbone) features.
    pub fn forward_head(&self, features: Tensor<B, 2>) -> Tensor<B, 2> {
        self.classifier.forward(self.dropout.forward(features))
    }

    /// Feature extraction only: `[B, C, H, W]` -> pooled `[B, 1280]`.
    pub fn forward_features(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut x = relu(self.stem.forward(input));
        for block in &self.blocks {
            x = block.forward(x);
        }
        x = relu(self.head_conv.forward(x));
        let x = self.pool.forward(x);
        x.flatten(1, 3)
    }
}

#[derive(Config, Debug)]
pub struct MobileNetV4Config {
    #[config(default = "4")]
    pub num_classes: usize,

    #[config(default = "4")]
    pub in_channels: usize,

    #[config(default = "0.2")]
    pub dropout: f64,
}

impl MobileNetV4Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MobileNetV4<B> {
        let cn = |in_c, out_c, k, s| {
            Block::ConvBnAct(
                ConvNormConfig::new(in_c, out_c)
                    .with_kernel_size(k)
                    .with_stride(s)
                    .init(device),
            )
        };
        let uib = |in_c, out_c, exp, dw_start: Option<usize>, dw_mid: Option<usize>, s| {
            Block::Uib(
                UibConfig::new(in_c, out_c, exp)
                    .with_dw_start_kernel(dw_start)
                    .with_dw_mid_kernel(dw_mid)
                    .with_stride(s)
                    .init(device),
            )
        };

        let stem = ConvNormConfig::new(self.in_channels, 32)
            .with_kernel_size(3)
            .with_stride(2)
            .init(device);

        let blocks = vec![
            // Stage 0 (ConvBnAct)
            cn(32, 16, 3, 2),
            cn(16, 16, 1, 1),
            // Stage 1 (ConvBnAct)
            cn(16, 48, 3, 2),
            cn(48, 32, 1, 1),
            // Stage 2 (UIB)
            uib(32, 48, 96, Some(5), Some(5), 2),
            uib(48, 48, 96, None, Some(3), 1),
            uib(48, 48, 96, None, Some(3), 1),
            uib(48, 48, 96, None, Some(3), 1),
            uib(48, 48, 96, None, Some(3), 1),
            uib(48, 48, 192, Some(3), None, 1),
            // Stage 3 (UIB)
            uib(48, 64, 288, Some(3), Some(3), 2),
            uib(64, 64, 256, Some(5), Some(5), 1),
            uib(64, 64, 256, None, Some(5), 1),
            uib(64, 64, 192, None, Some(5), 1),
            uib(64, 64, 256, None, Some(3), 1),
            uib(64, 64, 256, None, Some(3), 1),
            // Stage 4 (ConvBnAct)
            cn(64, 480, 1, 1),
        ];

        let head_conv = ConvNormConfig::new(480, 1280)
            .with_kernel_size(1)
            .init(device);

        MobileNetV4 {
            stem,
            blocks,
            head_conv,
            pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
            classifier: LinearConfig::new(1280, self.num_classes).init(device),
        }
    }
}
