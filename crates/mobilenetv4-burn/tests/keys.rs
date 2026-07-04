use burn::backend::NdArray;
use burn::tensor::Tensor;
use burn_store::ModuleSnapshot;

use mobilenetv4_burn::model::MobileNetV4Config;

type B = NdArray;

#[test]
fn forward_shape() {
    let device = Default::default();
    let model = MobileNetV4Config::new().init::<B>(&device);
    let x = Tensor::<B, 4>::zeros([2, 4, 128, 128], &device);
    let y = model.forward(x);
    assert_eq!(y.dims(), [2, 4]);
}

#[test]
fn loads_pretrained() {
    use burn_store::SafetensorsStore;

    let device = Default::default();
    let mut model = MobileNetV4Config::new().init::<B>(&device);
    let mut store =
        SafetensorsStore::from_file("../snout-train/assets/expr_pretrain.safetensors");
    let result = model.load_from(&mut store).expect("load expr_pretrain.safetensors");
    assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
    assert!(result.missing.is_empty(), "missing: {:?}", result.missing);
    assert!(result.unused.is_empty(), "unused: {:?}", result.unused);
    assert_eq!(result.applied.len(), 232);
}

#[test]
fn dump_keys() {
    let device = Default::default();
    let model = MobileNetV4Config::new().init::<B>(&device);

    let snapshots = model.collect(None, None, false);
    let mut keys: Vec<String> = snapshots.iter().map(|s| s.full_path()).collect();
    keys.sort();
    println!("=== {} keys (skip_enum_variants=false) ===", keys.len());
    for k in &keys {
        println!("{k}");
    }
}
