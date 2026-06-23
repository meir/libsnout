# Libsnout

This is a rust implementation of Project Babble's baballonia face tracking sofware.
It's designed to be a library; easy to integrate in a variety of frontend projects. However it can also be used as a CLI application through snout-cli.

- Building:
    - [Building and running the cli](#building-and-running-the-cli)
    - [Installing on NixOS](#installing-on-nixos)
- Configuring:
    - [Configuration file location](#configuration-file-location)
    - [Disabling specific tracking points](#disabling-specific-tracking-points)
    - [Finding your camera](#finding-your-camera)
    - [Using with VRCFT or oscavmgr](#using-with-vrcft-or-oscavmgr)
    - [Using with VRC native Eye Tracking](#using-with-vrc-native-eye-tracking)
    - [Face calibration](#face-calibration)
    - [Filter Settings](#filter-settings)
    - [Note on onnx model paths](#note-on-onnx-model-paths)
    - [Using a non-system onnxruntime](#using-a-non-system-onnxruntime)
- Usage:
    - [Tracking](#tracking)
    - [Training an eye model](#training-an-eye-model)
    - [Troubleshooting](#troubleshooting)
    - [Cropping](#notes-on-cropping)

## Required dependencies

Libsnout requires the following build dependencies (in the form of fedora package names):

- llvm
- llvm-devel
- onnxruntime
- onnxruntime-devel
- rust

## Building and running the CLI

Clone the repository,
```sh
git clone https://github.com/Darksecond/libsnout.git
```

and then build the program.

```sh
cd libsnout
cargo build --release -p snout-cli
```

The snout-cli executable will be located under `target/release/`

Help on how to use the cli tool can be obtained with:

```sh
snout-cli help
``` 

### Installing on NixOS

Add libsnout to your flake.nix inputs:

```nix
  libsnout.url = "github:Darksecond/libsnout";
```

Either use the package directly or add it to your overlays:

```nix
nixpkgs.overlay = [
  (final: prev: {
    snout-cli = libsnout.packages."${pkgs.stdenv.hostPlatform.system}".default;
  })
];

environment.systemPackages = with pkgs; [
  snout-cli
];
```
## Configuring 

### Configuration file location

snout-cli will search for a configuration file called `config.toml` in the following locations:

- $XDG_CONFIG_HOME/snout/config.toml
- $HOME/.config/snout/config.toml
- $HOME/.snout/config.toml
- /etc/snout/config.toml

A template configuration file can be found in this repo.

Make sure to edit it to suit your needs.

Relative paths referenced in the configuration file, will be relative to the location of the configuration file.

A specific configuration file, not located in any of the above paths, can still be used by specifying it through the `-c` flag when running snout-cli. Like so:

```sh
snout-cli -c ~/myconfig.toml track
```

### Disabling specific tracking points

Tracking can be disabled for specific points by setting their `camera` value to an empty string. Like so:

```toml 
[eye.right]
camera = ""

# <...>

[eye.left]
camera = ""

# <...>

[face]
camera = "http://192.168.178.162"

# <...> 
``` 

The above example will disable both of the eye cameras, leaving only the face camera active.

### Finding your camera

The names of connected usb cameras can be found like so: 

```sh
snout-cli list-cameras
``` 

Once you have located your desired camera in the outputted list, use the full name of the camera in the configuration file.

```toml
[eye.right]
camera = "Bigeye: Bigeye (800x400 @ 90fps)"
``` 

Wireless mjpeg cameras can be entered as a url, like so:

```toml
[eye.right]
camera = "http://192.168.178.162"
``` 

### Using with VRCFT or oscavmgr

The osc endpoint that tracking data gets sent to will need to be adjusted to be used with VRCFT.
The following configuration will work with VRCFT.avalonia:

```toml
[output.osc] 
destination = "127.0.0.1:8888"
``` 

The default endpoint if none is supplied in the config, already works with oscavmgr. But can be set manually, like so:

```toml
[output.osc] 
destination = "127.0.0.1:9400"
``` 

### Using with VRC native Eye Tracking

VRC offers a native eye tracking solution over osc that doesnt require a bridge like VRCFT or OscAvMgr, and works with any avatar.

It can be enabled by uncommenting the following lines in the configuration file, like so:

```toml
[output.vrchat]
destination = "127.0.0.1:9000"
max_pitch = 20.0 # Optional
max_yaw = 30.0 # Optional
```

if VRChat was set to use a different port than default for recieving OSC messages, changing the destination port is also required.

### Face calibration

Face calibration can be done by adjusting the upper and lower bounds of the different `[[face.calibration]]` tables in the configuration file, like so:

```toml
[[face.calibration]]
shape = "CheekPuffLeft"
lower = 0.3
upper = 0.6
```

The full list of shapes can be found in the template configuration file.

### Filter Settings

Filter settings for the eye and face tracking pipelines can be changed by adjusting the values of the `[eye.filter]` and `[face.filter]` tables in the configuration file, like so:

```toml
[eye.filter]
enable = true
min_cutoff = 0.5
beta = 3.0

# <...>

[face.filter]
enable = true
min_cutoff = 0.5
beta = 3.0
```

### Note on onnx model paths

The paths to the face and eye tracking onnx models are relative to the directory of the config file. An absolute path may be preferred and can be set by prefixing the path with a `/` like so:

```toml
[face]

# <...>

model = "/home/user/libsnout/faceModel.onnx" 
``` 

### Using a non-system onnxruntime

A libonnxruntime library file can be supplied in the configuration file through the `libonnxruntime` key, like so:

```toml
libonnxruntime = "onnxruntime-linux-x64-gpu-1.25.1/lib/libonnxruntime.so"

# <...>
```

This can be useful if
`snout-cli` crashes on exit due to an outdated system onnxruntime.

Precompiled releases for onnxruntime can be found on [its github](https://github.com/microsoft/onnxruntime/releases)

## Tracking

Libsnout comes with a working face tracking model. It's the same as in the baballonia repository, but ran through `onnxsim`.

Once you have set up your configuration file to point to your cameras, and set the output OSC destination to the correct values for your program of choice. You can start tracking with the following command:

```sh
snout-cli track
``` 

This will start recording, along with sending data to the OSC endpoint specified in the configuration file.

## Training an eye model
Eye models can be trained with the following command:

```sh
snout-cli train <user_cal.bin> <output.onnx>
``` 
the `<user_cal.bin>` file generated by baballonia can be found in the installation folder of the baballonia software. Next to the executable.
The resulting `<output.onnx>` can then be used in the configuration file, for the corresponding eye.

## Troubleshooting

A camera frame can be captured and written to a file with the following command to help with debugging tracking issues, along with aligning your face:  
```sh
snout-cli capture <SOURCE> <OUTPUT.jpeg>
``` 

`<SOURCE>` can be any of the following camera sources `left-eye`, `right-eye`, `face`,

`<OUTPUT.jpeg>` will be the name of the file that the camera frame gets written to.

## Notes on cropping

cropping the image works slightly differently; instead of providing top/left/right/bottom coordinates it uses major/minor shift and scale.
Scale 1 is 100%, increase it to zoom in (1.5 would be 150%). 
Major shift and minor shift range from -1 to 1.

Major shift shifts along the longest axis, minor shift shifts along the shortest axis. Minor shift only does something when zoomed in, if your input is a square then both will only function when zoomed in.

The camera stream will always be cropped into a square; so on a 16:9 image the sides are trimmed off along the longest axis, and Major shift will then allow you to shift the crop left or right. If you then zoom in on the cropped image, minor shift will allow you to shift the crop up or down.

It was designed this way to prevent users from squishing their face, since the model always wants a 240x240 pixel input and the image pipeline just squishes the cropped image to fit that, squishing your face if you don't have a perfectly square crop.

## License

Right now it's licensed under the same license as Baballonia from Project Babble is, considering this is a derivative work.
