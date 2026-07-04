# Configuring 


[Configuration file location](#configuration-file-location)

[Disabling specific tracking points](#disabling-specific-tracking-points)

[Finding your camera](#finding-your-camera)

[Rotating, flipping, and changing a cameras brightness](#rotating-flipping-and-changing-a-cameras-brightness)

[Cropping a camera](#cropping-a-camera)
- [Notes on cropping](#notes-on-cropping)

[Using with VRCFT or oscavmgr](#using-with-vrcft-or-oscavmgr)

[Using with VRC native Eye Tracking](#using-with-vrc-native-eye-tracking)

[Face calibration](#face-calibration)

[Filter Settings](#filter-settings)

[Note on onnx model paths](#note-on-onnx-model-paths)

[Using a non-system onnxruntime](#using-a-non-system-onnxruntime)

## Configuration file location

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

## Disabling specific tracking points

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

## Finding your camera

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

## Rotating, mirroring, and changing a cameras brightness

Changing the values for rotation and brightness, along with wether or not the camera is mirrored horizontally and/or vertically, can be achieved through the `<Tracking Point>.transform` tables.

For the face camera, this can be achieved like so:

```toml
[face.transform]
rotation = 90 #Rotate 90 degrees
brightness = 0.66 #Dim by 33%
vertical_flip = true # Mirrors camera vertically
horizontal_flip = false 
```

The brightness value is given as a percentage, where a value of 1 is 100% brightness (Original), and 0 is 0% brightness (Pitch black).
Values above 1 to increase the cameras brightness are allowed.

The value for rotation is given in whole degrees. Realistically you should only need 90, 180, and 270.

## Cropping a camera

Cropping a camera stream can be done through editing the values of the `<Tracking point>.crop` tables. For the face, this can be done like so

```toml
[face.crop]
scale = 1.2 #Zoom in 20%
major_shift = 0.0
minor_shift = 0.0
```

Before trying to crop your camera, make sure to read the [notes on cropping](#notes-on-cropping) below.

### Notes on cropping

cropping the image works slightly differently; instead of providing top/left/right/bottom coordinates it uses major/minor shift and scale.
Scale 1 is 100%, increase it to zoom in (1.5 would be 150%). 
Major shift and minor shift range from -1 to 1.

Major shift shifts along the longest axis, minor shift shifts along the shortest axis. Minor shift only does something when zoomed in, if your input is a square then both will only function when zoomed in.

The camera stream will always be cropped into a square; so on a 16:9 image the sides are trimmed off along the longest axis, and Major shift will then allow you to shift the crop left or right. If you then zoom in on the cropped image, minor shift will allow you to shift the crop up or down.

It was designed this way to prevent users from squishing their face, since the model always wants a 240x240 pixel input and the image pipeline just squishes the cropped image to fit that, squishing your face if you don't have a perfectly square crop.


## Using with VRCFT or oscavmgr

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

## Using with VRC native Eye Tracking

VRC offers a native eye tracking solution over osc that doesnt require a bridge like VRCFT or OscAvMgr, and works with any avatar.

It can be enabled by uncommenting the following lines in the configuration file, like so:

```toml
[output.vrchat]
destination = "127.0.0.1:9000"
max_pitch = 20.0 # Optional
max_yaw = 30.0 # Optional
```

if VRChat was set to use a different port than default for recieving OSC messages, changing the destination port is also required.

## Face calibration

Face calibration can be done by adjusting the upper and lower bounds of the different `[[face.calibration]]` tables in the configuration file, like so:

```toml
[[face.calibration]]
shape = "CheekPuffLeft"
lower = 0.3
upper = 0.6
```

The full list of shapes can be found in the template configuration file.

## Filter Settings

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

## Note on onnx model paths

The paths to the face and eye tracking onnx models are relative to the directory of the config file. An absolute path may be preferred and can be set by prefixing the path with a `/` like so:

```toml
[face]

# <...>

model = "/home/user/libsnout/faceModel.onnx" 
``` 

## Using a non-system onnxruntime

A libonnxruntime library file can be supplied in the configuration file through the `libonnxruntime` key, like so:

```toml
libonnxruntime = "onnxruntime-linux-x64-gpu-1.25.1/lib/libonnxruntime.so"

# <...>
```

This can be useful if
`snout-cli` crashes on exit due to an outdated system onnxruntime.

Precompiled releases for onnxruntime can be found on [its github](https://github.com/microsoft/onnxruntime/releases)

