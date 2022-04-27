Chain(
  Chain(
    Conv((3, 3), 3 => 8, relu, pad=1, stride=2),  # 224 parameters
    Conv((3, 3), 8 => 8, relu, pad=1, groups=8),  # 80 parameters
    Conv((1, 1), 8 => 16, relu),        # 144 parameters
    Conv((3, 3), 16 => 16, relu, pad=1, stride=2, groups=16),  # 160 parameters
    Conv((1, 1), 16 => 32, relu),       # 544 parameters
    Conv((3, 3), 32 => 32, relu, pad=1, groups=32),  # 320 parameters
    Conv((1, 1), 32 => 32, relu),       # 1_056 parameters
    Conv((3, 3), 32 => 32, relu, pad=1, stride=2, groups=32),  # 320 parameters
    Conv((1, 1), 32 => 64, relu),       # 2_112 parameters
    Conv((3, 3), 64 => 64, relu, pad=1, groups=64),  # 640 parameters
    Conv((1, 1), 64 => 64, relu),       # 4_160 parameters
    Conv((3, 3), 64 => 64, relu, pad=1, stride=2, groups=64),  # 640 parameters
    Conv((1, 1), 64 => 128, relu),      # 8_320 parameters
    Conv((3, 3), 128 => 128, relu, pad=1, groups=128),  # 1_280 parameters
    Conv((1, 1), 128 => 128, relu),     # 16_512 parameters
    Conv((3, 3), 128 => 128, relu, pad=1, groups=128),  # 1_280 parameters
    Conv((1, 1), 128 => 128, relu),     # 16_512 parameters
    Conv((3, 3), 128 => 128, relu, pad=1, groups=128),  # 1_280 parameters
    Conv((1, 1), 128 => 128, relu),     # 16_512 parameters
    Conv((3, 3), 128 => 128, relu, pad=1, groups=128),  # 1_280 parameters
    Conv((1, 1), 128 => 128, relu),     # 16_512 parameters
    Conv((3, 3), 128 => 128, relu, pad=1, groups=128),  # 1_280 parameters
    Conv((1, 1), 128 => 128, relu),     # 16_512 parameters
    Conv((3, 3), 128 => 128, relu, pad=1, stride=2, groups=128),  # 1_280 parameters
    Conv((1, 1), 128 => 256, relu),     # 33_024 parameters
    Conv((3, 3), 256 => 256, relu, pad=1, groups=256),  # 2_560 parameters
    Conv((1, 1), 256 => 256, relu),     # 65_792 parameters
  ),
  Chain(
    GlobalMeanPool(),
    MLUtils.flatten,
    Dense(256 => 64, relu),             # 16_448 parameters
    Dense(64 => 1),                     # 65 parameters
  ),
)                   # Total: 58 arrays, 226_849 parameters, 900.574 KiB.