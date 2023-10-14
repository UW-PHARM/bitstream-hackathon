Chain(
  Chain(
    Conv((3, 3), 3 => 8, pad=1, stride=2, bias=false),  # 216 parameters
    BatchNorm(8, slopehtanh),           # 16 parameters, plus 16
    Conv((3, 3), 8 => 8, pad=1, groups=8, bias=false),  # 72 parameters
    BatchNorm(8, slopehtanh),           # 16 parameters, plus 16
    Conv((1, 1), 8 => 16),              # 144 parameters
    BatchNorm(16, hardtanh),            # 32 parameters, plus 32
    Conv((3, 3), 16 => 16, pad=1, stride=2, groups=16, bias=false),  # 144 parameters
    BatchNorm(16, hardtanh),            # 32 parameters, plus 32
    Conv((1, 1), 16 => 32),             # 544 parameters
    BatchNorm(32, hardtanh),            # 64 parameters, plus 64
    Conv((3, 3), 32 => 32, pad=1, groups=32, bias=false),  # 288 parameters
    BatchNorm(32, hardtanh),            # 64 parameters, plus 64
    Conv((1, 1), 32 => 32),             # 1_056 parameters
    BatchNorm(32, hardtanh),            # 64 parameters, plus 64
    Conv((3, 3), 32 => 32, pad=1, stride=2, groups=32, bias=false),  # 288 parameters
    BatchNorm(32, hardtanh),            # 64 parameters, plus 64
    Conv((1, 1), 32 => 64),             # 2_112 parameters
    BatchNorm(64, hardtanh),            # 128 parameters, plus 128
    Conv((3, 3), 64 => 64, pad=1, groups=64, bias=false),  # 576 parameters
    BatchNorm(64, hardtanh),            # 128 parameters, plus 128
    Conv((1, 1), 64 => 64),             # 4_160 parameters
    BatchNorm(64, hardtanh),            # 128 parameters, plus 128
    Conv((3, 3), 64 => 64, pad=1, stride=2, groups=64, bias=false),  # 576 parameters
    BatchNorm(64, hardtanh),            # 128 parameters, plus 128
    Conv((1, 1), 64 => 128),            # 8_320 parameters
    BatchNorm(128, hardtanh),           # 256 parameters, plus 256
    Conv((3, 3), 128 => 128, pad=1, groups=128, bias=false),  # 1_152 parameters
    BatchNorm(128, hardtanh),           # 256 parameters, plus 256
    Conv((1, 1), 128 => 128),           # 16_512 parameters
    BatchNorm(128, hardtanh),           # 256 parameters, plus 256
    Conv((3, 3), 128 => 128, pad=1, groups=128, bias=false),  # 1_152 parameters
    BatchNorm(128, hardtanh),           # 256 parameters, plus 256
    Conv((1, 1), 128 => 128),           # 16_512 parameters
    BatchNorm(128, hardtanh),           # 256 parameters, plus 256
    Conv((3, 3), 128 => 128, pad=1, groups=128, bias=false),  # 1_152 parameters
    BatchNorm(128, hardtanh),           # 256 parameters, plus 256
    Conv((1, 1), 128 => 128),           # 16_512 parameters
    BatchNorm(128, hardtanh),           # 256 parameters, plus 256
    Conv((3, 3), 128 => 128, pad=1, groups=128, bias=false),  # 1_152 parameters
    BatchNorm(128, hardtanh),           # 256 parameters, plus 256
    Conv((1, 1), 128 => 128),           # 16_512 parameters
    BatchNorm(128, hardtanh),           # 256 parameters, plus 256
    Conv((3, 3), 128 => 128, pad=1, groups=128, bias=false),  # 1_152 parameters
    BatchNorm(128, hardtanh),           # 256 parameters, plus 256
    Conv((1, 1), 128 => 128),           # 16_512 parameters
    BatchNorm(128, hardtanh),           # 256 parameters, plus 256
    Conv((3, 3), 128 => 128, pad=1, stride=2, groups=128, bias=false),  # 1_152 parameters
    BatchNorm(128, hardtanh),           # 256 parameters, plus 256
    Conv((1, 1), 128 => 256),           # 33_024 parameters
    BatchNorm(256, hardtanh),           # 512 parameters, plus 512
    Conv((3, 3), 256 => 256, pad=1, groups=256, bias=false),  # 2_304 parameters
    BatchNorm(256, hardtanh),           # 512 parameters, plus 512
    Conv((1, 1), 256 => 256),           # 65_792 parameters
    BatchNorm(256, slopehtanh),         # 512 parameters, plus 512
  ),
  Chain(
    GlobalMeanPool(),
    MLUtils.flatten,
    Dense(256 => 64, slopehtanh),       # 16_448 parameters
    Dense(64 => 2),                     # 130 parameters
  ),
)         # Total: 98 trainable arrays, 231_138 parameters,
          # plus 54 non-trainable, 5_472 parameters, summarysize 944.695 KiB.