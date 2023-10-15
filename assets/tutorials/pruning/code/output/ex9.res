Chain(
  Chain(
    Conv((3, 3), 3 => 8, pad=1, stride=2, bias=false),  # 216 parameters
    BatchNorm(8, slopehtanh),           # 16 parameters, plus 16
    Conv((3, 3), 8 => 8, pad=1, groups=8, bias=false),  # 72 parameters
    BatchNorm(8, slopehtanh),           # 16 parameters, plus 16
    Conv((1, 1), 8 => 12),              # 108 parameters
    BatchNorm(12, hardtanh),            # 24 parameters, plus 24
    Conv((3, 3), 12 => 12, pad=1, stride=2, groups=12, bias=false),  # 108 parameters
    BatchNorm(12, hardtanh),            # 24 parameters, plus 24
    Conv((1, 1), 12 => 22),             # 286 parameters
    BatchNorm(22, hardtanh),            # 44 parameters, plus 44
    Conv((3, 3), 22 => 22, pad=1, groups=22, bias=false),  # 198 parameters
    BatchNorm(22, hardtanh),            # 44 parameters, plus 44
    Conv((1, 1), 22 => 20),             # 460 parameters
    BatchNorm(20, hardtanh),            # 40 parameters, plus 40
    Conv((3, 3), 20 => 20, pad=1, stride=2, groups=20, bias=false),  # 180 parameters
    BatchNorm(20, hardtanh),            # 40 parameters, plus 40
    Conv((1, 1), 20 => 42),             # 882 parameters
    BatchNorm(42, hardtanh),            # 84 parameters, plus 84
    Conv((3, 3), 42 => 42, pad=1, groups=42, bias=false),  # 378 parameters
    BatchNorm(42, hardtanh),            # 84 parameters, plus 84
    Conv((1, 1), 42 => 40),             # 1_720 parameters
    BatchNorm(40, hardtanh),            # 80 parameters, plus 80
    Conv((3, 3), 40 => 40, pad=1, stride=2, groups=40, bias=false),  # 360 parameters
    BatchNorm(40, hardtanh),            # 80 parameters, plus 80
    Conv((1, 1), 40 => 84),             # 3_444 parameters
    BatchNorm(84, hardtanh),            # 168 parameters, plus 168
    Conv((3, 3), 84 => 84, pad=1, groups=84, bias=false),  # 756 parameters
    BatchNorm(84, hardtanh),            # 168 parameters, plus 168
    Conv((1, 1), 84 => 83),             # 7_055 parameters
    BatchNorm(83, hardtanh),            # 166 parameters, plus 166
    Conv((3, 3), 83 => 83, pad=1, groups=83, bias=false),  # 747 parameters
    BatchNorm(83, hardtanh),            # 166 parameters, plus 166
    Conv((1, 1), 83 => 84),             # 7_056 parameters
    BatchNorm(84, hardtanh),            # 168 parameters, plus 168
    Conv((3, 3), 84 => 84, pad=1, groups=84, bias=false),  # 756 parameters
    BatchNorm(84, hardtanh),            # 168 parameters, plus 168
    Conv((1, 1), 84 => 85),             # 7_225 parameters
    BatchNorm(85, hardtanh),            # 170 parameters, plus 170
    Conv((3, 3), 85 => 85, pad=1, groups=85, bias=false),  # 765 parameters
    BatchNorm(85, hardtanh),            # 170 parameters, plus 170
    Conv((1, 1), 85 => 84),             # 7_224 parameters
    BatchNorm(84, hardtanh),            # 168 parameters, plus 168
    Conv((3, 3), 84 => 84, pad=1, groups=84, bias=false),  # 756 parameters
    BatchNorm(84, hardtanh),            # 168 parameters, plus 168
    Conv((1, 1), 84 => 83),             # 7_055 parameters
    BatchNorm(83, hardtanh),            # 166 parameters, plus 166
    Conv((3, 3), 83 => 83, pad=1, stride=2, groups=83, bias=false),  # 747 parameters
    BatchNorm(83, hardtanh),            # 166 parameters, plus 166
    Conv((1, 1), 83 => 163),            # 13_692 parameters
    BatchNorm(163, hardtanh),           # 326 parameters, plus 326
    Conv((3, 3), 163 => 163, pad=1, groups=163, bias=false),  # 1_467 parameters
    BatchNorm(163, hardtanh),           # 326 parameters, plus 326
    Conv((1, 1), 163 => 256),           # 41_984 parameters
    BatchNorm(256, slopehtanh),         # 512 parameters, plus 512
  ),
  Chain(
    GlobalMeanPool(),
    MLUtils.flatten,
    Dense(256 => 64, slopehtanh),       # 16_448 parameters
    Dense(64 => 1),                     # 65 parameters
  ),
)         # Total: 98 trainable arrays, 125_962 parameters,
          # plus 54 non-trainable, 3_752 parameters, summarysize 527.133 KiB.