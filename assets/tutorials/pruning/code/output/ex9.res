Chain(
  Chain(
    Conv((3, 3), 3 => 8, pad=1, stride=2, bias=false),  # 216 parameters
    BatchNorm(8, slopehtanh),           # 16 parameters, plus 16
    Conv((3, 3), 8 => 8, pad=1, groups=8, bias=false),  # 72 parameters
    BatchNorm(8, slopehtanh),           # 16 parameters, plus 16
    Conv((1, 1), 8 => 11),              # 99 parameters
    BatchNorm(11, slopehtanh),          # 22 parameters, plus 22
    Conv((3, 3), 11 => 11, pad=1, stride=2, groups=11, bias=false),  # 99 parameters
    BatchNorm(11, slopehtanh),          # 22 parameters, plus 22
    Conv((1, 1), 11 => 21),             # 252 parameters
    BatchNorm(21, slopehtanh),          # 42 parameters, plus 42
    Conv((3, 3), 21 => 21, pad=1, groups=21, bias=false),  # 189 parameters
    BatchNorm(21, slopehtanh),          # 42 parameters, plus 42
    Conv((1, 1), 21 => 20),             # 440 parameters
    BatchNorm(20, slopehtanh),          # 40 parameters, plus 40
    Conv((3, 3), 20 => 20, pad=1, stride=2, groups=20, bias=false),  # 180 parameters
    BatchNorm(20, slopehtanh),          # 40 parameters, plus 40
    Conv((1, 1), 20 => 44),             # 924 parameters
    BatchNorm(44, slopehtanh),          # 88 parameters, plus 88
    Conv((3, 3), 44 => 44, pad=1, groups=44, bias=false),  # 396 parameters
    BatchNorm(44, slopehtanh),          # 88 parameters, plus 88
    Conv((1, 1), 44 => 41),             # 1_845 parameters
    BatchNorm(41, slopehtanh),          # 82 parameters, plus 82
    Conv((3, 3), 41 => 41, pad=1, stride=2, groups=41, bias=false),  # 369 parameters
    BatchNorm(41, slopehtanh),          # 82 parameters, plus 82
    Conv((1, 1), 41 => 83),             # 3_486 parameters
    BatchNorm(83, slopehtanh),          # 166 parameters, plus 166
    Conv((3, 3), 83 => 83, pad=1, groups=83, bias=false),  # 747 parameters
    BatchNorm(83, slopehtanh),          # 166 parameters, plus 166
    Conv((1, 1), 83 => 85),             # 7_140 parameters
    BatchNorm(85, slopehtanh),          # 170 parameters, plus 170
    Conv((3, 3), 85 => 85, pad=1, groups=85, bias=false),  # 765 parameters
    BatchNorm(85, slopehtanh),          # 170 parameters, plus 170
    Conv((1, 1), 85 => 86),             # 7_396 parameters
    BatchNorm(86, slopehtanh),          # 172 parameters, plus 172
    Conv((3, 3), 86 => 86, pad=1, groups=86, bias=false),  # 774 parameters
    BatchNorm(86, slopehtanh),          # 172 parameters, plus 172
    Conv((1, 1), 86 => 82),             # 7_134 parameters
    BatchNorm(82, slopehtanh),          # 164 parameters, plus 164
    Conv((3, 3), 82 => 82, pad=1, groups=82, bias=false),  # 738 parameters
    BatchNorm(82, slopehtanh),          # 164 parameters, plus 164
    Conv((1, 1), 82 => 81),             # 6_723 parameters
    BatchNorm(81, slopehtanh),          # 162 parameters, plus 162
    Conv((3, 3), 81 => 81, pad=1, groups=81, bias=false),  # 729 parameters
    BatchNorm(81, slopehtanh),          # 162 parameters, plus 162
    Conv((1, 1), 81 => 82),             # 6_724 parameters
    BatchNorm(82, slopehtanh),          # 164 parameters, plus 164
    Conv((3, 3), 82 => 82, pad=1, stride=2, groups=82, bias=false),  # 738 parameters
    BatchNorm(82, slopehtanh),          # 164 parameters, plus 164
    Conv((1, 1), 82 => 162),            # 13_446 parameters
    BatchNorm(162, slopehtanh),         # 324 parameters, plus 324
    Conv((3, 3), 162 => 162, pad=1, groups=162, bias=false),  # 1_458 parameters
    BatchNorm(162, slopehtanh),         # 324 parameters, plus 324
    Conv((1, 1), 162 => 256),           # 41_728 parameters
    BatchNorm(256, slopehtanh),         # 512 parameters, plus 512
  ),
  Chain(
    GlobalMeanPool(),
    MLUtils.flatten,
    Dense(256 => 64, slopehtanh),       # 16_448 parameters
    Dense(64 => 1),                     # 65 parameters
  ),
)         # Total: 98 trainable arrays, 125_056 parameters,
          # plus 54 non-trainable, 3_736 parameters, summarysize 523.531 KiB.