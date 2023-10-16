Chain(
  Chain(
    Conv((3, 3), 3 => 8, pad=1, stride=2, bias=false),  # 216 parameters
    BatchNorm(8, slopehtanh),           # 16 parameters, plus 16
    Conv((3, 3), 8 => 8, pad=1, groups=8, bias=false),  # 72 parameters
    BatchNorm(8, slopehtanh),           # 16 parameters, plus 16
    Conv((1, 1), 8 => 11),              # 99 parameters
    BatchNorm(11, hardtanh),            # 22 parameters, plus 22
    Conv((3, 3), 11 => 11, pad=1, stride=2, groups=11, bias=false),  # 99 parameters
    BatchNorm(11, hardtanh),            # 22 parameters, plus 22
    Conv((1, 1), 11 => 20),             # 240 parameters
    BatchNorm(20, hardtanh),            # 40 parameters, plus 40
    Conv((3, 3), 20 => 20, pad=1, groups=20, bias=false),  # 180 parameters
    BatchNorm(20, hardtanh),            # 40 parameters, plus 40
    Conv((1, 1), 20 => 20),             # 420 parameters
    BatchNorm(20, hardtanh),            # 40 parameters, plus 40
    Conv((3, 3), 20 => 20, pad=1, stride=2, groups=20, bias=false),  # 180 parameters
    BatchNorm(20, hardtanh),            # 40 parameters, plus 40
    Conv((1, 1), 20 => 41),             # 861 parameters
    BatchNorm(41, hardtanh),            # 82 parameters, plus 82
    Conv((3, 3), 41 => 41, pad=1, groups=41, bias=false),  # 369 parameters
    BatchNorm(41, hardtanh),            # 82 parameters, plus 82
    Conv((1, 1), 41 => 43),             # 1_806 parameters
    BatchNorm(43, hardtanh),            # 86 parameters, plus 86
    Conv((3, 3), 43 => 43, pad=1, stride=2, groups=43, bias=false),  # 387 parameters
    BatchNorm(43, hardtanh),            # 86 parameters, plus 86
    Conv((1, 1), 43 => 84),             # 3_696 parameters
    BatchNorm(84, hardtanh),            # 168 parameters, plus 168
    Conv((3, 3), 84 => 84, pad=1, groups=84, bias=false),  # 756 parameters
    BatchNorm(84, hardtanh),            # 168 parameters, plus 168
    Conv((1, 1), 84 => 82),             # 6_970 parameters
    BatchNorm(82, hardtanh),            # 164 parameters, plus 164
    Conv((3, 3), 82 => 82, pad=1, groups=82, bias=false),  # 738 parameters
    BatchNorm(82, hardtanh),            # 164 parameters, plus 164
    Conv((1, 1), 82 => 83),             # 6_889 parameters
    BatchNorm(83, hardtanh),            # 166 parameters, plus 166
    Conv((3, 3), 83 => 83, pad=1, groups=83, bias=false),  # 747 parameters
    BatchNorm(83, hardtanh),            # 166 parameters, plus 166
    Conv((1, 1), 83 => 83),             # 6_972 parameters
    BatchNorm(83, hardtanh),            # 166 parameters, plus 166
    Conv((3, 3), 83 => 83, pad=1, groups=83, bias=false),  # 747 parameters
    BatchNorm(83, hardtanh),            # 166 parameters, plus 166
    Conv((1, 1), 83 => 81),             # 6_804 parameters
    BatchNorm(81, hardtanh),            # 162 parameters, plus 162
    Conv((3, 3), 81 => 81, pad=1, groups=81, bias=false),  # 729 parameters
    BatchNorm(81, hardtanh),            # 162 parameters, plus 162
    Conv((1, 1), 81 => 84),             # 6_888 parameters
    BatchNorm(84, hardtanh),            # 168 parameters, plus 168
    Conv((3, 3), 84 => 84, pad=1, stride=2, groups=84, bias=false),  # 756 parameters
    BatchNorm(84, hardtanh),            # 168 parameters, plus 168
    Conv((1, 1), 84 => 167),            # 14_195 parameters
    BatchNorm(167, hardtanh),           # 334 parameters, plus 334
    Conv((3, 3), 167 => 167, pad=1, groups=167, bias=false),  # 1_503 parameters
    BatchNorm(167, hardtanh),           # 334 parameters, plus 334
    Conv((1, 1), 167 => 256),           # 43_008 parameters
    BatchNorm(256, slopehtanh),         # 512 parameters, plus 512
  ),
  Chain(
    GlobalMeanPool(),
    MLUtils.flatten,
    Dense(256 => 64, slopehtanh),       # 16_448 parameters
    Dense(64 => 1),                     # 65 parameters
  ),
)         # Total: 98 trainable arrays, 126_580 parameters,
          # plus 54 non-trainable, 3_740 parameters, summarysize 529.500 KiB.