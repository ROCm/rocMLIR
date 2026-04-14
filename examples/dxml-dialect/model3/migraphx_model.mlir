module {
  func.func @torch_jit(%arg0: !migraphx.shaped<1x3x300x300xf16, 270000x90000x300x1>) -> (!migraphx.shaped<1x4212x2xf16, 8424x2x1>, !migraphx.shaped<1x4212x4xf16, 16848x4x1>) attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "1.10"} {
    %0 = migraphx.literal(dense<0.000000e+00> : tensor<1x4212x2xf16>) : <1x4212x2xf16, 8424x2x1>
    %1 = migraphx.literal(dense<0.000000e+00> : tensor<1xf16>) : <1xf16, 1>
    %2 = migraphx.literal(dense<0.000000e+00> : tensor<1x4212x2xf16>) : <1x4212x2xf16, 8424x2x1>
    %3 = migraphx.literal(dense<0.000000e+00> : tensor<1x4212x2xf16>) : <1x4212x2xf16, 8424x2x1>
    %4 = migraphx.literal(dense<0.000000e+00> : tensor<1xf16>) : <1xf16, 1>
    %5 = migraphx.literal(dense<0.000000e+00> : tensor<12x576x1x1xf16>) : <12x576x1x1xf16, 576x1x1x1>
    %6 = migraphx.literal(dense<0.000000e+00> : tensor<12xf16>) : <12xf16, 1>
    %7 = migraphx.literal(dense<0.000000e+00> : tensor<24x1280x1x1xf16>) : <24x1280x1x1xf16, 1280x1x1x1>
    %8 = migraphx.literal(dense<0.000000e+00> : tensor<24xf16>) : <24xf16, 1>
    %9 = migraphx.literal(dense<0.000000e+00> : tensor<48x512x1x1xf16>) : <48x512x1x1xf16, 512x1x1x1>
    %10 = migraphx.literal(dense<0.000000e+00> : tensor<48xf16>) : <48xf16, 1>
    %11 = migraphx.literal(dense<0.000000e+00> : tensor<48x256x1x1xf16>) : <48x256x1x1xf16, 256x1x1x1>
    %12 = migraphx.literal(dense<0.000000e+00> : tensor<48xf16>) : <48xf16, 1>
    %13 = migraphx.literal(dense<0.000000e+00> : tensor<12x256x1x1xf16>) : <12x256x1x1xf16, 256x1x1x1>
    %14 = migraphx.literal(dense<0.000000e+00> : tensor<12xf16>) : <12xf16, 1>
    %15 = migraphx.literal(dense<0.000000e+00> : tensor<12x64x1x1xf16>) : <12x64x1x1xf16, 64x1x1x1>
    %16 = migraphx.literal(dense<0.000000e+00> : tensor<12xf16>) : <12xf16, 1>
    %17 = migraphx.literal(dense<0.000000e+00> : tensor<24x576x1x1xf16>) : <24x576x1x1xf16, 576x1x1x1>
    %18 = migraphx.literal(dense<0.000000e+00> : tensor<24xf16>) : <24xf16, 1>
    %19 = migraphx.literal(dense<0.000000e+00> : tensor<48x1280x1x1xf16>) : <48x1280x1x1xf16, 1280x1x1x1>
    %20 = migraphx.literal(dense<0.000000e+00> : tensor<48xf16>) : <48xf16, 1>
    %21 = migraphx.literal(dense<0.000000e+00> : tensor<96x512x1x1xf16>) : <96x512x1x1xf16, 512x1x1x1>
    %22 = migraphx.literal(dense<0.000000e+00> : tensor<96xf16>) : <96xf16, 1>
    %23 = migraphx.literal(dense<0.000000e+00> : tensor<96x256x1x1xf16>) : <96x256x1x1xf16, 256x1x1x1>
    %24 = migraphx.literal(dense<0.000000e+00> : tensor<96xf16>) : <96xf16, 1>
    %25 = migraphx.literal(dense<0.000000e+00> : tensor<24x256x1x1xf16>) : <24x256x1x1xf16, 256x1x1x1>
    %26 = migraphx.literal(dense<0.000000e+00> : tensor<24xf16>) : <24xf16, 1>
    %27 = migraphx.literal(dense<0.000000e+00> : tensor<24x64x1x1xf16>) : <24x64x1x1xf16, 64x1x1x1>
    %28 = migraphx.literal(dense<0.000000e+00> : tensor<24xf16>) : <24xf16, 1>
    %29 = migraphx.literal(dense<0.000000e+00> : tensor<32x3x3x3xf16>) : <32x3x3x3xf16, 27x9x3x1>
    %30 = migraphx.literal(dense<0.000000e+00> : tensor<32xf16>) : <32xf16, 1>
    %31 = migraphx.literal(dense<0.000000e+00> : tensor<32x1x3x3xf16>) : <32x1x3x3xf16, 9x9x3x1>
    %32 = migraphx.literal(dense<0.000000e+00> : tensor<32xf16>) : <32xf16, 1>
    %33 = migraphx.literal(dense<0.000000e+00> : tensor<16x32x1x1xf16>) : <16x32x1x1xf16, 32x1x1x1>
    %34 = migraphx.literal(dense<0.000000e+00> : tensor<16xf16>) : <16xf16, 1>
    %35 = migraphx.literal(dense<0.000000e+00> : tensor<96x16x1x1xf16>) : <96x16x1x1xf16, 16x1x1x1>
    %36 = migraphx.literal(dense<0.000000e+00> : tensor<96xf16>) : <96xf16, 1>
    %37 = migraphx.literal(dense<0.000000e+00> : tensor<96x1x3x3xf16>) : <96x1x3x3xf16, 9x9x3x1>
    %38 = migraphx.literal(dense<0.000000e+00> : tensor<96xf16>) : <96xf16, 1>
    %39 = migraphx.literal(dense<0.000000e+00> : tensor<24x96x1x1xf16>) : <24x96x1x1xf16, 96x1x1x1>
    %40 = migraphx.literal(dense<0.000000e+00> : tensor<24xf16>) : <24xf16, 1>
    %41 = migraphx.literal(dense<0.000000e+00> : tensor<144x24x1x1xf16>) : <144x24x1x1xf16, 24x1x1x1>
    %42 = migraphx.literal(dense<0.000000e+00> : tensor<144xf16>) : <144xf16, 1>
    %43 = migraphx.literal(dense<0.000000e+00> : tensor<144x1x3x3xf16>) : <144x1x3x3xf16, 9x9x3x1>
    %44 = migraphx.literal(dense<0.000000e+00> : tensor<144xf16>) : <144xf16, 1>
    %45 = migraphx.literal(dense<0.000000e+00> : tensor<24x144x1x1xf16>) : <24x144x1x1xf16, 144x1x1x1>
    %46 = migraphx.literal(dense<0.000000e+00> : tensor<24xf16>) : <24xf16, 1>
    %47 = migraphx.literal(dense<0.000000e+00> : tensor<144x24x1x1xf16>) : <144x24x1x1xf16, 24x1x1x1>
    %48 = migraphx.literal(dense<0.000000e+00> : tensor<144xf16>) : <144xf16, 1>
    %49 = migraphx.literal(dense<0.000000e+00> : tensor<144x1x3x3xf16>) : <144x1x3x3xf16, 9x9x3x1>
    %50 = migraphx.literal(dense<0.000000e+00> : tensor<144xf16>) : <144xf16, 1>
    %51 = migraphx.literal(dense<0.000000e+00> : tensor<32x144x1x1xf16>) : <32x144x1x1xf16, 144x1x1x1>
    %52 = migraphx.literal(dense<0.000000e+00> : tensor<32xf16>) : <32xf16, 1>
    %53 = migraphx.literal(dense<0.000000e+00> : tensor<192x32x1x1xf16>) : <192x32x1x1xf16, 32x1x1x1>
    %54 = migraphx.literal(dense<0.000000e+00> : tensor<192xf16>) : <192xf16, 1>
    %55 = migraphx.literal(dense<0.000000e+00> : tensor<192x1x3x3xf16>) : <192x1x3x3xf16, 9x9x3x1>
    %56 = migraphx.literal(dense<0.000000e+00> : tensor<192xf16>) : <192xf16, 1>
    %57 = migraphx.literal(dense<0.000000e+00> : tensor<32x192x1x1xf16>) : <32x192x1x1xf16, 192x1x1x1>
    %58 = migraphx.literal(dense<0.000000e+00> : tensor<32xf16>) : <32xf16, 1>
    %59 = migraphx.literal(dense<0.000000e+00> : tensor<192x32x1x1xf16>) : <192x32x1x1xf16, 32x1x1x1>
    %60 = migraphx.literal(dense<0.000000e+00> : tensor<192xf16>) : <192xf16, 1>
    %61 = migraphx.literal(dense<0.000000e+00> : tensor<192x1x3x3xf16>) : <192x1x3x3xf16, 9x9x3x1>
    %62 = migraphx.literal(dense<0.000000e+00> : tensor<192xf16>) : <192xf16, 1>
    %63 = migraphx.literal(dense<0.000000e+00> : tensor<32x192x1x1xf16>) : <32x192x1x1xf16, 192x1x1x1>
    %64 = migraphx.literal(dense<0.000000e+00> : tensor<32xf16>) : <32xf16, 1>
    %65 = migraphx.literal(dense<0.000000e+00> : tensor<192x32x1x1xf16>) : <192x32x1x1xf16, 32x1x1x1>
    %66 = migraphx.literal(dense<0.000000e+00> : tensor<192xf16>) : <192xf16, 1>
    %67 = migraphx.literal(dense<0.000000e+00> : tensor<192x1x3x3xf16>) : <192x1x3x3xf16, 9x9x3x1>
    %68 = migraphx.literal(dense<0.000000e+00> : tensor<192xf16>) : <192xf16, 1>
    %69 = migraphx.literal(dense<0.000000e+00> : tensor<64x192x1x1xf16>) : <64x192x1x1xf16, 192x1x1x1>
    %70 = migraphx.literal(dense<0.000000e+00> : tensor<64xf16>) : <64xf16, 1>
    %71 = migraphx.literal(dense<0.000000e+00> : tensor<384x64x1x1xf16>) : <384x64x1x1xf16, 64x1x1x1>
    %72 = migraphx.literal(dense<0.000000e+00> : tensor<384xf16>) : <384xf16, 1>
    %73 = migraphx.literal(dense<0.000000e+00> : tensor<384x1x3x3xf16>) : <384x1x3x3xf16, 9x9x3x1>
    %74 = migraphx.literal(dense<0.000000e+00> : tensor<384xf16>) : <384xf16, 1>
    %75 = migraphx.literal(dense<0.000000e+00> : tensor<64x384x1x1xf16>) : <64x384x1x1xf16, 384x1x1x1>
    %76 = migraphx.literal(dense<0.000000e+00> : tensor<64xf16>) : <64xf16, 1>
    %77 = migraphx.literal(dense<0.000000e+00> : tensor<384x64x1x1xf16>) : <384x64x1x1xf16, 64x1x1x1>
    %78 = migraphx.literal(dense<0.000000e+00> : tensor<384xf16>) : <384xf16, 1>
    %79 = migraphx.literal(dense<0.000000e+00> : tensor<384x1x3x3xf16>) : <384x1x3x3xf16, 9x9x3x1>
    %80 = migraphx.literal(dense<0.000000e+00> : tensor<384xf16>) : <384xf16, 1>
    %81 = migraphx.literal(dense<0.000000e+00> : tensor<64x384x1x1xf16>) : <64x384x1x1xf16, 384x1x1x1>
    %82 = migraphx.literal(dense<0.000000e+00> : tensor<64xf16>) : <64xf16, 1>
    %83 = migraphx.literal(dense<0.000000e+00> : tensor<384x64x1x1xf16>) : <384x64x1x1xf16, 64x1x1x1>
    %84 = migraphx.literal(dense<0.000000e+00> : tensor<384xf16>) : <384xf16, 1>
    %85 = migraphx.literal(dense<0.000000e+00> : tensor<384x1x3x3xf16>) : <384x1x3x3xf16, 9x9x3x1>
    %86 = migraphx.literal(dense<0.000000e+00> : tensor<384xf16>) : <384xf16, 1>
    %87 = migraphx.literal(dense<0.000000e+00> : tensor<64x384x1x1xf16>) : <64x384x1x1xf16, 384x1x1x1>
    %88 = migraphx.literal(dense<0.000000e+00> : tensor<64xf16>) : <64xf16, 1>
    %89 = migraphx.literal(dense<0.000000e+00> : tensor<384x64x1x1xf16>) : <384x64x1x1xf16, 64x1x1x1>
    %90 = migraphx.literal(dense<0.000000e+00> : tensor<384xf16>) : <384xf16, 1>
    %91 = migraphx.literal(dense<0.000000e+00> : tensor<384x1x3x3xf16>) : <384x1x3x3xf16, 9x9x3x1>
    %92 = migraphx.literal(dense<0.000000e+00> : tensor<384xf16>) : <384xf16, 1>
    %93 = migraphx.literal(dense<0.000000e+00> : tensor<96x384x1x1xf16>) : <96x384x1x1xf16, 384x1x1x1>
    %94 = migraphx.literal(dense<0.000000e+00> : tensor<96xf16>) : <96xf16, 1>
    %95 = migraphx.literal(dense<0.000000e+00> : tensor<576x96x1x1xf16>) : <576x96x1x1xf16, 96x1x1x1>
    %96 = migraphx.literal(dense<0.000000e+00> : tensor<576xf16>) : <576xf16, 1>
    %97 = migraphx.literal(dense<0.000000e+00> : tensor<576x1x3x3xf16>) : <576x1x3x3xf16, 9x9x3x1>
    %98 = migraphx.literal(dense<0.000000e+00> : tensor<576xf16>) : <576xf16, 1>
    %99 = migraphx.literal(dense<0.000000e+00> : tensor<96x576x1x1xf16>) : <96x576x1x1xf16, 576x1x1x1>
    %100 = migraphx.literal(dense<0.000000e+00> : tensor<96xf16>) : <96xf16, 1>
    %101 = migraphx.literal(dense<0.000000e+00> : tensor<576x96x1x1xf16>) : <576x96x1x1xf16, 96x1x1x1>
    %102 = migraphx.literal(dense<0.000000e+00> : tensor<576xf16>) : <576xf16, 1>
    %103 = migraphx.literal(dense<0.000000e+00> : tensor<576x1x3x3xf16>) : <576x1x3x3xf16, 9x9x3x1>
    %104 = migraphx.literal(dense<0.000000e+00> : tensor<576xf16>) : <576xf16, 1>
    %105 = migraphx.literal(dense<0.000000e+00> : tensor<96x576x1x1xf16>) : <96x576x1x1xf16, 576x1x1x1>
    %106 = migraphx.literal(dense<0.000000e+00> : tensor<96xf16>) : <96xf16, 1>
    %107 = migraphx.literal(dense<0.000000e+00> : tensor<576x96x1x1xf16>) : <576x96x1x1xf16, 96x1x1x1>
    %108 = migraphx.literal(dense<0.000000e+00> : tensor<576xf16>) : <576xf16, 1>
    %109 = migraphx.literal(dense<0.000000e+00> : tensor<576x1x3x3xf16>) : <576x1x3x3xf16, 9x9x3x1>
    %110 = migraphx.literal(dense<0.000000e+00> : tensor<576xf16>) : <576xf16, 1>
    %111 = migraphx.literal(dense<0.000000e+00> : tensor<160x576x1x1xf16>) : <160x576x1x1xf16, 576x1x1x1>
    %112 = migraphx.literal(dense<0.000000e+00> : tensor<160xf16>) : <160xf16, 1>
    %113 = migraphx.literal(dense<0.000000e+00> : tensor<576x1x3x3xf16>) : <576x1x3x3xf16, 9x9x3x1>
    %114 = migraphx.literal(dense<0.000000e+00> : tensor<576xf16>) : <576xf16, 1>
    %115 = migraphx.literal(dense<0.000000e+00> : tensor<576x1x3x3xf16>) : <576x1x3x3xf16, 9x9x3x1>
    %116 = migraphx.literal(dense<0.000000e+00> : tensor<576xf16>) : <576xf16, 1>
    %117 = migraphx.literal(dense<0.000000e+00> : tensor<960x160x1x1xf16>) : <960x160x1x1xf16, 160x1x1x1>
    %118 = migraphx.literal(dense<0.000000e+00> : tensor<960xf16>) : <960xf16, 1>
    %119 = migraphx.literal(dense<0.000000e+00> : tensor<960x1x3x3xf16>) : <960x1x3x3xf16, 9x9x3x1>
    %120 = migraphx.literal(dense<0.000000e+00> : tensor<960xf16>) : <960xf16, 1>
    %121 = migraphx.literal(dense<0.000000e+00> : tensor<160x960x1x1xf16>) : <160x960x1x1xf16, 960x1x1x1>
    %122 = migraphx.literal(dense<0.000000e+00> : tensor<160xf16>) : <160xf16, 1>
    %123 = migraphx.literal(dense<0.000000e+00> : tensor<960x160x1x1xf16>) : <960x160x1x1xf16, 160x1x1x1>
    %124 = migraphx.literal(dense<0.000000e+00> : tensor<960xf16>) : <960xf16, 1>
    %125 = migraphx.literal(dense<0.000000e+00> : tensor<960x1x3x3xf16>) : <960x1x3x3xf16, 9x9x3x1>
    %126 = migraphx.literal(dense<0.000000e+00> : tensor<960xf16>) : <960xf16, 1>
    %127 = migraphx.literal(dense<0.000000e+00> : tensor<160x960x1x1xf16>) : <160x960x1x1xf16, 960x1x1x1>
    %128 = migraphx.literal(dense<0.000000e+00> : tensor<160xf16>) : <160xf16, 1>
    %129 = migraphx.literal(dense<0.000000e+00> : tensor<960x160x1x1xf16>) : <960x160x1x1xf16, 160x1x1x1>
    %130 = migraphx.literal(dense<0.000000e+00> : tensor<960xf16>) : <960xf16, 1>
    %131 = migraphx.literal(dense<0.000000e+00> : tensor<960x1x3x3xf16>) : <960x1x3x3xf16, 9x9x3x1>
    %132 = migraphx.literal(dense<0.000000e+00> : tensor<960xf16>) : <960xf16, 1>
    %133 = migraphx.literal(dense<0.000000e+00> : tensor<320x960x1x1xf16>) : <320x960x1x1xf16, 960x1x1x1>
    %134 = migraphx.literal(dense<0.000000e+00> : tensor<320xf16>) : <320xf16, 1>
    %135 = migraphx.literal(dense<0.000000e+00> : tensor<1280x320x1x1xf16>) : <1280x320x1x1xf16, 320x1x1x1>
    %136 = migraphx.literal(dense<0.000000e+00> : tensor<1280xf16>) : <1280xf16, 1>
    %137 = migraphx.literal(dense<0.000000e+00> : tensor<1280x1x3x3xf16>) : <1280x1x3x3xf16, 9x9x3x1>
    %138 = migraphx.literal(dense<0.000000e+00> : tensor<1280xf16>) : <1280xf16, 1>
    %139 = migraphx.literal(dense<0.000000e+00> : tensor<1280x1x3x3xf16>) : <1280x1x3x3xf16, 9x9x3x1>
    %140 = migraphx.literal(dense<0.000000e+00> : tensor<1280xf16>) : <1280xf16, 1>
    %141 = migraphx.literal(dense<0.000000e+00> : tensor<256x1280x1x1xf16>) : <256x1280x1x1xf16, 1280x1x1x1>
    %142 = migraphx.literal(dense<0.000000e+00> : tensor<256xf16>) : <256xf16, 1>
    %143 = migraphx.literal(dense<0.000000e+00> : tensor<256x1x3x3xf16>) : <256x1x3x3xf16, 9x9x3x1>
    %144 = migraphx.literal(dense<0.000000e+00> : tensor<256xf16>) : <256xf16, 1>
    %145 = migraphx.literal(dense<0.000000e+00> : tensor<512x256x1x1xf16>) : <512x256x1x1xf16, 256x1x1x1>
    %146 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %147 = migraphx.literal(dense<0.000000e+00> : tensor<512x1x3x3xf16>) : <512x1x3x3xf16, 9x9x3x1>
    %148 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %149 = migraphx.literal(dense<0.000000e+00> : tensor<512x1x3x3xf16>) : <512x1x3x3xf16, 9x9x3x1>
    %150 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %151 = migraphx.literal(dense<0.000000e+00> : tensor<128x512x1x1xf16>) : <128x512x1x1xf16, 512x1x1x1>
    %152 = migraphx.literal(dense<0.000000e+00> : tensor<128xf16>) : <128xf16, 1>
    %153 = migraphx.literal(dense<0.000000e+00> : tensor<128x1x3x3xf16>) : <128x1x3x3xf16, 9x9x3x1>
    %154 = migraphx.literal(dense<0.000000e+00> : tensor<128xf16>) : <128xf16, 1>
    %155 = migraphx.literal(dense<0.000000e+00> : tensor<256x128x1x1xf16>) : <256x128x1x1xf16, 128x1x1x1>
    %156 = migraphx.literal(dense<0.000000e+00> : tensor<256xf16>) : <256xf16, 1>
    %157 = migraphx.literal(dense<0.000000e+00> : tensor<256x1x3x3xf16>) : <256x1x3x3xf16, 9x9x3x1>
    %158 = migraphx.literal(dense<0.000000e+00> : tensor<256xf16>) : <256xf16, 1>
    %159 = migraphx.literal(dense<0.000000e+00> : tensor<256x1x3x3xf16>) : <256x1x3x3xf16, 9x9x3x1>
    %160 = migraphx.literal(dense<0.000000e+00> : tensor<256xf16>) : <256xf16, 1>
    %161 = migraphx.literal(dense<0.000000e+00> : tensor<128x256x1x1xf16>) : <128x256x1x1xf16, 256x1x1x1>
    %162 = migraphx.literal(dense<0.000000e+00> : tensor<128xf16>) : <128xf16, 1>
    %163 = migraphx.literal(dense<0.000000e+00> : tensor<128x1x3x3xf16>) : <128x1x3x3xf16, 9x9x3x1>
    %164 = migraphx.literal(dense<0.000000e+00> : tensor<128xf16>) : <128xf16, 1>
    %165 = migraphx.literal(dense<0.000000e+00> : tensor<256x128x1x1xf16>) : <256x128x1x1xf16, 128x1x1x1>
    %166 = migraphx.literal(dense<0.000000e+00> : tensor<256xf16>) : <256xf16, 1>
    %167 = migraphx.literal(dense<0.000000e+00> : tensor<256x1x3x3xf16>) : <256x1x3x3xf16, 9x9x3x1>
    %168 = migraphx.literal(dense<0.000000e+00> : tensor<256xf16>) : <256xf16, 1>
    %169 = migraphx.literal(dense<0.000000e+00> : tensor<256x1x3x3xf16>) : <256x1x3x3xf16, 9x9x3x1>
    %170 = migraphx.literal(dense<0.000000e+00> : tensor<256xf16>) : <256xf16, 1>
    %171 = migraphx.literal(dense<0.000000e+00> : tensor<64x256x1x1xf16>) : <64x256x1x1xf16, 256x1x1x1>
    %172 = migraphx.literal(dense<0.000000e+00> : tensor<64xf16>) : <64xf16, 1>
    %173 = migraphx.literal(dense<0.000000e+00> : tensor<64x1x3x3xf16>) : <64x1x3x3xf16, 9x9x3x1>
    %174 = migraphx.literal(dense<0.000000e+00> : tensor<64xf16>) : <64xf16, 1>
    %175 = migraphx.literal(dense<0.000000e+00> : tensor<64x64x1x1xf16>) : <64x64x1x1xf16, 64x1x1x1>
    %176 = migraphx.literal(dense<0.000000e+00> : tensor<64xf16>) : <64xf16, 1>
    %177 = migraphx.literal(dense<0.000000e+00> : tensor<1xf16>) : <1xf16, 1>
    %178 = migraphx.literal(dense<0.000000e+00> : tensor<1xf16>) : <1xf16, 1>
    %179 = migraphx.convolution %arg0, %29 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], stride = [2, 2]} : <1x3x300x300xf16, 270000x90000x300x1>, <32x3x3x3xf16, 27x9x3x1> -> <1x32x150x150xf16, 720000x22500x150x1>
    %180 = migraphx.multibroadcast %30 {out_lens = [1, 32, 150, 150]} : <32xf16, 1> -> <1x32x150x150xf16, 0x1x0x0>
    %181 = migraphx.add %179, %180 : <1x32x150x150xf16, 720000x22500x150x1>, <1x32x150x150xf16, 0x1x0x0> -> <1x32x150x150xf16, 720000x22500x150x1>
    %182 = builtin.unrealized_conversion_cast %181 : !migraphx.shaped<1x32x150x150xf16, 720000x22500x150x1> to !dxgml.tensor<1x32x150x150x!dxgml.float16>
    %183 = dxgml_op.clip (%182) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x32x150x150x!dxgml.float16>) -> !dxgml.tensor<1x32x150x150x!dxgml.float16>
    %184 = builtin.unrealized_conversion_cast %183 : !dxgml.tensor<1x32x150x150x!dxgml.float16> to !migraphx.shaped<1x32x150x150xf16, 720000x22500x150x1>
    %185 = migraphx.convolution %184, %31 {dilation = [1, 1], group = 32 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x32x150x150xf16, 720000x22500x150x1>, <32x1x3x3xf16, 9x9x3x1> -> <1x32x150x150xf16, 720000x22500x150x1>
    %186 = migraphx.multibroadcast %32 {out_lens = [1, 32, 150, 150]} : <32xf16, 1> -> <1x32x150x150xf16, 0x1x0x0>
    %187 = migraphx.add %185, %186 : <1x32x150x150xf16, 720000x22500x150x1>, <1x32x150x150xf16, 0x1x0x0> -> <1x32x150x150xf16, 720000x22500x150x1>
    %188 = builtin.unrealized_conversion_cast %187 : !migraphx.shaped<1x32x150x150xf16, 720000x22500x150x1> to !dxgml.tensor<1x32x150x150x!dxgml.float16>
    %189 = dxgml_op.clip (%188) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x32x150x150x!dxgml.float16>) -> !dxgml.tensor<1x32x150x150x!dxgml.float16>
    %190 = builtin.unrealized_conversion_cast %189 : !dxgml.tensor<1x32x150x150x!dxgml.float16> to !migraphx.shaped<1x32x150x150xf16, 720000x22500x150x1>
    %191 = migraphx.convolution %190, %33 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x32x150x150xf16, 720000x22500x150x1>, <16x32x1x1xf16, 32x1x1x1> -> <1x16x150x150xf16, 360000x22500x150x1>
    %192 = migraphx.multibroadcast %34 {out_lens = [1, 16, 150, 150]} : <16xf16, 1> -> <1x16x150x150xf16, 0x1x0x0>
    %193 = migraphx.add %191, %192 : <1x16x150x150xf16, 360000x22500x150x1>, <1x16x150x150xf16, 0x1x0x0> -> <1x16x150x150xf16, 360000x22500x150x1>
    %194 = migraphx.convolution %193, %35 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x16x150x150xf16, 360000x22500x150x1>, <96x16x1x1xf16, 16x1x1x1> -> <1x96x150x150xf16, 2160000x22500x150x1>
    %195 = migraphx.multibroadcast %36 {out_lens = [1, 96, 150, 150]} : <96xf16, 1> -> <1x96x150x150xf16, 0x1x0x0>
    %196 = migraphx.add %194, %195 : <1x96x150x150xf16, 2160000x22500x150x1>, <1x96x150x150xf16, 0x1x0x0> -> <1x96x150x150xf16, 2160000x22500x150x1>
    %197 = builtin.unrealized_conversion_cast %196 : !migraphx.shaped<1x96x150x150xf16, 2160000x22500x150x1> to !dxgml.tensor<1x96x150x150x!dxgml.float16>
    %198 = dxgml_op.clip (%197) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x96x150x150x!dxgml.float16>) -> !dxgml.tensor<1x96x150x150x!dxgml.float16>
    %199 = builtin.unrealized_conversion_cast %198 : !dxgml.tensor<1x96x150x150x!dxgml.float16> to !migraphx.shaped<1x96x150x150xf16, 2160000x22500x150x1>
    %200 = migraphx.convolution %199, %37 {dilation = [1, 1], group = 96 : i64, padding = [1, 1, 1, 1], stride = [2, 2]} : <1x96x150x150xf16, 2160000x22500x150x1>, <96x1x3x3xf16, 9x9x3x1> -> <1x96x75x75xf16, 540000x5625x75x1>
    %201 = migraphx.multibroadcast %38 {out_lens = [1, 96, 75, 75]} : <96xf16, 1> -> <1x96x75x75xf16, 0x1x0x0>
    %202 = migraphx.add %200, %201 : <1x96x75x75xf16, 540000x5625x75x1>, <1x96x75x75xf16, 0x1x0x0> -> <1x96x75x75xf16, 540000x5625x75x1>
    %203 = builtin.unrealized_conversion_cast %202 : !migraphx.shaped<1x96x75x75xf16, 540000x5625x75x1> to !dxgml.tensor<1x96x75x75x!dxgml.float16>
    %204 = dxgml_op.clip (%203) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x96x75x75x!dxgml.float16>) -> !dxgml.tensor<1x96x75x75x!dxgml.float16>
    %205 = builtin.unrealized_conversion_cast %204 : !dxgml.tensor<1x96x75x75x!dxgml.float16> to !migraphx.shaped<1x96x75x75xf16, 540000x5625x75x1>
    %206 = migraphx.convolution %205, %39 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x96x75x75xf16, 540000x5625x75x1>, <24x96x1x1xf16, 96x1x1x1> -> <1x24x75x75xf16, 135000x5625x75x1>
    %207 = migraphx.multibroadcast %40 {out_lens = [1, 24, 75, 75]} : <24xf16, 1> -> <1x24x75x75xf16, 0x1x0x0>
    %208 = migraphx.add %206, %207 : <1x24x75x75xf16, 135000x5625x75x1>, <1x24x75x75xf16, 0x1x0x0> -> <1x24x75x75xf16, 135000x5625x75x1>
    %209 = migraphx.convolution %208, %41 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x24x75x75xf16, 135000x5625x75x1>, <144x24x1x1xf16, 24x1x1x1> -> <1x144x75x75xf16, 810000x5625x75x1>
    %210 = migraphx.multibroadcast %42 {out_lens = [1, 144, 75, 75]} : <144xf16, 1> -> <1x144x75x75xf16, 0x1x0x0>
    %211 = migraphx.add %209, %210 : <1x144x75x75xf16, 810000x5625x75x1>, <1x144x75x75xf16, 0x1x0x0> -> <1x144x75x75xf16, 810000x5625x75x1>
    %212 = builtin.unrealized_conversion_cast %211 : !migraphx.shaped<1x144x75x75xf16, 810000x5625x75x1> to !dxgml.tensor<1x144x75x75x!dxgml.float16>
    %213 = dxgml_op.clip (%212) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x144x75x75x!dxgml.float16>) -> !dxgml.tensor<1x144x75x75x!dxgml.float16>
    %214 = builtin.unrealized_conversion_cast %213 : !dxgml.tensor<1x144x75x75x!dxgml.float16> to !migraphx.shaped<1x144x75x75xf16, 810000x5625x75x1>
    %215 = migraphx.convolution %214, %43 {dilation = [1, 1], group = 144 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x144x75x75xf16, 810000x5625x75x1>, <144x1x3x3xf16, 9x9x3x1> -> <1x144x75x75xf16, 810000x5625x75x1>
    %216 = migraphx.multibroadcast %44 {out_lens = [1, 144, 75, 75]} : <144xf16, 1> -> <1x144x75x75xf16, 0x1x0x0>
    %217 = migraphx.add %215, %216 : <1x144x75x75xf16, 810000x5625x75x1>, <1x144x75x75xf16, 0x1x0x0> -> <1x144x75x75xf16, 810000x5625x75x1>
    %218 = builtin.unrealized_conversion_cast %217 : !migraphx.shaped<1x144x75x75xf16, 810000x5625x75x1> to !dxgml.tensor<1x144x75x75x!dxgml.float16>
    %219 = dxgml_op.clip (%218) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x144x75x75x!dxgml.float16>) -> !dxgml.tensor<1x144x75x75x!dxgml.float16>
    %220 = builtin.unrealized_conversion_cast %219 : !dxgml.tensor<1x144x75x75x!dxgml.float16> to !migraphx.shaped<1x144x75x75xf16, 810000x5625x75x1>
    %221 = migraphx.convolution %220, %45 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x144x75x75xf16, 810000x5625x75x1>, <24x144x1x1xf16, 144x1x1x1> -> <1x24x75x75xf16, 135000x5625x75x1>
    %222 = migraphx.multibroadcast %46 {out_lens = [1, 24, 75, 75]} : <24xf16, 1> -> <1x24x75x75xf16, 0x1x0x0>
    %223 = migraphx.add %221, %222 : <1x24x75x75xf16, 135000x5625x75x1>, <1x24x75x75xf16, 0x1x0x0> -> <1x24x75x75xf16, 135000x5625x75x1>
    %224 = migraphx.add %208, %223 : <1x24x75x75xf16, 135000x5625x75x1>, <1x24x75x75xf16, 135000x5625x75x1> -> <1x24x75x75xf16, 135000x5625x75x1>
    %225 = migraphx.convolution %224, %47 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x24x75x75xf16, 135000x5625x75x1>, <144x24x1x1xf16, 24x1x1x1> -> <1x144x75x75xf16, 810000x5625x75x1>
    %226 = migraphx.multibroadcast %48 {out_lens = [1, 144, 75, 75]} : <144xf16, 1> -> <1x144x75x75xf16, 0x1x0x0>
    %227 = migraphx.add %225, %226 : <1x144x75x75xf16, 810000x5625x75x1>, <1x144x75x75xf16, 0x1x0x0> -> <1x144x75x75xf16, 810000x5625x75x1>
    %228 = builtin.unrealized_conversion_cast %227 : !migraphx.shaped<1x144x75x75xf16, 810000x5625x75x1> to !dxgml.tensor<1x144x75x75x!dxgml.float16>
    %229 = dxgml_op.clip (%228) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x144x75x75x!dxgml.float16>) -> !dxgml.tensor<1x144x75x75x!dxgml.float16>
    %230 = builtin.unrealized_conversion_cast %229 : !dxgml.tensor<1x144x75x75x!dxgml.float16> to !migraphx.shaped<1x144x75x75xf16, 810000x5625x75x1>
    %231 = migraphx.convolution %230, %49 {dilation = [1, 1], group = 144 : i64, padding = [1, 1, 1, 1], stride = [2, 2]} : <1x144x75x75xf16, 810000x5625x75x1>, <144x1x3x3xf16, 9x9x3x1> -> <1x144x38x38xf16, 207936x1444x38x1>
    %232 = migraphx.multibroadcast %50 {out_lens = [1, 144, 38, 38]} : <144xf16, 1> -> <1x144x38x38xf16, 0x1x0x0>
    %233 = migraphx.add %231, %232 : <1x144x38x38xf16, 207936x1444x38x1>, <1x144x38x38xf16, 0x1x0x0> -> <1x144x38x38xf16, 207936x1444x38x1>
    %234 = builtin.unrealized_conversion_cast %233 : !migraphx.shaped<1x144x38x38xf16, 207936x1444x38x1> to !dxgml.tensor<1x144x38x38x!dxgml.float16>
    %235 = dxgml_op.clip (%234) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x144x38x38x!dxgml.float16>) -> !dxgml.tensor<1x144x38x38x!dxgml.float16>
    %236 = builtin.unrealized_conversion_cast %235 : !dxgml.tensor<1x144x38x38x!dxgml.float16> to !migraphx.shaped<1x144x38x38xf16, 207936x1444x38x1>
    %237 = migraphx.convolution %236, %51 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x144x38x38xf16, 207936x1444x38x1>, <32x144x1x1xf16, 144x1x1x1> -> <1x32x38x38xf16, 46208x1444x38x1>
    %238 = migraphx.multibroadcast %52 {out_lens = [1, 32, 38, 38]} : <32xf16, 1> -> <1x32x38x38xf16, 0x1x0x0>
    %239 = migraphx.add %237, %238 : <1x32x38x38xf16, 46208x1444x38x1>, <1x32x38x38xf16, 0x1x0x0> -> <1x32x38x38xf16, 46208x1444x38x1>
    %240 = migraphx.convolution %239, %53 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x32x38x38xf16, 46208x1444x38x1>, <192x32x1x1xf16, 32x1x1x1> -> <1x192x38x38xf16, 277248x1444x38x1>
    %241 = migraphx.multibroadcast %54 {out_lens = [1, 192, 38, 38]} : <192xf16, 1> -> <1x192x38x38xf16, 0x1x0x0>
    %242 = migraphx.add %240, %241 : <1x192x38x38xf16, 277248x1444x38x1>, <1x192x38x38xf16, 0x1x0x0> -> <1x192x38x38xf16, 277248x1444x38x1>
    %243 = builtin.unrealized_conversion_cast %242 : !migraphx.shaped<1x192x38x38xf16, 277248x1444x38x1> to !dxgml.tensor<1x192x38x38x!dxgml.float16>
    %244 = dxgml_op.clip (%243) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x192x38x38x!dxgml.float16>) -> !dxgml.tensor<1x192x38x38x!dxgml.float16>
    %245 = builtin.unrealized_conversion_cast %244 : !dxgml.tensor<1x192x38x38x!dxgml.float16> to !migraphx.shaped<1x192x38x38xf16, 277248x1444x38x1>
    %246 = migraphx.convolution %245, %55 {dilation = [1, 1], group = 192 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x192x38x38xf16, 277248x1444x38x1>, <192x1x3x3xf16, 9x9x3x1> -> <1x192x38x38xf16, 277248x1444x38x1>
    %247 = migraphx.multibroadcast %56 {out_lens = [1, 192, 38, 38]} : <192xf16, 1> -> <1x192x38x38xf16, 0x1x0x0>
    %248 = migraphx.add %246, %247 : <1x192x38x38xf16, 277248x1444x38x1>, <1x192x38x38xf16, 0x1x0x0> -> <1x192x38x38xf16, 277248x1444x38x1>
    %249 = builtin.unrealized_conversion_cast %248 : !migraphx.shaped<1x192x38x38xf16, 277248x1444x38x1> to !dxgml.tensor<1x192x38x38x!dxgml.float16>
    %250 = dxgml_op.clip (%249) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x192x38x38x!dxgml.float16>) -> !dxgml.tensor<1x192x38x38x!dxgml.float16>
    %251 = builtin.unrealized_conversion_cast %250 : !dxgml.tensor<1x192x38x38x!dxgml.float16> to !migraphx.shaped<1x192x38x38xf16, 277248x1444x38x1>
    %252 = migraphx.convolution %251, %57 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x192x38x38xf16, 277248x1444x38x1>, <32x192x1x1xf16, 192x1x1x1> -> <1x32x38x38xf16, 46208x1444x38x1>
    %253 = migraphx.multibroadcast %58 {out_lens = [1, 32, 38, 38]} : <32xf16, 1> -> <1x32x38x38xf16, 0x1x0x0>
    %254 = migraphx.add %252, %253 : <1x32x38x38xf16, 46208x1444x38x1>, <1x32x38x38xf16, 0x1x0x0> -> <1x32x38x38xf16, 46208x1444x38x1>
    %255 = migraphx.add %239, %254 : <1x32x38x38xf16, 46208x1444x38x1>, <1x32x38x38xf16, 46208x1444x38x1> -> <1x32x38x38xf16, 46208x1444x38x1>
    %256 = migraphx.convolution %255, %59 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x32x38x38xf16, 46208x1444x38x1>, <192x32x1x1xf16, 32x1x1x1> -> <1x192x38x38xf16, 277248x1444x38x1>
    %257 = migraphx.multibroadcast %60 {out_lens = [1, 192, 38, 38]} : <192xf16, 1> -> <1x192x38x38xf16, 0x1x0x0>
    %258 = migraphx.add %256, %257 : <1x192x38x38xf16, 277248x1444x38x1>, <1x192x38x38xf16, 0x1x0x0> -> <1x192x38x38xf16, 277248x1444x38x1>
    %259 = builtin.unrealized_conversion_cast %258 : !migraphx.shaped<1x192x38x38xf16, 277248x1444x38x1> to !dxgml.tensor<1x192x38x38x!dxgml.float16>
    %260 = dxgml_op.clip (%259) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x192x38x38x!dxgml.float16>) -> !dxgml.tensor<1x192x38x38x!dxgml.float16>
    %261 = builtin.unrealized_conversion_cast %260 : !dxgml.tensor<1x192x38x38x!dxgml.float16> to !migraphx.shaped<1x192x38x38xf16, 277248x1444x38x1>
    %262 = migraphx.convolution %261, %61 {dilation = [1, 1], group = 192 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x192x38x38xf16, 277248x1444x38x1>, <192x1x3x3xf16, 9x9x3x1> -> <1x192x38x38xf16, 277248x1444x38x1>
    %263 = migraphx.multibroadcast %62 {out_lens = [1, 192, 38, 38]} : <192xf16, 1> -> <1x192x38x38xf16, 0x1x0x0>
    %264 = migraphx.add %262, %263 : <1x192x38x38xf16, 277248x1444x38x1>, <1x192x38x38xf16, 0x1x0x0> -> <1x192x38x38xf16, 277248x1444x38x1>
    %265 = builtin.unrealized_conversion_cast %264 : !migraphx.shaped<1x192x38x38xf16, 277248x1444x38x1> to !dxgml.tensor<1x192x38x38x!dxgml.float16>
    %266 = dxgml_op.clip (%265) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x192x38x38x!dxgml.float16>) -> !dxgml.tensor<1x192x38x38x!dxgml.float16>
    %267 = builtin.unrealized_conversion_cast %266 : !dxgml.tensor<1x192x38x38x!dxgml.float16> to !migraphx.shaped<1x192x38x38xf16, 277248x1444x38x1>
    %268 = migraphx.convolution %267, %63 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x192x38x38xf16, 277248x1444x38x1>, <32x192x1x1xf16, 192x1x1x1> -> <1x32x38x38xf16, 46208x1444x38x1>
    %269 = migraphx.multibroadcast %64 {out_lens = [1, 32, 38, 38]} : <32xf16, 1> -> <1x32x38x38xf16, 0x1x0x0>
    %270 = migraphx.add %268, %269 : <1x32x38x38xf16, 46208x1444x38x1>, <1x32x38x38xf16, 0x1x0x0> -> <1x32x38x38xf16, 46208x1444x38x1>
    %271 = migraphx.add %255, %270 : <1x32x38x38xf16, 46208x1444x38x1>, <1x32x38x38xf16, 46208x1444x38x1> -> <1x32x38x38xf16, 46208x1444x38x1>
    %272 = migraphx.convolution %271, %65 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x32x38x38xf16, 46208x1444x38x1>, <192x32x1x1xf16, 32x1x1x1> -> <1x192x38x38xf16, 277248x1444x38x1>
    %273 = migraphx.multibroadcast %66 {out_lens = [1, 192, 38, 38]} : <192xf16, 1> -> <1x192x38x38xf16, 0x1x0x0>
    %274 = migraphx.add %272, %273 : <1x192x38x38xf16, 277248x1444x38x1>, <1x192x38x38xf16, 0x1x0x0> -> <1x192x38x38xf16, 277248x1444x38x1>
    %275 = builtin.unrealized_conversion_cast %274 : !migraphx.shaped<1x192x38x38xf16, 277248x1444x38x1> to !dxgml.tensor<1x192x38x38x!dxgml.float16>
    %276 = dxgml_op.clip (%275) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x192x38x38x!dxgml.float16>) -> !dxgml.tensor<1x192x38x38x!dxgml.float16>
    %277 = builtin.unrealized_conversion_cast %276 : !dxgml.tensor<1x192x38x38x!dxgml.float16> to !migraphx.shaped<1x192x38x38xf16, 277248x1444x38x1>
    %278 = migraphx.convolution %277, %67 {dilation = [1, 1], group = 192 : i64, padding = [1, 1, 1, 1], stride = [2, 2]} : <1x192x38x38xf16, 277248x1444x38x1>, <192x1x3x3xf16, 9x9x3x1> -> <1x192x19x19xf16, 69312x361x19x1>
    %279 = migraphx.multibroadcast %68 {out_lens = [1, 192, 19, 19]} : <192xf16, 1> -> <1x192x19x19xf16, 0x1x0x0>
    %280 = migraphx.add %278, %279 : <1x192x19x19xf16, 69312x361x19x1>, <1x192x19x19xf16, 0x1x0x0> -> <1x192x19x19xf16, 69312x361x19x1>
    %281 = builtin.unrealized_conversion_cast %280 : !migraphx.shaped<1x192x19x19xf16, 69312x361x19x1> to !dxgml.tensor<1x192x19x19x!dxgml.float16>
    %282 = dxgml_op.clip (%281) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x192x19x19x!dxgml.float16>) -> !dxgml.tensor<1x192x19x19x!dxgml.float16>
    %283 = builtin.unrealized_conversion_cast %282 : !dxgml.tensor<1x192x19x19x!dxgml.float16> to !migraphx.shaped<1x192x19x19xf16, 69312x361x19x1>
    %284 = migraphx.convolution %283, %69 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x192x19x19xf16, 69312x361x19x1>, <64x192x1x1xf16, 192x1x1x1> -> <1x64x19x19xf16, 23104x361x19x1>
    %285 = migraphx.multibroadcast %70 {out_lens = [1, 64, 19, 19]} : <64xf16, 1> -> <1x64x19x19xf16, 0x1x0x0>
    %286 = migraphx.add %284, %285 : <1x64x19x19xf16, 23104x361x19x1>, <1x64x19x19xf16, 0x1x0x0> -> <1x64x19x19xf16, 23104x361x19x1>
    %287 = migraphx.convolution %286, %71 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x64x19x19xf16, 23104x361x19x1>, <384x64x1x1xf16, 64x1x1x1> -> <1x384x19x19xf16, 138624x361x19x1>
    %288 = migraphx.multibroadcast %72 {out_lens = [1, 384, 19, 19]} : <384xf16, 1> -> <1x384x19x19xf16, 0x1x0x0>
    %289 = migraphx.add %287, %288 : <1x384x19x19xf16, 138624x361x19x1>, <1x384x19x19xf16, 0x1x0x0> -> <1x384x19x19xf16, 138624x361x19x1>
    %290 = builtin.unrealized_conversion_cast %289 : !migraphx.shaped<1x384x19x19xf16, 138624x361x19x1> to !dxgml.tensor<1x384x19x19x!dxgml.float16>
    %291 = dxgml_op.clip (%290) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x384x19x19x!dxgml.float16>) -> !dxgml.tensor<1x384x19x19x!dxgml.float16>
    %292 = builtin.unrealized_conversion_cast %291 : !dxgml.tensor<1x384x19x19x!dxgml.float16> to !migraphx.shaped<1x384x19x19xf16, 138624x361x19x1>
    %293 = migraphx.convolution %292, %73 {dilation = [1, 1], group = 384 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x384x19x19xf16, 138624x361x19x1>, <384x1x3x3xf16, 9x9x3x1> -> <1x384x19x19xf16, 138624x361x19x1>
    %294 = migraphx.multibroadcast %74 {out_lens = [1, 384, 19, 19]} : <384xf16, 1> -> <1x384x19x19xf16, 0x1x0x0>
    %295 = migraphx.add %293, %294 : <1x384x19x19xf16, 138624x361x19x1>, <1x384x19x19xf16, 0x1x0x0> -> <1x384x19x19xf16, 138624x361x19x1>
    %296 = builtin.unrealized_conversion_cast %295 : !migraphx.shaped<1x384x19x19xf16, 138624x361x19x1> to !dxgml.tensor<1x384x19x19x!dxgml.float16>
    %297 = dxgml_op.clip (%296) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x384x19x19x!dxgml.float16>) -> !dxgml.tensor<1x384x19x19x!dxgml.float16>
    %298 = builtin.unrealized_conversion_cast %297 : !dxgml.tensor<1x384x19x19x!dxgml.float16> to !migraphx.shaped<1x384x19x19xf16, 138624x361x19x1>
    %299 = migraphx.convolution %298, %75 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x384x19x19xf16, 138624x361x19x1>, <64x384x1x1xf16, 384x1x1x1> -> <1x64x19x19xf16, 23104x361x19x1>
    %300 = migraphx.multibroadcast %76 {out_lens = [1, 64, 19, 19]} : <64xf16, 1> -> <1x64x19x19xf16, 0x1x0x0>
    %301 = migraphx.add %299, %300 : <1x64x19x19xf16, 23104x361x19x1>, <1x64x19x19xf16, 0x1x0x0> -> <1x64x19x19xf16, 23104x361x19x1>
    %302 = migraphx.add %286, %301 : <1x64x19x19xf16, 23104x361x19x1>, <1x64x19x19xf16, 23104x361x19x1> -> <1x64x19x19xf16, 23104x361x19x1>
    %303 = migraphx.convolution %302, %77 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x64x19x19xf16, 23104x361x19x1>, <384x64x1x1xf16, 64x1x1x1> -> <1x384x19x19xf16, 138624x361x19x1>
    %304 = migraphx.multibroadcast %78 {out_lens = [1, 384, 19, 19]} : <384xf16, 1> -> <1x384x19x19xf16, 0x1x0x0>
    %305 = migraphx.add %303, %304 : <1x384x19x19xf16, 138624x361x19x1>, <1x384x19x19xf16, 0x1x0x0> -> <1x384x19x19xf16, 138624x361x19x1>
    %306 = builtin.unrealized_conversion_cast %305 : !migraphx.shaped<1x384x19x19xf16, 138624x361x19x1> to !dxgml.tensor<1x384x19x19x!dxgml.float16>
    %307 = dxgml_op.clip (%306) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x384x19x19x!dxgml.float16>) -> !dxgml.tensor<1x384x19x19x!dxgml.float16>
    %308 = builtin.unrealized_conversion_cast %307 : !dxgml.tensor<1x384x19x19x!dxgml.float16> to !migraphx.shaped<1x384x19x19xf16, 138624x361x19x1>
    %309 = migraphx.convolution %308, %79 {dilation = [1, 1], group = 384 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x384x19x19xf16, 138624x361x19x1>, <384x1x3x3xf16, 9x9x3x1> -> <1x384x19x19xf16, 138624x361x19x1>
    %310 = migraphx.multibroadcast %80 {out_lens = [1, 384, 19, 19]} : <384xf16, 1> -> <1x384x19x19xf16, 0x1x0x0>
    %311 = migraphx.add %309, %310 : <1x384x19x19xf16, 138624x361x19x1>, <1x384x19x19xf16, 0x1x0x0> -> <1x384x19x19xf16, 138624x361x19x1>
    %312 = builtin.unrealized_conversion_cast %311 : !migraphx.shaped<1x384x19x19xf16, 138624x361x19x1> to !dxgml.tensor<1x384x19x19x!dxgml.float16>
    %313 = dxgml_op.clip (%312) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x384x19x19x!dxgml.float16>) -> !dxgml.tensor<1x384x19x19x!dxgml.float16>
    %314 = builtin.unrealized_conversion_cast %313 : !dxgml.tensor<1x384x19x19x!dxgml.float16> to !migraphx.shaped<1x384x19x19xf16, 138624x361x19x1>
    %315 = migraphx.convolution %314, %81 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x384x19x19xf16, 138624x361x19x1>, <64x384x1x1xf16, 384x1x1x1> -> <1x64x19x19xf16, 23104x361x19x1>
    %316 = migraphx.multibroadcast %82 {out_lens = [1, 64, 19, 19]} : <64xf16, 1> -> <1x64x19x19xf16, 0x1x0x0>
    %317 = migraphx.add %315, %316 : <1x64x19x19xf16, 23104x361x19x1>, <1x64x19x19xf16, 0x1x0x0> -> <1x64x19x19xf16, 23104x361x19x1>
    %318 = migraphx.add %302, %317 : <1x64x19x19xf16, 23104x361x19x1>, <1x64x19x19xf16, 23104x361x19x1> -> <1x64x19x19xf16, 23104x361x19x1>
    %319 = migraphx.convolution %318, %83 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x64x19x19xf16, 23104x361x19x1>, <384x64x1x1xf16, 64x1x1x1> -> <1x384x19x19xf16, 138624x361x19x1>
    %320 = migraphx.multibroadcast %84 {out_lens = [1, 384, 19, 19]} : <384xf16, 1> -> <1x384x19x19xf16, 0x1x0x0>
    %321 = migraphx.add %319, %320 : <1x384x19x19xf16, 138624x361x19x1>, <1x384x19x19xf16, 0x1x0x0> -> <1x384x19x19xf16, 138624x361x19x1>
    %322 = builtin.unrealized_conversion_cast %321 : !migraphx.shaped<1x384x19x19xf16, 138624x361x19x1> to !dxgml.tensor<1x384x19x19x!dxgml.float16>
    %323 = dxgml_op.clip (%322) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x384x19x19x!dxgml.float16>) -> !dxgml.tensor<1x384x19x19x!dxgml.float16>
    %324 = builtin.unrealized_conversion_cast %323 : !dxgml.tensor<1x384x19x19x!dxgml.float16> to !migraphx.shaped<1x384x19x19xf16, 138624x361x19x1>
    %325 = migraphx.convolution %324, %85 {dilation = [1, 1], group = 384 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x384x19x19xf16, 138624x361x19x1>, <384x1x3x3xf16, 9x9x3x1> -> <1x384x19x19xf16, 138624x361x19x1>
    %326 = migraphx.multibroadcast %86 {out_lens = [1, 384, 19, 19]} : <384xf16, 1> -> <1x384x19x19xf16, 0x1x0x0>
    %327 = migraphx.add %325, %326 : <1x384x19x19xf16, 138624x361x19x1>, <1x384x19x19xf16, 0x1x0x0> -> <1x384x19x19xf16, 138624x361x19x1>
    %328 = builtin.unrealized_conversion_cast %327 : !migraphx.shaped<1x384x19x19xf16, 138624x361x19x1> to !dxgml.tensor<1x384x19x19x!dxgml.float16>
    %329 = dxgml_op.clip (%328) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x384x19x19x!dxgml.float16>) -> !dxgml.tensor<1x384x19x19x!dxgml.float16>
    %330 = builtin.unrealized_conversion_cast %329 : !dxgml.tensor<1x384x19x19x!dxgml.float16> to !migraphx.shaped<1x384x19x19xf16, 138624x361x19x1>
    %331 = migraphx.convolution %330, %87 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x384x19x19xf16, 138624x361x19x1>, <64x384x1x1xf16, 384x1x1x1> -> <1x64x19x19xf16, 23104x361x19x1>
    %332 = migraphx.multibroadcast %88 {out_lens = [1, 64, 19, 19]} : <64xf16, 1> -> <1x64x19x19xf16, 0x1x0x0>
    %333 = migraphx.add %331, %332 : <1x64x19x19xf16, 23104x361x19x1>, <1x64x19x19xf16, 0x1x0x0> -> <1x64x19x19xf16, 23104x361x19x1>
    %334 = migraphx.add %318, %333 : <1x64x19x19xf16, 23104x361x19x1>, <1x64x19x19xf16, 23104x361x19x1> -> <1x64x19x19xf16, 23104x361x19x1>
    %335 = migraphx.convolution %334, %89 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x64x19x19xf16, 23104x361x19x1>, <384x64x1x1xf16, 64x1x1x1> -> <1x384x19x19xf16, 138624x361x19x1>
    %336 = migraphx.multibroadcast %90 {out_lens = [1, 384, 19, 19]} : <384xf16, 1> -> <1x384x19x19xf16, 0x1x0x0>
    %337 = migraphx.add %335, %336 : <1x384x19x19xf16, 138624x361x19x1>, <1x384x19x19xf16, 0x1x0x0> -> <1x384x19x19xf16, 138624x361x19x1>
    %338 = builtin.unrealized_conversion_cast %337 : !migraphx.shaped<1x384x19x19xf16, 138624x361x19x1> to !dxgml.tensor<1x384x19x19x!dxgml.float16>
    %339 = dxgml_op.clip (%338) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x384x19x19x!dxgml.float16>) -> !dxgml.tensor<1x384x19x19x!dxgml.float16>
    %340 = builtin.unrealized_conversion_cast %339 : !dxgml.tensor<1x384x19x19x!dxgml.float16> to !migraphx.shaped<1x384x19x19xf16, 138624x361x19x1>
    %341 = migraphx.convolution %340, %91 {dilation = [1, 1], group = 384 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x384x19x19xf16, 138624x361x19x1>, <384x1x3x3xf16, 9x9x3x1> -> <1x384x19x19xf16, 138624x361x19x1>
    %342 = migraphx.multibroadcast %92 {out_lens = [1, 384, 19, 19]} : <384xf16, 1> -> <1x384x19x19xf16, 0x1x0x0>
    %343 = migraphx.add %341, %342 : <1x384x19x19xf16, 138624x361x19x1>, <1x384x19x19xf16, 0x1x0x0> -> <1x384x19x19xf16, 138624x361x19x1>
    %344 = builtin.unrealized_conversion_cast %343 : !migraphx.shaped<1x384x19x19xf16, 138624x361x19x1> to !dxgml.tensor<1x384x19x19x!dxgml.float16>
    %345 = dxgml_op.clip (%344) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x384x19x19x!dxgml.float16>) -> !dxgml.tensor<1x384x19x19x!dxgml.float16>
    %346 = builtin.unrealized_conversion_cast %345 : !dxgml.tensor<1x384x19x19x!dxgml.float16> to !migraphx.shaped<1x384x19x19xf16, 138624x361x19x1>
    %347 = migraphx.convolution %346, %93 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x384x19x19xf16, 138624x361x19x1>, <96x384x1x1xf16, 384x1x1x1> -> <1x96x19x19xf16, 34656x361x19x1>
    %348 = migraphx.multibroadcast %94 {out_lens = [1, 96, 19, 19]} : <96xf16, 1> -> <1x96x19x19xf16, 0x1x0x0>
    %349 = migraphx.add %347, %348 : <1x96x19x19xf16, 34656x361x19x1>, <1x96x19x19xf16, 0x1x0x0> -> <1x96x19x19xf16, 34656x361x19x1>
    %350 = migraphx.convolution %349, %95 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x96x19x19xf16, 34656x361x19x1>, <576x96x1x1xf16, 96x1x1x1> -> <1x576x19x19xf16, 207936x361x19x1>
    %351 = migraphx.multibroadcast %96 {out_lens = [1, 576, 19, 19]} : <576xf16, 1> -> <1x576x19x19xf16, 0x1x0x0>
    %352 = migraphx.add %350, %351 : <1x576x19x19xf16, 207936x361x19x1>, <1x576x19x19xf16, 0x1x0x0> -> <1x576x19x19xf16, 207936x361x19x1>
    %353 = builtin.unrealized_conversion_cast %352 : !migraphx.shaped<1x576x19x19xf16, 207936x361x19x1> to !dxgml.tensor<1x576x19x19x!dxgml.float16>
    %354 = dxgml_op.clip (%353) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x576x19x19x!dxgml.float16>) -> !dxgml.tensor<1x576x19x19x!dxgml.float16>
    %355 = builtin.unrealized_conversion_cast %354 : !dxgml.tensor<1x576x19x19x!dxgml.float16> to !migraphx.shaped<1x576x19x19xf16, 207936x361x19x1>
    %356 = migraphx.convolution %355, %97 {dilation = [1, 1], group = 576 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x576x19x19xf16, 207936x361x19x1>, <576x1x3x3xf16, 9x9x3x1> -> <1x576x19x19xf16, 207936x361x19x1>
    %357 = migraphx.multibroadcast %98 {out_lens = [1, 576, 19, 19]} : <576xf16, 1> -> <1x576x19x19xf16, 0x1x0x0>
    %358 = migraphx.add %356, %357 : <1x576x19x19xf16, 207936x361x19x1>, <1x576x19x19xf16, 0x1x0x0> -> <1x576x19x19xf16, 207936x361x19x1>
    %359 = builtin.unrealized_conversion_cast %358 : !migraphx.shaped<1x576x19x19xf16, 207936x361x19x1> to !dxgml.tensor<1x576x19x19x!dxgml.float16>
    %360 = dxgml_op.clip (%359) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x576x19x19x!dxgml.float16>) -> !dxgml.tensor<1x576x19x19x!dxgml.float16>
    %361 = builtin.unrealized_conversion_cast %360 : !dxgml.tensor<1x576x19x19x!dxgml.float16> to !migraphx.shaped<1x576x19x19xf16, 207936x361x19x1>
    %362 = migraphx.convolution %361, %99 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x576x19x19xf16, 207936x361x19x1>, <96x576x1x1xf16, 576x1x1x1> -> <1x96x19x19xf16, 34656x361x19x1>
    %363 = migraphx.multibroadcast %100 {out_lens = [1, 96, 19, 19]} : <96xf16, 1> -> <1x96x19x19xf16, 0x1x0x0>
    %364 = migraphx.add %362, %363 : <1x96x19x19xf16, 34656x361x19x1>, <1x96x19x19xf16, 0x1x0x0> -> <1x96x19x19xf16, 34656x361x19x1>
    %365 = migraphx.add %349, %364 : <1x96x19x19xf16, 34656x361x19x1>, <1x96x19x19xf16, 34656x361x19x1> -> <1x96x19x19xf16, 34656x361x19x1>
    %366 = migraphx.convolution %365, %101 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x96x19x19xf16, 34656x361x19x1>, <576x96x1x1xf16, 96x1x1x1> -> <1x576x19x19xf16, 207936x361x19x1>
    %367 = migraphx.multibroadcast %102 {out_lens = [1, 576, 19, 19]} : <576xf16, 1> -> <1x576x19x19xf16, 0x1x0x0>
    %368 = migraphx.add %366, %367 : <1x576x19x19xf16, 207936x361x19x1>, <1x576x19x19xf16, 0x1x0x0> -> <1x576x19x19xf16, 207936x361x19x1>
    %369 = builtin.unrealized_conversion_cast %368 : !migraphx.shaped<1x576x19x19xf16, 207936x361x19x1> to !dxgml.tensor<1x576x19x19x!dxgml.float16>
    %370 = dxgml_op.clip (%369) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x576x19x19x!dxgml.float16>) -> !dxgml.tensor<1x576x19x19x!dxgml.float16>
    %371 = builtin.unrealized_conversion_cast %370 : !dxgml.tensor<1x576x19x19x!dxgml.float16> to !migraphx.shaped<1x576x19x19xf16, 207936x361x19x1>
    %372 = migraphx.convolution %371, %103 {dilation = [1, 1], group = 576 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x576x19x19xf16, 207936x361x19x1>, <576x1x3x3xf16, 9x9x3x1> -> <1x576x19x19xf16, 207936x361x19x1>
    %373 = migraphx.multibroadcast %104 {out_lens = [1, 576, 19, 19]} : <576xf16, 1> -> <1x576x19x19xf16, 0x1x0x0>
    %374 = migraphx.add %372, %373 : <1x576x19x19xf16, 207936x361x19x1>, <1x576x19x19xf16, 0x1x0x0> -> <1x576x19x19xf16, 207936x361x19x1>
    %375 = builtin.unrealized_conversion_cast %374 : !migraphx.shaped<1x576x19x19xf16, 207936x361x19x1> to !dxgml.tensor<1x576x19x19x!dxgml.float16>
    %376 = dxgml_op.clip (%375) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x576x19x19x!dxgml.float16>) -> !dxgml.tensor<1x576x19x19x!dxgml.float16>
    %377 = builtin.unrealized_conversion_cast %376 : !dxgml.tensor<1x576x19x19x!dxgml.float16> to !migraphx.shaped<1x576x19x19xf16, 207936x361x19x1>
    %378 = migraphx.convolution %377, %105 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x576x19x19xf16, 207936x361x19x1>, <96x576x1x1xf16, 576x1x1x1> -> <1x96x19x19xf16, 34656x361x19x1>
    %379 = migraphx.multibroadcast %106 {out_lens = [1, 96, 19, 19]} : <96xf16, 1> -> <1x96x19x19xf16, 0x1x0x0>
    %380 = migraphx.add %378, %379 : <1x96x19x19xf16, 34656x361x19x1>, <1x96x19x19xf16, 0x1x0x0> -> <1x96x19x19xf16, 34656x361x19x1>
    %381 = migraphx.add %365, %380 : <1x96x19x19xf16, 34656x361x19x1>, <1x96x19x19xf16, 34656x361x19x1> -> <1x96x19x19xf16, 34656x361x19x1>
    %382 = migraphx.convolution %381, %107 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x96x19x19xf16, 34656x361x19x1>, <576x96x1x1xf16, 96x1x1x1> -> <1x576x19x19xf16, 207936x361x19x1>
    %383 = migraphx.multibroadcast %108 {out_lens = [1, 576, 19, 19]} : <576xf16, 1> -> <1x576x19x19xf16, 0x1x0x0>
    %384 = migraphx.add %382, %383 : <1x576x19x19xf16, 207936x361x19x1>, <1x576x19x19xf16, 0x1x0x0> -> <1x576x19x19xf16, 207936x361x19x1>
    %385 = builtin.unrealized_conversion_cast %384 : !migraphx.shaped<1x576x19x19xf16, 207936x361x19x1> to !dxgml.tensor<1x576x19x19x!dxgml.float16>
    %386 = dxgml_op.clip (%385) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x576x19x19x!dxgml.float16>) -> !dxgml.tensor<1x576x19x19x!dxgml.float16>
    %387 = builtin.unrealized_conversion_cast %386 : !dxgml.tensor<1x576x19x19x!dxgml.float16> to !migraphx.shaped<1x576x19x19xf16, 207936x361x19x1>
    %388 = migraphx.convolution %387, %109 {dilation = [1, 1], group = 576 : i64, padding = [1, 1, 1, 1], stride = [2, 2]} : <1x576x19x19xf16, 207936x361x19x1>, <576x1x3x3xf16, 9x9x3x1> -> <1x576x10x10xf16, 57600x100x10x1>
    %389 = migraphx.multibroadcast %110 {out_lens = [1, 576, 10, 10]} : <576xf16, 1> -> <1x576x10x10xf16, 0x1x0x0>
    %390 = migraphx.add %388, %389 : <1x576x10x10xf16, 57600x100x10x1>, <1x576x10x10xf16, 0x1x0x0> -> <1x576x10x10xf16, 57600x100x10x1>
    %391 = builtin.unrealized_conversion_cast %390 : !migraphx.shaped<1x576x10x10xf16, 57600x100x10x1> to !dxgml.tensor<1x576x10x10x!dxgml.float16>
    %392 = dxgml_op.clip (%391) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x576x10x10x!dxgml.float16>) -> !dxgml.tensor<1x576x10x10x!dxgml.float16>
    %393 = builtin.unrealized_conversion_cast %392 : !dxgml.tensor<1x576x10x10x!dxgml.float16> to !migraphx.shaped<1x576x10x10xf16, 57600x100x10x1>
    %394 = migraphx.convolution %393, %111 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x576x10x10xf16, 57600x100x10x1>, <160x576x1x1xf16, 576x1x1x1> -> <1x160x10x10xf16, 16000x100x10x1>
    %395 = migraphx.multibroadcast %112 {out_lens = [1, 160, 10, 10]} : <160xf16, 1> -> <1x160x10x10xf16, 0x1x0x0>
    %396 = migraphx.add %394, %395 : <1x160x10x10xf16, 16000x100x10x1>, <1x160x10x10xf16, 0x1x0x0> -> <1x160x10x10xf16, 16000x100x10x1>
    %397 = migraphx.convolution %387, %113 {dilation = [1, 1], group = 576 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x576x19x19xf16, 207936x361x19x1>, <576x1x3x3xf16, 9x9x3x1> -> <1x576x19x19xf16, 207936x361x19x1>
    %398 = migraphx.multibroadcast %114 {out_lens = [1, 576, 19, 19]} : <576xf16, 1> -> <1x576x19x19xf16, 0x1x0x0>
    %399 = migraphx.add %397, %398 : <1x576x19x19xf16, 207936x361x19x1>, <1x576x19x19xf16, 0x1x0x0> -> <1x576x19x19xf16, 207936x361x19x1>
    %400 = builtin.unrealized_conversion_cast %399 : !migraphx.shaped<1x576x19x19xf16, 207936x361x19x1> to !dxgml.tensor<1x576x19x19x!dxgml.float16>
    %401 = dxgml_op.clip (%400) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x576x19x19x!dxgml.float16>) -> !dxgml.tensor<1x576x19x19x!dxgml.float16>
    %402 = builtin.unrealized_conversion_cast %401 : !dxgml.tensor<1x576x19x19x!dxgml.float16> to !migraphx.shaped<1x576x19x19xf16, 207936x361x19x1>
    %403 = migraphx.convolution %402, %5 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x576x19x19xf16, 207936x361x19x1>, <12x576x1x1xf16, 576x1x1x1> -> <1x12x19x19xf16, 4332x361x19x1>
    %404 = migraphx.multibroadcast %6 {out_lens = [1, 12, 19, 19]} : <12xf16, 1> -> <1x12x19x19xf16, 0x1x0x0>
    %405 = migraphx.add %403, %404 : <1x12x19x19xf16, 4332x361x19x1>, <1x12x19x19xf16, 0x1x0x0> -> <1x12x19x19xf16, 4332x361x19x1>
    %406 = migraphx.transpose %405 {permutation = [0, 2, 3, 1]} : <1x12x19x19xf16, 4332x361x19x1> -> <1x19x19x12xf16, 4332x228x12x1>
    %407 = migraphx.reshape %406 {dims = [1, 2166, 2]} : <1x19x19x12xf16, 4332x228x12x1> -> <1x2166x2xf16, 4332x2x1>
    %408 = builtin.unrealized_conversion_cast %407 : !migraphx.shaped<1x2166x2xf16, 4332x2x1> to !dxgml.tensor<1x2166x2x!dxgml.float16>
    %409 = migraphx.convolution %387, %115 {dilation = [1, 1], group = 576 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x576x19x19xf16, 207936x361x19x1>, <576x1x3x3xf16, 9x9x3x1> -> <1x576x19x19xf16, 207936x361x19x1>
    %410 = migraphx.multibroadcast %116 {out_lens = [1, 576, 19, 19]} : <576xf16, 1> -> <1x576x19x19xf16, 0x1x0x0>
    %411 = migraphx.add %409, %410 : <1x576x19x19xf16, 207936x361x19x1>, <1x576x19x19xf16, 0x1x0x0> -> <1x576x19x19xf16, 207936x361x19x1>
    %412 = builtin.unrealized_conversion_cast %411 : !migraphx.shaped<1x576x19x19xf16, 207936x361x19x1> to !dxgml.tensor<1x576x19x19x!dxgml.float16>
    %413 = dxgml_op.clip (%412) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x576x19x19x!dxgml.float16>) -> !dxgml.tensor<1x576x19x19x!dxgml.float16>
    %414 = builtin.unrealized_conversion_cast %413 : !dxgml.tensor<1x576x19x19x!dxgml.float16> to !migraphx.shaped<1x576x19x19xf16, 207936x361x19x1>
    %415 = migraphx.convolution %414, %17 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x576x19x19xf16, 207936x361x19x1>, <24x576x1x1xf16, 576x1x1x1> -> <1x24x19x19xf16, 8664x361x19x1>
    %416 = migraphx.multibroadcast %18 {out_lens = [1, 24, 19, 19]} : <24xf16, 1> -> <1x24x19x19xf16, 0x1x0x0>
    %417 = migraphx.add %415, %416 : <1x24x19x19xf16, 8664x361x19x1>, <1x24x19x19xf16, 0x1x0x0> -> <1x24x19x19xf16, 8664x361x19x1>
    %418 = migraphx.transpose %417 {permutation = [0, 2, 3, 1]} : <1x24x19x19xf16, 8664x361x19x1> -> <1x19x19x24xf16, 8664x456x24x1>
    %419 = migraphx.reshape %418 {dims = [1, 2166, 4]} : <1x19x19x24xf16, 8664x456x24x1> -> <1x2166x4xf16, 8664x4x1>
    %420 = builtin.unrealized_conversion_cast %419 : !migraphx.shaped<1x2166x4xf16, 8664x4x1> to !dxgml.tensor<1x2166x4x!dxgml.float16>
    %421 = migraphx.convolution %396, %117 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x160x10x10xf16, 16000x100x10x1>, <960x160x1x1xf16, 160x1x1x1> -> <1x960x10x10xf16, 96000x100x10x1>
    %422 = migraphx.multibroadcast %118 {out_lens = [1, 960, 10, 10]} : <960xf16, 1> -> <1x960x10x10xf16, 0x1x0x0>
    %423 = migraphx.add %421, %422 : <1x960x10x10xf16, 96000x100x10x1>, <1x960x10x10xf16, 0x1x0x0> -> <1x960x10x10xf16, 96000x100x10x1>
    %424 = builtin.unrealized_conversion_cast %423 : !migraphx.shaped<1x960x10x10xf16, 96000x100x10x1> to !dxgml.tensor<1x960x10x10x!dxgml.float16>
    %425 = dxgml_op.clip (%424) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x960x10x10x!dxgml.float16>) -> !dxgml.tensor<1x960x10x10x!dxgml.float16>
    %426 = builtin.unrealized_conversion_cast %425 : !dxgml.tensor<1x960x10x10x!dxgml.float16> to !migraphx.shaped<1x960x10x10xf16, 96000x100x10x1>
    %427 = migraphx.convolution %426, %119 {dilation = [1, 1], group = 960 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x960x10x10xf16, 96000x100x10x1>, <960x1x3x3xf16, 9x9x3x1> -> <1x960x10x10xf16, 96000x100x10x1>
    %428 = migraphx.multibroadcast %120 {out_lens = [1, 960, 10, 10]} : <960xf16, 1> -> <1x960x10x10xf16, 0x1x0x0>
    %429 = migraphx.add %427, %428 : <1x960x10x10xf16, 96000x100x10x1>, <1x960x10x10xf16, 0x1x0x0> -> <1x960x10x10xf16, 96000x100x10x1>
    %430 = builtin.unrealized_conversion_cast %429 : !migraphx.shaped<1x960x10x10xf16, 96000x100x10x1> to !dxgml.tensor<1x960x10x10x!dxgml.float16>
    %431 = dxgml_op.clip (%430) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x960x10x10x!dxgml.float16>) -> !dxgml.tensor<1x960x10x10x!dxgml.float16>
    %432 = builtin.unrealized_conversion_cast %431 : !dxgml.tensor<1x960x10x10x!dxgml.float16> to !migraphx.shaped<1x960x10x10xf16, 96000x100x10x1>
    %433 = migraphx.convolution %432, %121 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x960x10x10xf16, 96000x100x10x1>, <160x960x1x1xf16, 960x1x1x1> -> <1x160x10x10xf16, 16000x100x10x1>
    %434 = migraphx.multibroadcast %122 {out_lens = [1, 160, 10, 10]} : <160xf16, 1> -> <1x160x10x10xf16, 0x1x0x0>
    %435 = migraphx.add %433, %434 : <1x160x10x10xf16, 16000x100x10x1>, <1x160x10x10xf16, 0x1x0x0> -> <1x160x10x10xf16, 16000x100x10x1>
    %436 = migraphx.add %396, %435 : <1x160x10x10xf16, 16000x100x10x1>, <1x160x10x10xf16, 16000x100x10x1> -> <1x160x10x10xf16, 16000x100x10x1>
    %437 = migraphx.convolution %436, %123 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x160x10x10xf16, 16000x100x10x1>, <960x160x1x1xf16, 160x1x1x1> -> <1x960x10x10xf16, 96000x100x10x1>
    %438 = migraphx.multibroadcast %124 {out_lens = [1, 960, 10, 10]} : <960xf16, 1> -> <1x960x10x10xf16, 0x1x0x0>
    %439 = migraphx.add %437, %438 : <1x960x10x10xf16, 96000x100x10x1>, <1x960x10x10xf16, 0x1x0x0> -> <1x960x10x10xf16, 96000x100x10x1>
    %440 = builtin.unrealized_conversion_cast %439 : !migraphx.shaped<1x960x10x10xf16, 96000x100x10x1> to !dxgml.tensor<1x960x10x10x!dxgml.float16>
    %441 = dxgml_op.clip (%440) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x960x10x10x!dxgml.float16>) -> !dxgml.tensor<1x960x10x10x!dxgml.float16>
    %442 = builtin.unrealized_conversion_cast %441 : !dxgml.tensor<1x960x10x10x!dxgml.float16> to !migraphx.shaped<1x960x10x10xf16, 96000x100x10x1>
    %443 = migraphx.convolution %442, %125 {dilation = [1, 1], group = 960 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x960x10x10xf16, 96000x100x10x1>, <960x1x3x3xf16, 9x9x3x1> -> <1x960x10x10xf16, 96000x100x10x1>
    %444 = migraphx.multibroadcast %126 {out_lens = [1, 960, 10, 10]} : <960xf16, 1> -> <1x960x10x10xf16, 0x1x0x0>
    %445 = migraphx.add %443, %444 : <1x960x10x10xf16, 96000x100x10x1>, <1x960x10x10xf16, 0x1x0x0> -> <1x960x10x10xf16, 96000x100x10x1>
    %446 = builtin.unrealized_conversion_cast %445 : !migraphx.shaped<1x960x10x10xf16, 96000x100x10x1> to !dxgml.tensor<1x960x10x10x!dxgml.float16>
    %447 = dxgml_op.clip (%446) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x960x10x10x!dxgml.float16>) -> !dxgml.tensor<1x960x10x10x!dxgml.float16>
    %448 = builtin.unrealized_conversion_cast %447 : !dxgml.tensor<1x960x10x10x!dxgml.float16> to !migraphx.shaped<1x960x10x10xf16, 96000x100x10x1>
    %449 = migraphx.convolution %448, %127 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x960x10x10xf16, 96000x100x10x1>, <160x960x1x1xf16, 960x1x1x1> -> <1x160x10x10xf16, 16000x100x10x1>
    %450 = migraphx.multibroadcast %128 {out_lens = [1, 160, 10, 10]} : <160xf16, 1> -> <1x160x10x10xf16, 0x1x0x0>
    %451 = migraphx.add %449, %450 : <1x160x10x10xf16, 16000x100x10x1>, <1x160x10x10xf16, 0x1x0x0> -> <1x160x10x10xf16, 16000x100x10x1>
    %452 = migraphx.add %436, %451 : <1x160x10x10xf16, 16000x100x10x1>, <1x160x10x10xf16, 16000x100x10x1> -> <1x160x10x10xf16, 16000x100x10x1>
    %453 = migraphx.convolution %452, %129 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x160x10x10xf16, 16000x100x10x1>, <960x160x1x1xf16, 160x1x1x1> -> <1x960x10x10xf16, 96000x100x10x1>
    %454 = migraphx.multibroadcast %130 {out_lens = [1, 960, 10, 10]} : <960xf16, 1> -> <1x960x10x10xf16, 0x1x0x0>
    %455 = migraphx.add %453, %454 : <1x960x10x10xf16, 96000x100x10x1>, <1x960x10x10xf16, 0x1x0x0> -> <1x960x10x10xf16, 96000x100x10x1>
    %456 = builtin.unrealized_conversion_cast %455 : !migraphx.shaped<1x960x10x10xf16, 96000x100x10x1> to !dxgml.tensor<1x960x10x10x!dxgml.float16>
    %457 = dxgml_op.clip (%456) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x960x10x10x!dxgml.float16>) -> !dxgml.tensor<1x960x10x10x!dxgml.float16>
    %458 = builtin.unrealized_conversion_cast %457 : !dxgml.tensor<1x960x10x10x!dxgml.float16> to !migraphx.shaped<1x960x10x10xf16, 96000x100x10x1>
    %459 = migraphx.convolution %458, %131 {dilation = [1, 1], group = 960 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x960x10x10xf16, 96000x100x10x1>, <960x1x3x3xf16, 9x9x3x1> -> <1x960x10x10xf16, 96000x100x10x1>
    %460 = migraphx.multibroadcast %132 {out_lens = [1, 960, 10, 10]} : <960xf16, 1> -> <1x960x10x10xf16, 0x1x0x0>
    %461 = migraphx.add %459, %460 : <1x960x10x10xf16, 96000x100x10x1>, <1x960x10x10xf16, 0x1x0x0> -> <1x960x10x10xf16, 96000x100x10x1>
    %462 = builtin.unrealized_conversion_cast %461 : !migraphx.shaped<1x960x10x10xf16, 96000x100x10x1> to !dxgml.tensor<1x960x10x10x!dxgml.float16>
    %463 = dxgml_op.clip (%462) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x960x10x10x!dxgml.float16>) -> !dxgml.tensor<1x960x10x10x!dxgml.float16>
    %464 = builtin.unrealized_conversion_cast %463 : !dxgml.tensor<1x960x10x10x!dxgml.float16> to !migraphx.shaped<1x960x10x10xf16, 96000x100x10x1>
    %465 = migraphx.convolution %464, %133 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x960x10x10xf16, 96000x100x10x1>, <320x960x1x1xf16, 960x1x1x1> -> <1x320x10x10xf16, 32000x100x10x1>
    %466 = migraphx.multibroadcast %134 {out_lens = [1, 320, 10, 10]} : <320xf16, 1> -> <1x320x10x10xf16, 0x1x0x0>
    %467 = migraphx.add %465, %466 : <1x320x10x10xf16, 32000x100x10x1>, <1x320x10x10xf16, 0x1x0x0> -> <1x320x10x10xf16, 32000x100x10x1>
    %468 = migraphx.convolution %467, %135 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x320x10x10xf16, 32000x100x10x1>, <1280x320x1x1xf16, 320x1x1x1> -> <1x1280x10x10xf16, 128000x100x10x1>
    %469 = migraphx.multibroadcast %136 {out_lens = [1, 1280, 10, 10]} : <1280xf16, 1> -> <1x1280x10x10xf16, 0x1x0x0>
    %470 = migraphx.add %468, %469 : <1x1280x10x10xf16, 128000x100x10x1>, <1x1280x10x10xf16, 0x1x0x0> -> <1x1280x10x10xf16, 128000x100x10x1>
    %471 = builtin.unrealized_conversion_cast %470 : !migraphx.shaped<1x1280x10x10xf16, 128000x100x10x1> to !dxgml.tensor<1x1280x10x10x!dxgml.float16>
    %472 = dxgml_op.clip (%471) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x1280x10x10x!dxgml.float16>) -> !dxgml.tensor<1x1280x10x10x!dxgml.float16>
    %473 = builtin.unrealized_conversion_cast %472 : !dxgml.tensor<1x1280x10x10x!dxgml.float16> to !migraphx.shaped<1x1280x10x10xf16, 128000x100x10x1>
    %474 = migraphx.convolution %473, %137 {dilation = [1, 1], group = 1280 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x1280x10x10xf16, 128000x100x10x1>, <1280x1x3x3xf16, 9x9x3x1> -> <1x1280x10x10xf16, 128000x100x10x1>
    %475 = migraphx.multibroadcast %138 {out_lens = [1, 1280, 10, 10]} : <1280xf16, 1> -> <1x1280x10x10xf16, 0x1x0x0>
    %476 = migraphx.add %474, %475 : <1x1280x10x10xf16, 128000x100x10x1>, <1x1280x10x10xf16, 0x1x0x0> -> <1x1280x10x10xf16, 128000x100x10x1>
    %477 = builtin.unrealized_conversion_cast %476 : !migraphx.shaped<1x1280x10x10xf16, 128000x100x10x1> to !dxgml.tensor<1x1280x10x10x!dxgml.float16>
    %478 = dxgml_op.clip (%477) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x1280x10x10x!dxgml.float16>) -> !dxgml.tensor<1x1280x10x10x!dxgml.float16>
    %479 = builtin.unrealized_conversion_cast %478 : !dxgml.tensor<1x1280x10x10x!dxgml.float16> to !migraphx.shaped<1x1280x10x10xf16, 128000x100x10x1>
    %480 = migraphx.convolution %479, %7 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x1280x10x10xf16, 128000x100x10x1>, <24x1280x1x1xf16, 1280x1x1x1> -> <1x24x10x10xf16, 2400x100x10x1>
    %481 = migraphx.multibroadcast %8 {out_lens = [1, 24, 10, 10]} : <24xf16, 1> -> <1x24x10x10xf16, 0x1x0x0>
    %482 = migraphx.add %480, %481 : <1x24x10x10xf16, 2400x100x10x1>, <1x24x10x10xf16, 0x1x0x0> -> <1x24x10x10xf16, 2400x100x10x1>
    %483 = migraphx.transpose %482 {permutation = [0, 2, 3, 1]} : <1x24x10x10xf16, 2400x100x10x1> -> <1x10x10x24xf16, 2400x240x24x1>
    %484 = migraphx.reshape %483 {dims = [1, 1200, 2]} : <1x10x10x24xf16, 2400x240x24x1> -> <1x1200x2xf16, 2400x2x1>
    %485 = builtin.unrealized_conversion_cast %484 : !migraphx.shaped<1x1200x2xf16, 2400x2x1> to !dxgml.tensor<1x1200x2x!dxgml.float16>
    %486 = migraphx.convolution %473, %139 {dilation = [1, 1], group = 1280 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x1280x10x10xf16, 128000x100x10x1>, <1280x1x3x3xf16, 9x9x3x1> -> <1x1280x10x10xf16, 128000x100x10x1>
    %487 = migraphx.multibroadcast %140 {out_lens = [1, 1280, 10, 10]} : <1280xf16, 1> -> <1x1280x10x10xf16, 0x1x0x0>
    %488 = migraphx.add %486, %487 : <1x1280x10x10xf16, 128000x100x10x1>, <1x1280x10x10xf16, 0x1x0x0> -> <1x1280x10x10xf16, 128000x100x10x1>
    %489 = builtin.unrealized_conversion_cast %488 : !migraphx.shaped<1x1280x10x10xf16, 128000x100x10x1> to !dxgml.tensor<1x1280x10x10x!dxgml.float16>
    %490 = dxgml_op.clip (%489) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x1280x10x10x!dxgml.float16>) -> !dxgml.tensor<1x1280x10x10x!dxgml.float16>
    %491 = builtin.unrealized_conversion_cast %490 : !dxgml.tensor<1x1280x10x10x!dxgml.float16> to !migraphx.shaped<1x1280x10x10xf16, 128000x100x10x1>
    %492 = migraphx.convolution %491, %19 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x1280x10x10xf16, 128000x100x10x1>, <48x1280x1x1xf16, 1280x1x1x1> -> <1x48x10x10xf16, 4800x100x10x1>
    %493 = migraphx.multibroadcast %20 {out_lens = [1, 48, 10, 10]} : <48xf16, 1> -> <1x48x10x10xf16, 0x1x0x0>
    %494 = migraphx.add %492, %493 : <1x48x10x10xf16, 4800x100x10x1>, <1x48x10x10xf16, 0x1x0x0> -> <1x48x10x10xf16, 4800x100x10x1>
    %495 = migraphx.transpose %494 {permutation = [0, 2, 3, 1]} : <1x48x10x10xf16, 4800x100x10x1> -> <1x10x10x48xf16, 4800x480x48x1>
    %496 = migraphx.reshape %495 {dims = [1, 1200, 4]} : <1x10x10x48xf16, 4800x480x48x1> -> <1x1200x4xf16, 4800x4x1>
    %497 = builtin.unrealized_conversion_cast %496 : !migraphx.shaped<1x1200x4xf16, 4800x4x1> to !dxgml.tensor<1x1200x4x!dxgml.float16>
    %498 = migraphx.convolution %473, %141 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x1280x10x10xf16, 128000x100x10x1>, <256x1280x1x1xf16, 1280x1x1x1> -> <1x256x10x10xf16, 25600x100x10x1>
    %499 = migraphx.multibroadcast %142 {out_lens = [1, 256, 10, 10]} : <256xf16, 1> -> <1x256x10x10xf16, 0x1x0x0>
    %500 = migraphx.add %498, %499 : <1x256x10x10xf16, 25600x100x10x1>, <1x256x10x10xf16, 0x1x0x0> -> <1x256x10x10xf16, 25600x100x10x1>
    %501 = builtin.unrealized_conversion_cast %500 : !migraphx.shaped<1x256x10x10xf16, 25600x100x10x1> to !dxgml.tensor<1x256x10x10x!dxgml.float16>
    %502 = dxgml_op.clip (%501) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x256x10x10x!dxgml.float16>) -> !dxgml.tensor<1x256x10x10x!dxgml.float16>
    %503 = builtin.unrealized_conversion_cast %502 : !dxgml.tensor<1x256x10x10x!dxgml.float16> to !migraphx.shaped<1x256x10x10xf16, 25600x100x10x1>
    %504 = migraphx.convolution %503, %143 {dilation = [1, 1], group = 256 : i64, padding = [1, 1, 1, 1], stride = [2, 2]} : <1x256x10x10xf16, 25600x100x10x1>, <256x1x3x3xf16, 9x9x3x1> -> <1x256x5x5xf16, 6400x25x5x1>
    %505 = migraphx.multibroadcast %144 {out_lens = [1, 256, 5, 5]} : <256xf16, 1> -> <1x256x5x5xf16, 0x1x0x0>
    %506 = migraphx.add %504, %505 : <1x256x5x5xf16, 6400x25x5x1>, <1x256x5x5xf16, 0x1x0x0> -> <1x256x5x5xf16, 6400x25x5x1>
    %507 = builtin.unrealized_conversion_cast %506 : !migraphx.shaped<1x256x5x5xf16, 6400x25x5x1> to !dxgml.tensor<1x256x5x5x!dxgml.float16>
    %508 = dxgml_op.clip (%507) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x256x5x5x!dxgml.float16>) -> !dxgml.tensor<1x256x5x5x!dxgml.float16>
    %509 = builtin.unrealized_conversion_cast %508 : !dxgml.tensor<1x256x5x5x!dxgml.float16> to !migraphx.shaped<1x256x5x5xf16, 6400x25x5x1>
    %510 = migraphx.convolution %509, %145 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x256x5x5xf16, 6400x25x5x1>, <512x256x1x1xf16, 256x1x1x1> -> <1x512x5x5xf16, 12800x25x5x1>
    %511 = migraphx.multibroadcast %146 {out_lens = [1, 512, 5, 5]} : <512xf16, 1> -> <1x512x5x5xf16, 0x1x0x0>
    %512 = migraphx.add %510, %511 : <1x512x5x5xf16, 12800x25x5x1>, <1x512x5x5xf16, 0x1x0x0> -> <1x512x5x5xf16, 12800x25x5x1>
    %513 = migraphx.convolution %512, %147 {dilation = [1, 1], group = 512 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x512x5x5xf16, 12800x25x5x1>, <512x1x3x3xf16, 9x9x3x1> -> <1x512x5x5xf16, 12800x25x5x1>
    %514 = migraphx.multibroadcast %148 {out_lens = [1, 512, 5, 5]} : <512xf16, 1> -> <1x512x5x5xf16, 0x1x0x0>
    %515 = migraphx.add %513, %514 : <1x512x5x5xf16, 12800x25x5x1>, <1x512x5x5xf16, 0x1x0x0> -> <1x512x5x5xf16, 12800x25x5x1>
    %516 = builtin.unrealized_conversion_cast %515 : !migraphx.shaped<1x512x5x5xf16, 12800x25x5x1> to !dxgml.tensor<1x512x5x5x!dxgml.float16>
    %517 = dxgml_op.clip (%516) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x512x5x5x!dxgml.float16>) -> !dxgml.tensor<1x512x5x5x!dxgml.float16>
    %518 = builtin.unrealized_conversion_cast %517 : !dxgml.tensor<1x512x5x5x!dxgml.float16> to !migraphx.shaped<1x512x5x5xf16, 12800x25x5x1>
    %519 = migraphx.convolution %518, %9 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x512x5x5xf16, 12800x25x5x1>, <48x512x1x1xf16, 512x1x1x1> -> <1x48x5x5xf16, 1200x25x5x1>
    %520 = migraphx.multibroadcast %10 {out_lens = [1, 48, 5, 5]} : <48xf16, 1> -> <1x48x5x5xf16, 0x1x0x0>
    %521 = migraphx.add %519, %520 : <1x48x5x5xf16, 1200x25x5x1>, <1x48x5x5xf16, 0x1x0x0> -> <1x48x5x5xf16, 1200x25x5x1>
    %522 = migraphx.transpose %521 {permutation = [0, 2, 3, 1]} : <1x48x5x5xf16, 1200x25x5x1> -> <1x5x5x48xf16, 1200x240x48x1>
    %523 = migraphx.reshape %522 {dims = [1, 600, 2]} : <1x5x5x48xf16, 1200x240x48x1> -> <1x600x2xf16, 1200x2x1>
    %524 = builtin.unrealized_conversion_cast %523 : !migraphx.shaped<1x600x2xf16, 1200x2x1> to !dxgml.tensor<1x600x2x!dxgml.float16>
    %525 = migraphx.convolution %512, %149 {dilation = [1, 1], group = 512 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x512x5x5xf16, 12800x25x5x1>, <512x1x3x3xf16, 9x9x3x1> -> <1x512x5x5xf16, 12800x25x5x1>
    %526 = migraphx.multibroadcast %150 {out_lens = [1, 512, 5, 5]} : <512xf16, 1> -> <1x512x5x5xf16, 0x1x0x0>
    %527 = migraphx.add %525, %526 : <1x512x5x5xf16, 12800x25x5x1>, <1x512x5x5xf16, 0x1x0x0> -> <1x512x5x5xf16, 12800x25x5x1>
    %528 = builtin.unrealized_conversion_cast %527 : !migraphx.shaped<1x512x5x5xf16, 12800x25x5x1> to !dxgml.tensor<1x512x5x5x!dxgml.float16>
    %529 = dxgml_op.clip (%528) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x512x5x5x!dxgml.float16>) -> !dxgml.tensor<1x512x5x5x!dxgml.float16>
    %530 = builtin.unrealized_conversion_cast %529 : !dxgml.tensor<1x512x5x5x!dxgml.float16> to !migraphx.shaped<1x512x5x5xf16, 12800x25x5x1>
    %531 = migraphx.convolution %530, %21 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x512x5x5xf16, 12800x25x5x1>, <96x512x1x1xf16, 512x1x1x1> -> <1x96x5x5xf16, 2400x25x5x1>
    %532 = migraphx.multibroadcast %22 {out_lens = [1, 96, 5, 5]} : <96xf16, 1> -> <1x96x5x5xf16, 0x1x0x0>
    %533 = migraphx.add %531, %532 : <1x96x5x5xf16, 2400x25x5x1>, <1x96x5x5xf16, 0x1x0x0> -> <1x96x5x5xf16, 2400x25x5x1>
    %534 = migraphx.transpose %533 {permutation = [0, 2, 3, 1]} : <1x96x5x5xf16, 2400x25x5x1> -> <1x5x5x96xf16, 2400x480x96x1>
    %535 = migraphx.reshape %534 {dims = [1, 600, 4]} : <1x5x5x96xf16, 2400x480x96x1> -> <1x600x4xf16, 2400x4x1>
    %536 = builtin.unrealized_conversion_cast %535 : !migraphx.shaped<1x600x4xf16, 2400x4x1> to !dxgml.tensor<1x600x4x!dxgml.float16>
    %537 = migraphx.convolution %512, %151 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x512x5x5xf16, 12800x25x5x1>, <128x512x1x1xf16, 512x1x1x1> -> <1x128x5x5xf16, 3200x25x5x1>
    %538 = migraphx.multibroadcast %152 {out_lens = [1, 128, 5, 5]} : <128xf16, 1> -> <1x128x5x5xf16, 0x1x0x0>
    %539 = migraphx.add %537, %538 : <1x128x5x5xf16, 3200x25x5x1>, <1x128x5x5xf16, 0x1x0x0> -> <1x128x5x5xf16, 3200x25x5x1>
    %540 = builtin.unrealized_conversion_cast %539 : !migraphx.shaped<1x128x5x5xf16, 3200x25x5x1> to !dxgml.tensor<1x128x5x5x!dxgml.float16>
    %541 = dxgml_op.clip (%540) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x128x5x5x!dxgml.float16>) -> !dxgml.tensor<1x128x5x5x!dxgml.float16>
    %542 = builtin.unrealized_conversion_cast %541 : !dxgml.tensor<1x128x5x5x!dxgml.float16> to !migraphx.shaped<1x128x5x5xf16, 3200x25x5x1>
    %543 = migraphx.convolution %542, %153 {dilation = [1, 1], group = 128 : i64, padding = [1, 1, 1, 1], stride = [2, 2]} : <1x128x5x5xf16, 3200x25x5x1>, <128x1x3x3xf16, 9x9x3x1> -> <1x128x3x3xf16, 1152x9x3x1>
    %544 = migraphx.multibroadcast %154 {out_lens = [1, 128, 3, 3]} : <128xf16, 1> -> <1x128x3x3xf16, 0x1x0x0>
    %545 = migraphx.add %543, %544 : <1x128x3x3xf16, 1152x9x3x1>, <1x128x3x3xf16, 0x1x0x0> -> <1x128x3x3xf16, 1152x9x3x1>
    %546 = builtin.unrealized_conversion_cast %545 : !migraphx.shaped<1x128x3x3xf16, 1152x9x3x1> to !dxgml.tensor<1x128x3x3x!dxgml.float16>
    %547 = dxgml_op.clip (%546) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x128x3x3x!dxgml.float16>) -> !dxgml.tensor<1x128x3x3x!dxgml.float16>
    %548 = builtin.unrealized_conversion_cast %547 : !dxgml.tensor<1x128x3x3x!dxgml.float16> to !migraphx.shaped<1x128x3x3xf16, 1152x9x3x1>
    %549 = migraphx.convolution %548, %155 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x128x3x3xf16, 1152x9x3x1>, <256x128x1x1xf16, 128x1x1x1> -> <1x256x3x3xf16, 2304x9x3x1>
    %550 = migraphx.multibroadcast %156 {out_lens = [1, 256, 3, 3]} : <256xf16, 1> -> <1x256x3x3xf16, 0x1x0x0>
    %551 = migraphx.add %549, %550 : <1x256x3x3xf16, 2304x9x3x1>, <1x256x3x3xf16, 0x1x0x0> -> <1x256x3x3xf16, 2304x9x3x1>
    %552 = migraphx.convolution %551, %157 {dilation = [1, 1], group = 256 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x256x3x3xf16, 2304x9x3x1>, <256x1x3x3xf16, 9x9x3x1> -> <1x256x3x3xf16, 2304x9x3x1>
    %553 = migraphx.multibroadcast %158 {out_lens = [1, 256, 3, 3]} : <256xf16, 1> -> <1x256x3x3xf16, 0x1x0x0>
    %554 = migraphx.add %552, %553 : <1x256x3x3xf16, 2304x9x3x1>, <1x256x3x3xf16, 0x1x0x0> -> <1x256x3x3xf16, 2304x9x3x1>
    %555 = builtin.unrealized_conversion_cast %554 : !migraphx.shaped<1x256x3x3xf16, 2304x9x3x1> to !dxgml.tensor<1x256x3x3x!dxgml.float16>
    %556 = dxgml_op.clip (%555) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x256x3x3x!dxgml.float16>) -> !dxgml.tensor<1x256x3x3x!dxgml.float16>
    %557 = builtin.unrealized_conversion_cast %556 : !dxgml.tensor<1x256x3x3x!dxgml.float16> to !migraphx.shaped<1x256x3x3xf16, 2304x9x3x1>
    %558 = migraphx.convolution %557, %11 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x256x3x3xf16, 2304x9x3x1>, <48x256x1x1xf16, 256x1x1x1> -> <1x48x3x3xf16, 432x9x3x1>
    %559 = migraphx.multibroadcast %12 {out_lens = [1, 48, 3, 3]} : <48xf16, 1> -> <1x48x3x3xf16, 0x1x0x0>
    %560 = migraphx.add %558, %559 : <1x48x3x3xf16, 432x9x3x1>, <1x48x3x3xf16, 0x1x0x0> -> <1x48x3x3xf16, 432x9x3x1>
    %561 = migraphx.transpose %560 {permutation = [0, 2, 3, 1]} : <1x48x3x3xf16, 432x9x3x1> -> <1x3x3x48xf16, 432x144x48x1>
    %562 = migraphx.reshape %561 {dims = [1, 216, 2]} : <1x3x3x48xf16, 432x144x48x1> -> <1x216x2xf16, 432x2x1>
    %563 = builtin.unrealized_conversion_cast %562 : !migraphx.shaped<1x216x2xf16, 432x2x1> to !dxgml.tensor<1x216x2x!dxgml.float16>
    %564 = migraphx.convolution %551, %159 {dilation = [1, 1], group = 256 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x256x3x3xf16, 2304x9x3x1>, <256x1x3x3xf16, 9x9x3x1> -> <1x256x3x3xf16, 2304x9x3x1>
    %565 = migraphx.multibroadcast %160 {out_lens = [1, 256, 3, 3]} : <256xf16, 1> -> <1x256x3x3xf16, 0x1x0x0>
    %566 = migraphx.add %564, %565 : <1x256x3x3xf16, 2304x9x3x1>, <1x256x3x3xf16, 0x1x0x0> -> <1x256x3x3xf16, 2304x9x3x1>
    %567 = builtin.unrealized_conversion_cast %566 : !migraphx.shaped<1x256x3x3xf16, 2304x9x3x1> to !dxgml.tensor<1x256x3x3x!dxgml.float16>
    %568 = dxgml_op.clip (%567) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x256x3x3x!dxgml.float16>) -> !dxgml.tensor<1x256x3x3x!dxgml.float16>
    %569 = builtin.unrealized_conversion_cast %568 : !dxgml.tensor<1x256x3x3x!dxgml.float16> to !migraphx.shaped<1x256x3x3xf16, 2304x9x3x1>
    %570 = migraphx.convolution %569, %23 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x256x3x3xf16, 2304x9x3x1>, <96x256x1x1xf16, 256x1x1x1> -> <1x96x3x3xf16, 864x9x3x1>
    %571 = migraphx.multibroadcast %24 {out_lens = [1, 96, 3, 3]} : <96xf16, 1> -> <1x96x3x3xf16, 0x1x0x0>
    %572 = migraphx.add %570, %571 : <1x96x3x3xf16, 864x9x3x1>, <1x96x3x3xf16, 0x1x0x0> -> <1x96x3x3xf16, 864x9x3x1>
    %573 = migraphx.transpose %572 {permutation = [0, 2, 3, 1]} : <1x96x3x3xf16, 864x9x3x1> -> <1x3x3x96xf16, 864x288x96x1>
    %574 = migraphx.reshape %573 {dims = [1, 216, 4]} : <1x3x3x96xf16, 864x288x96x1> -> <1x216x4xf16, 864x4x1>
    %575 = builtin.unrealized_conversion_cast %574 : !migraphx.shaped<1x216x4xf16, 864x4x1> to !dxgml.tensor<1x216x4x!dxgml.float16>
    %576 = migraphx.convolution %551, %161 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x256x3x3xf16, 2304x9x3x1>, <128x256x1x1xf16, 256x1x1x1> -> <1x128x3x3xf16, 1152x9x3x1>
    %577 = migraphx.multibroadcast %162 {out_lens = [1, 128, 3, 3]} : <128xf16, 1> -> <1x128x3x3xf16, 0x1x0x0>
    %578 = migraphx.add %576, %577 : <1x128x3x3xf16, 1152x9x3x1>, <1x128x3x3xf16, 0x1x0x0> -> <1x128x3x3xf16, 1152x9x3x1>
    %579 = builtin.unrealized_conversion_cast %578 : !migraphx.shaped<1x128x3x3xf16, 1152x9x3x1> to !dxgml.tensor<1x128x3x3x!dxgml.float16>
    %580 = dxgml_op.clip (%579) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x128x3x3x!dxgml.float16>) -> !dxgml.tensor<1x128x3x3x!dxgml.float16>
    %581 = builtin.unrealized_conversion_cast %580 : !dxgml.tensor<1x128x3x3x!dxgml.float16> to !migraphx.shaped<1x128x3x3xf16, 1152x9x3x1>
    %582 = migraphx.convolution %581, %163 {dilation = [1, 1], group = 128 : i64, padding = [1, 1, 1, 1], stride = [2, 2]} : <1x128x3x3xf16, 1152x9x3x1>, <128x1x3x3xf16, 9x9x3x1> -> <1x128x2x2xf16, 512x4x2x1>
    %583 = migraphx.multibroadcast %164 {out_lens = [1, 128, 2, 2]} : <128xf16, 1> -> <1x128x2x2xf16, 0x1x0x0>
    %584 = migraphx.add %582, %583 : <1x128x2x2xf16, 512x4x2x1>, <1x128x2x2xf16, 0x1x0x0> -> <1x128x2x2xf16, 512x4x2x1>
    %585 = builtin.unrealized_conversion_cast %584 : !migraphx.shaped<1x128x2x2xf16, 512x4x2x1> to !dxgml.tensor<1x128x2x2x!dxgml.float16>
    %586 = dxgml_op.clip (%585) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x128x2x2x!dxgml.float16>) -> !dxgml.tensor<1x128x2x2x!dxgml.float16>
    %587 = builtin.unrealized_conversion_cast %586 : !dxgml.tensor<1x128x2x2x!dxgml.float16> to !migraphx.shaped<1x128x2x2xf16, 512x4x2x1>
    %588 = migraphx.convolution %587, %165 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x128x2x2xf16, 512x4x2x1>, <256x128x1x1xf16, 128x1x1x1> -> <1x256x2x2xf16, 1024x4x2x1>
    %589 = migraphx.multibroadcast %166 {out_lens = [1, 256, 2, 2]} : <256xf16, 1> -> <1x256x2x2xf16, 0x1x0x0>
    %590 = migraphx.add %588, %589 : <1x256x2x2xf16, 1024x4x2x1>, <1x256x2x2xf16, 0x1x0x0> -> <1x256x2x2xf16, 1024x4x2x1>
    %591 = migraphx.convolution %590, %167 {dilation = [1, 1], group = 256 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x256x2x2xf16, 1024x4x2x1>, <256x1x3x3xf16, 9x9x3x1> -> <1x256x2x2xf16, 1024x4x2x1>
    %592 = migraphx.multibroadcast %168 {out_lens = [1, 256, 2, 2]} : <256xf16, 1> -> <1x256x2x2xf16, 0x1x0x0>
    %593 = migraphx.add %591, %592 : <1x256x2x2xf16, 1024x4x2x1>, <1x256x2x2xf16, 0x1x0x0> -> <1x256x2x2xf16, 1024x4x2x1>
    %594 = builtin.unrealized_conversion_cast %593 : !migraphx.shaped<1x256x2x2xf16, 1024x4x2x1> to !dxgml.tensor<1x256x2x2x!dxgml.float16>
    %595 = dxgml_op.clip (%594) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x256x2x2x!dxgml.float16>) -> !dxgml.tensor<1x256x2x2x!dxgml.float16>
    %596 = builtin.unrealized_conversion_cast %595 : !dxgml.tensor<1x256x2x2x!dxgml.float16> to !migraphx.shaped<1x256x2x2xf16, 1024x4x2x1>
    %597 = migraphx.convolution %596, %13 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x256x2x2xf16, 1024x4x2x1>, <12x256x1x1xf16, 256x1x1x1> -> <1x12x2x2xf16, 48x4x2x1>
    %598 = migraphx.multibroadcast %14 {out_lens = [1, 12, 2, 2]} : <12xf16, 1> -> <1x12x2x2xf16, 0x1x0x0>
    %599 = migraphx.add %597, %598 : <1x12x2x2xf16, 48x4x2x1>, <1x12x2x2xf16, 0x1x0x0> -> <1x12x2x2xf16, 48x4x2x1>
    %600 = migraphx.transpose %599 {permutation = [0, 2, 3, 1]} : <1x12x2x2xf16, 48x4x2x1> -> <1x2x2x12xf16, 48x24x12x1>
    %601 = migraphx.reshape %600 {dims = [1, 24, 2]} : <1x2x2x12xf16, 48x24x12x1> -> <1x24x2xf16, 48x2x1>
    %602 = builtin.unrealized_conversion_cast %601 : !migraphx.shaped<1x24x2xf16, 48x2x1> to !dxgml.tensor<1x24x2x!dxgml.float16>
    %603 = migraphx.convolution %590, %169 {dilation = [1, 1], group = 256 : i64, padding = [1, 1, 1, 1], stride = [1, 1]} : <1x256x2x2xf16, 1024x4x2x1>, <256x1x3x3xf16, 9x9x3x1> -> <1x256x2x2xf16, 1024x4x2x1>
    %604 = migraphx.multibroadcast %170 {out_lens = [1, 256, 2, 2]} : <256xf16, 1> -> <1x256x2x2xf16, 0x1x0x0>
    %605 = migraphx.add %603, %604 : <1x256x2x2xf16, 1024x4x2x1>, <1x256x2x2xf16, 0x1x0x0> -> <1x256x2x2xf16, 1024x4x2x1>
    %606 = builtin.unrealized_conversion_cast %605 : !migraphx.shaped<1x256x2x2xf16, 1024x4x2x1> to !dxgml.tensor<1x256x2x2x!dxgml.float16>
    %607 = dxgml_op.clip (%606) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x256x2x2x!dxgml.float16>) -> !dxgml.tensor<1x256x2x2x!dxgml.float16>
    %608 = builtin.unrealized_conversion_cast %607 : !dxgml.tensor<1x256x2x2x!dxgml.float16> to !migraphx.shaped<1x256x2x2xf16, 1024x4x2x1>
    %609 = migraphx.convolution %608, %25 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x256x2x2xf16, 1024x4x2x1>, <24x256x1x1xf16, 256x1x1x1> -> <1x24x2x2xf16, 96x4x2x1>
    %610 = migraphx.multibroadcast %26 {out_lens = [1, 24, 2, 2]} : <24xf16, 1> -> <1x24x2x2xf16, 0x1x0x0>
    %611 = migraphx.add %609, %610 : <1x24x2x2xf16, 96x4x2x1>, <1x24x2x2xf16, 0x1x0x0> -> <1x24x2x2xf16, 96x4x2x1>
    %612 = migraphx.transpose %611 {permutation = [0, 2, 3, 1]} : <1x24x2x2xf16, 96x4x2x1> -> <1x2x2x24xf16, 96x48x24x1>
    %613 = migraphx.reshape %612 {dims = [1, 24, 4]} : <1x2x2x24xf16, 96x48x24x1> -> <1x24x4xf16, 96x4x1>
    %614 = builtin.unrealized_conversion_cast %613 : !migraphx.shaped<1x24x4xf16, 96x4x1> to !dxgml.tensor<1x24x4x!dxgml.float16>
    %615 = migraphx.convolution %590, %171 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x256x2x2xf16, 1024x4x2x1>, <64x256x1x1xf16, 256x1x1x1> -> <1x64x2x2xf16, 256x4x2x1>
    %616 = migraphx.multibroadcast %172 {out_lens = [1, 64, 2, 2]} : <64xf16, 1> -> <1x64x2x2xf16, 0x1x0x0>
    %617 = migraphx.add %615, %616 : <1x64x2x2xf16, 256x4x2x1>, <1x64x2x2xf16, 0x1x0x0> -> <1x64x2x2xf16, 256x4x2x1>
    %618 = builtin.unrealized_conversion_cast %617 : !migraphx.shaped<1x64x2x2xf16, 256x4x2x1> to !dxgml.tensor<1x64x2x2x!dxgml.float16>
    %619 = dxgml_op.clip (%618) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x64x2x2x!dxgml.float16>) -> !dxgml.tensor<1x64x2x2x!dxgml.float16>
    %620 = builtin.unrealized_conversion_cast %619 : !dxgml.tensor<1x64x2x2x!dxgml.float16> to !migraphx.shaped<1x64x2x2xf16, 256x4x2x1>
    %621 = migraphx.convolution %620, %173 {dilation = [1, 1], group = 64 : i64, padding = [1, 1, 1, 1], stride = [2, 2]} : <1x64x2x2xf16, 256x4x2x1>, <64x1x3x3xf16, 9x9x3x1> -> <1x64x1x1xf16, 64x1x1x1>
    %622 = migraphx.multibroadcast %174 {out_lens = [1, 64, 1, 1]} : <64xf16, 1> -> <1x64x1x1xf16, 0x1x0x0>
    %623 = migraphx.add %621, %622 : <1x64x1x1xf16, 64x1x1x1>, <1x64x1x1xf16, 0x1x0x0> -> <1x64x1x1xf16, 64x1x1x1>
    %624 = builtin.unrealized_conversion_cast %623 : !migraphx.shaped<1x64x1x1xf16, 64x1x1x1> to !dxgml.tensor<1x64x1x1x!dxgml.float16>
    %625 = dxgml_op.clip (%624) {max = #dxgml.float<6.000000e+00 : !dxgml.float32> : !dxgml.float32, min = #dxgml.float<0.000000e+00 : !dxgml.float32> : !dxgml.float32} : (!dxgml.tensor<1x64x1x1x!dxgml.float16>) -> !dxgml.tensor<1x64x1x1x!dxgml.float16>
    %626 = builtin.unrealized_conversion_cast %625 : !dxgml.tensor<1x64x1x1x!dxgml.float16> to !migraphx.shaped<1x64x1x1xf16, 64x1x1x1>
    %627 = migraphx.convolution %626, %175 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x64x1x1xf16, 64x1x1x1>, <64x64x1x1xf16, 64x1x1x1> -> <1x64x1x1xf16, 64x1x1x1>
    %628 = migraphx.multibroadcast %176 {out_lens = [1, 64, 1, 1]} : <64xf16, 1> -> <1x64x1x1xf16, 0x1x0x0>
    %629 = migraphx.add %627, %628 : <1x64x1x1xf16, 64x1x1x1>, <1x64x1x1xf16, 0x1x0x0> -> <1x64x1x1xf16, 64x1x1x1>
    %630 = migraphx.convolution %629, %15 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x64x1x1xf16, 64x1x1x1>, <12x64x1x1xf16, 64x1x1x1> -> <1x12x1x1xf16, 12x1x1x1>
    %631 = migraphx.multibroadcast %16 {out_lens = [1, 12, 1, 1]} : <12xf16, 1> -> <1x12x1x1xf16, 0x1x0x0>
    %632 = migraphx.add %630, %631 : <1x12x1x1xf16, 12x1x1x1>, <1x12x1x1xf16, 0x1x0x0> -> <1x12x1x1xf16, 12x1x1x1>
    %633 = migraphx.transpose %632 {permutation = [0, 2, 3, 1]} : <1x12x1x1xf16, 12x1x1x1> -> <1x1x1x12xf16, 12x12x12x1>
    %634 = migraphx.reshape %633 {dims = [1, 6, 2]} : <1x1x1x12xf16, 12x12x12x1> -> <1x6x2xf16, 12x2x1>
    %635 = builtin.unrealized_conversion_cast %634 : !migraphx.shaped<1x6x2xf16, 12x2x1> to !dxgml.tensor<1x6x2x!dxgml.float16>
    %636 = migraphx.convolution %629, %27 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], stride = [1, 1]} : <1x64x1x1xf16, 64x1x1x1>, <24x64x1x1xf16, 64x1x1x1> -> <1x24x1x1xf16, 24x1x1x1>
    %637 = migraphx.multibroadcast %28 {out_lens = [1, 24, 1, 1]} : <24xf16, 1> -> <1x24x1x1xf16, 0x1x0x0>
    %638 = migraphx.add %636, %637 : <1x24x1x1xf16, 24x1x1x1>, <1x24x1x1xf16, 0x1x0x0> -> <1x24x1x1xf16, 24x1x1x1>
    %639 = migraphx.transpose %638 {permutation = [0, 2, 3, 1]} : <1x24x1x1xf16, 24x1x1x1> -> <1x1x1x24xf16, 24x24x24x1>
    %640 = migraphx.reshape %639 {dims = [1, 6, 4]} : <1x1x1x24xf16, 24x24x24x1> -> <1x6x4xf16, 24x4x1>
    %641 = builtin.unrealized_conversion_cast %640 : !migraphx.shaped<1x6x4xf16, 24x4x1> to !dxgml.tensor<1x6x4x!dxgml.float16>
    %642 = dxgml_op.concat (%408, %485, %524, %563, %602, %635) {axis = #dxgml.integer<1 : !dxgml.int64> : !dxgml.int64} : (!dxgml.tensor<1x2166x2x!dxgml.float16>, !dxgml.tensor<1x1200x2x!dxgml.float16>, !dxgml.tensor<1x600x2x!dxgml.float16>, !dxgml.tensor<1x216x2x!dxgml.float16>, !dxgml.tensor<1x24x2x!dxgml.float16>, !dxgml.tensor<1x6x2x!dxgml.float16>) -> !dxgml.tensor<1x4212x2x!dxgml.float16>
    %643 = builtin.unrealized_conversion_cast %642 : !dxgml.tensor<1x4212x2x!dxgml.float16> to !migraphx.shaped<1x4212x2xf16, 8424x2x1>
    %644 = dxgml_op.concat (%420, %497, %536, %575, %614, %641) {axis = #dxgml.integer<1 : !dxgml.int64> : !dxgml.int64} : (!dxgml.tensor<1x2166x4x!dxgml.float16>, !dxgml.tensor<1x1200x4x!dxgml.float16>, !dxgml.tensor<1x600x4x!dxgml.float16>, !dxgml.tensor<1x216x4x!dxgml.float16>, !dxgml.tensor<1x24x4x!dxgml.float16>, !dxgml.tensor<1x6x4x!dxgml.float16>) -> !dxgml.tensor<1x4212x4x!dxgml.float16>
    %645 = builtin.unrealized_conversion_cast %644 : !dxgml.tensor<1x4212x4x!dxgml.float16> to !migraphx.shaped<1x4212x4xf16, 16848x4x1>
    %646 = migraphx.softmax %643 {axis = 2 : i64} : <1x4212x2xf16, 8424x2x1> -> <1x4212x2xf16, 8424x2x1>
    %647 = migraphx.slice %645 {axes = [2], ends = [2], starts = [0]} : <1x4212x4xf16, 16848x4x1> -> <1x4212x2xf16, 8424x2x1>
    %648 = migraphx.mul %647, %4 : <1x4212x2xf16, 8424x2x1>, <1xf16, 1> -> <1x4212x2xf16, 8424x2x1>
    %649 = migraphx.mul %648, %3 : <1x4212x2xf16, 8424x2x1>, <1x4212x2xf16, 8424x2x1> -> <1x4212x2xf16, 8424x2x1>
    %650 = migraphx.add %649, %2 : <1x4212x2xf16, 8424x2x1>, <1x4212x2xf16, 8424x2x1> -> <1x4212x2xf16, 8424x2x1>
    %651 = builtin.unrealized_conversion_cast %650 : !migraphx.shaped<1x4212x2xf16, 8424x2x1> to !dxgml.tensor<1x4212x2x!dxgml.float16>
    %652 = migraphx.slice %645 {axes = [2], ends = [9223372036854775807], starts = [2]} : <1x4212x4xf16, 16848x4x1> -> <1x4212x2xf16, 8424x2x1>
    %653 = migraphx.mul %652, %1 : <1x4212x2xf16, 8424x2x1>, <1xf16, 1> -> <1x4212x2xf16, 8424x2x1>
    %654 = migraphx.exp %653 : <1x4212x2xf16, 8424x2x1> -> <1x4212x2xf16, 8424x2x1>
    %655 = migraphx.mul %654, %0 : <1x4212x2xf16, 8424x2x1>, <1x4212x2xf16, 8424x2x1> -> <1x4212x2xf16, 8424x2x1>
    %656 = builtin.unrealized_conversion_cast %655 : !migraphx.shaped<1x4212x2xf16, 8424x2x1> to !dxgml.tensor<1x4212x2x!dxgml.float16>
    %657 = dxgml_op.concat (%651, %656) {axis = #dxgml.integer<2 : !dxgml.int64> : !dxgml.int64} : (!dxgml.tensor<1x4212x2x!dxgml.float16>, !dxgml.tensor<1x4212x2x!dxgml.float16>) -> !dxgml.tensor<1x4212x4x!dxgml.float16>
    %658 = builtin.unrealized_conversion_cast %657 : !dxgml.tensor<1x4212x4x!dxgml.float16> to !migraphx.shaped<1x4212x4xf16, 16848x4x1>
    %659 = migraphx.slice %658 {axes = [2], ends = [2], starts = [0]} : <1x4212x4xf16, 16848x4x1> -> <1x4212x2xf16, 8424x2x1>
    %660 = migraphx.slice %658 {axes = [2], ends = [9223372036854775807], starts = [2]} : <1x4212x4xf16, 16848x4x1> -> <1x4212x2xf16, 8424x2x1>
    %661 = migraphx.div %660, %177 : <1x4212x2xf16, 8424x2x1>, <1xf16, 1> -> <1x4212x2xf16, 8424x2x1>
    %662 = migraphx.sub %659, %661 : <1x4212x2xf16, 8424x2x1>, <1x4212x2xf16, 8424x2x1> -> <1x4212x2xf16, 8424x2x1>
    %663 = builtin.unrealized_conversion_cast %662 : !migraphx.shaped<1x4212x2xf16, 8424x2x1> to !dxgml.tensor<1x4212x2x!dxgml.float16>
    %664 = migraphx.slice %658 {axes = [2], ends = [2], starts = [0]} : <1x4212x4xf16, 16848x4x1> -> <1x4212x2xf16, 8424x2x1>
    %665 = migraphx.slice %658 {axes = [2], ends = [9223372036854775807], starts = [2]} : <1x4212x4xf16, 16848x4x1> -> <1x4212x2xf16, 8424x2x1>
    %666 = migraphx.div %665, %178 : <1x4212x2xf16, 8424x2x1>, <1xf16, 1> -> <1x4212x2xf16, 8424x2x1>
    %667 = migraphx.add %664, %666 : <1x4212x2xf16, 8424x2x1>, <1x4212x2xf16, 8424x2x1> -> <1x4212x2xf16, 8424x2x1>
    %668 = builtin.unrealized_conversion_cast %667 : !migraphx.shaped<1x4212x2xf16, 8424x2x1> to !dxgml.tensor<1x4212x2x!dxgml.float16>
    %669 = dxgml_op.concat (%663, %668) {axis = #dxgml.integer<2 : !dxgml.int64> : !dxgml.int64} : (!dxgml.tensor<1x4212x2x!dxgml.float16>, !dxgml.tensor<1x4212x2x!dxgml.float16>) -> !dxgml.tensor<1x4212x4x!dxgml.float16>
    %670 = builtin.unrealized_conversion_cast %669 : !dxgml.tensor<1x4212x4x!dxgml.float16> to !migraphx.shaped<1x4212x4xf16, 16848x4x1>
    return %646, %670 : !migraphx.shaped<1x4212x2xf16, 8424x2x1>, !migraphx.shaped<1x4212x4xf16, 16848x4x1>
  }
}
