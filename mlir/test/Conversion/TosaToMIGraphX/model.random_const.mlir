module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 716 : i32}}  {
  func @main(%arg0: tensor<100x224x224x3xf32>) -> tensor<100x1000xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "x", outputs = "Identity"}} {
    %0 = migraphx.constant() {shape = []} : tensor<f32>
    %1 = migraphx.constant() {shape = [1 : i32]} : tensor<1xf32>
    %2 = migraphx.constant() {shape = [4 : i32]} : tensor<4xi32>
    %3 = migraphx.constant() {shape = [2048 : i32, 1000 : i32]} : tensor<2048x1000xf32>
    %4 = migraphx.constant() {shape = [1000 : i32]} : tensor<1000xf32>
    %5 = migraphx.constant() {shape = [4 : i32, 2 : i32]} : tensor<4x2xi32>
    %6 = migraphx.constant() {shape = [1 : i32, 1 : i32, 512 : i32, 2048 : i32]} : tensor<1x1x512x2048xf32>
    %7 = migraphx.constant() {shape = [2048 : i32]} : tensor<2048xf32>
    %8 = migraphx.constant() {shape = [2048 : i32]} : tensor<2048xf32>
    %9 = migraphx.constant() {shape = [2048 : i32]} : tensor<2048xf32>
    %10 = migraphx.constant() {shape = [2048 : i32]} : tensor<2048xf32>
    %11 = migraphx.constant() {shape = [3 : i32, 3 : i32, 512 : i32, 512 : i32]} : tensor<3x3x512x512xf32>
    %12 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %13 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %14 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %15 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %16 = migraphx.constant() {shape = [1 : i32, 1 : i32, 2048 : i32, 512 : i32]} : tensor<1x1x2048x512xf32>
    %17 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %18 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %19 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %20 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %21 = migraphx.constant() {shape = [1 : i32, 1 : i32, 512 : i32, 2048 : i32]} : tensor<1x1x512x2048xf32>
    %22 = migraphx.constant() {shape = [2048 : i32]} : tensor<2048xf32>
    %23 = migraphx.constant() {shape = [2048 : i32]} : tensor<2048xf32>
    %24 = migraphx.constant() {shape = [2048 : i32]} : tensor<2048xf32>
    %25 = migraphx.constant() {shape = [2048 : i32]} : tensor<2048xf32>
    %26 = migraphx.constant() {shape = [3 : i32, 3 : i32, 512 : i32, 512 : i32]} : tensor<3x3x512x512xf32>
    %27 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %28 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %29 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %30 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %31 = migraphx.constant() {shape = [1 : i32, 1 : i32, 2048 : i32, 512 : i32]} : tensor<1x1x2048x512xf32>
    %32 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %33 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %34 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %35 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %36 = migraphx.constant() {shape = [1 : i32, 1 : i32, 512 : i32, 2048 : i32]} : tensor<1x1x512x2048xf32>
    %37 = migraphx.constant() {shape = [2048 : i32]} : tensor<2048xf32>
    %38 = migraphx.constant() {shape = [2048 : i32]} : tensor<2048xf32>
    %39 = migraphx.constant() {shape = [2048 : i32]} : tensor<2048xf32>
    %40 = migraphx.constant() {shape = [3 : i32, 3 : i32, 512 : i32, 512 : i32]} : tensor<3x3x512x512xf32>
    %41 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %42 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %43 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %44 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %45 = migraphx.constant() {shape = [1 : i32, 1 : i32, 1024 : i32, 512 : i32]} : tensor<1x1x1024x512xf32>
    %46 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %47 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %48 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %49 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %50 = migraphx.constant() {shape = [1 : i32, 1 : i32, 1024 : i32, 2048 : i32]} : tensor<1x1x1024x2048xf32>
    %51 = migraphx.constant() {shape = [2048 : i32]} : tensor<2048xf32>
    %52 = migraphx.constant() {shape = [2048 : i32]} : tensor<2048xf32>
    %53 = migraphx.constant() {shape = [2048 : i32]} : tensor<2048xf32>
    %54 = migraphx.constant() {shape = [2048 : i32]} : tensor<2048xf32>
    %55 = migraphx.constant() {shape = [2048 : i32]} : tensor<2048xf32>
    %56 = migraphx.constant() {shape = [1 : i32, 1 : i32, 256 : i32, 1024 : i32]} : tensor<1x1x256x1024xf32>
    %57 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %58 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %59 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %60 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %61 = migraphx.constant() {shape = [3 : i32, 3 : i32, 256 : i32, 256 : i32]} : tensor<3x3x256x256xf32>
    %62 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %63 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %64 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %65 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %66 = migraphx.constant() {shape = [1 : i32, 1 : i32, 1024 : i32, 256 : i32]} : tensor<1x1x1024x256xf32>
    %67 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %68 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %69 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %70 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %71 = migraphx.constant() {shape = [1 : i32, 1 : i32, 256 : i32, 1024 : i32]} : tensor<1x1x256x1024xf32>
    %72 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %73 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %74 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %75 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %76 = migraphx.constant() {shape = [3 : i32, 3 : i32, 256 : i32, 256 : i32]} : tensor<3x3x256x256xf32>
    %77 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %78 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %79 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %80 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %81 = migraphx.constant() {shape = [1 : i32, 1 : i32, 1024 : i32, 256 : i32]} : tensor<1x1x1024x256xf32>
    %82 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %83 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %84 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %85 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %86 = migraphx.constant() {shape = [1 : i32, 1 : i32, 256 : i32, 1024 : i32]} : tensor<1x1x256x1024xf32>
    %87 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %88 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %89 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %90 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %91 = migraphx.constant() {shape = [3 : i32, 3 : i32, 256 : i32, 256 : i32]} : tensor<3x3x256x256xf32>
    %92 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %93 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %94 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %95 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %96 = migraphx.constant() {shape = [1 : i32, 1 : i32, 1024 : i32, 256 : i32]} : tensor<1x1x1024x256xf32>
    %97 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %98 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %99 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %100 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %101 = migraphx.constant() {shape = [1 : i32, 1 : i32, 256 : i32, 1024 : i32]} : tensor<1x1x256x1024xf32>
    %102 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %103 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %104 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %105 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %106 = migraphx.constant() {shape = [3 : i32, 3 : i32, 256 : i32, 256 : i32]} : tensor<3x3x256x256xf32>
    %107 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %108 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %109 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %110 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %111 = migraphx.constant() {shape = [1 : i32, 1 : i32, 1024 : i32, 256 : i32]} : tensor<1x1x1024x256xf32>
    %112 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %113 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %114 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %115 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %116 = migraphx.constant() {shape = [1 : i32, 1 : i32, 256 : i32, 1024 : i32]} : tensor<1x1x256x1024xf32>
    %117 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %118 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %119 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %120 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %121 = migraphx.constant() {shape = [3 : i32, 3 : i32, 256 : i32, 256 : i32]} : tensor<3x3x256x256xf32>
    %122 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %123 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %124 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %125 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %126 = migraphx.constant() {shape = [1 : i32, 1 : i32, 1024 : i32, 256 : i32]} : tensor<1x1x1024x256xf32>
    %127 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %128 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %129 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %130 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %131 = migraphx.constant() {shape = [1 : i32, 1 : i32, 256 : i32, 1024 : i32]} : tensor<1x1x256x1024xf32>
    %132 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %133 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %134 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %135 = migraphx.constant() {shape = [3 : i32, 3 : i32, 256 : i32, 256 : i32]} : tensor<3x3x256x256xf32>
    %136 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %137 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %138 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %139 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %140 = migraphx.constant() {shape = [1 : i32, 1 : i32, 512 : i32, 256 : i32]} : tensor<1x1x512x256xf32>
    %141 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %142 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %143 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %144 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %145 = migraphx.constant() {shape = [1 : i32, 1 : i32, 512 : i32, 1024 : i32]} : tensor<1x1x512x1024xf32>
    %146 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %147 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %148 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %149 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %150 = migraphx.constant() {shape = [1024 : i32]} : tensor<1024xf32>
    %151 = migraphx.constant() {shape = [1 : i32, 1 : i32, 128 : i32, 512 : i32]} : tensor<1x1x128x512xf32>
    %152 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %153 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %154 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %155 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %156 = migraphx.constant() {shape = [3 : i32, 3 : i32, 128 : i32, 128 : i32]} : tensor<3x3x128x128xf32>
    %157 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %158 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %159 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %160 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %161 = migraphx.constant() {shape = [1 : i32, 1 : i32, 512 : i32, 128 : i32]} : tensor<1x1x512x128xf32>
    %162 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %163 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %164 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %165 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %166 = migraphx.constant() {shape = [1 : i32, 1 : i32, 128 : i32, 512 : i32]} : tensor<1x1x128x512xf32>
    %167 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %168 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %169 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %170 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %171 = migraphx.constant() {shape = [3 : i32, 3 : i32, 128 : i32, 128 : i32]} : tensor<3x3x128x128xf32>
    %172 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %173 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %174 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %175 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %176 = migraphx.constant() {shape = [1 : i32, 1 : i32, 512 : i32, 128 : i32]} : tensor<1x1x512x128xf32>
    %177 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %178 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %179 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %180 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %181 = migraphx.constant() {shape = [1 : i32, 1 : i32, 128 : i32, 512 : i32]} : tensor<1x1x128x512xf32>
    %182 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %183 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %184 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %185 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %186 = migraphx.constant() {shape = [3 : i32, 3 : i32, 128 : i32, 128 : i32]} : tensor<3x3x128x128xf32>
    %187 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %188 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %189 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %190 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %191 = migraphx.constant() {shape = [1 : i32, 1 : i32, 512 : i32, 128 : i32]} : tensor<1x1x512x128xf32>
    %192 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %193 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %194 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %195 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %196 = migraphx.constant() {shape = [1 : i32, 1 : i32, 128 : i32, 512 : i32]} : tensor<1x1x128x512xf32>
    %197 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %198 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %199 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %200 = migraphx.constant() {shape = [3 : i32, 3 : i32, 128 : i32, 128 : i32]} : tensor<3x3x128x128xf32>
    %201 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %202 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %203 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %204 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %205 = migraphx.constant() {shape = [1 : i32, 1 : i32, 256 : i32, 128 : i32]} : tensor<1x1x256x128xf32>
    %206 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %207 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %208 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %209 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %210 = migraphx.constant() {shape = [128 : i32]} : tensor<128xf32>
    %211 = migraphx.constant() {shape = [1 : i32, 1 : i32, 256 : i32, 512 : i32]} : tensor<1x1x256x512xf32>
    %212 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %213 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %214 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %215 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %216 = migraphx.constant() {shape = [512 : i32]} : tensor<512xf32>
    %217 = migraphx.constant() {shape = [1 : i32, 1 : i32, 64 : i32, 256 : i32]} : tensor<1x1x64x256xf32>
    %218 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %219 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %220 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %221 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %222 = migraphx.constant() {shape = [3 : i32, 3 : i32, 64 : i32, 64 : i32]} : tensor<3x3x64x64xf32>
    %223 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %224 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %225 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %226 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %227 = migraphx.constant() {shape = [1 : i32, 1 : i32, 256 : i32, 64 : i32]} : tensor<1x1x256x64xf32>
    %228 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %229 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %230 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %231 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %232 = migraphx.constant() {shape = [1 : i32, 1 : i32, 64 : i32, 256 : i32]} : tensor<1x1x64x256xf32>
    %233 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %234 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %235 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %236 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %237 = migraphx.constant() {shape = [3 : i32, 3 : i32, 64 : i32, 64 : i32]} : tensor<3x3x64x64xf32>
    %238 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %239 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %240 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %241 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %242 = migraphx.constant() {shape = [1 : i32, 1 : i32, 256 : i32, 64 : i32]} : tensor<1x1x256x64xf32>
    %243 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %244 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %245 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %246 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %247 = migraphx.constant() {shape = [1 : i32, 1 : i32, 64 : i32, 256 : i32]} : tensor<1x1x64x256xf32>
    %248 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %249 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %250 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %251 = migraphx.constant() {shape = [3 : i32, 3 : i32, 64 : i32, 64 : i32]} : tensor<3x3x64x64xf32>
    %252 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %253 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %254 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %255 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %256 = migraphx.constant() {shape = [1 : i32, 1 : i32, 64 : i32, 64 : i32]} : tensor<1x1x64x64xf32>
    %257 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %258 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %259 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %260 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %261 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %262 = migraphx.constant() {shape = [1 : i32, 1 : i32, 64 : i32, 256 : i32]} : tensor<1x1x64x256xf32>
    %263 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %264 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %265 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %266 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %267 = migraphx.constant() {shape = [256 : i32]} : tensor<256xf32>
    %268 = migraphx.constant() {shape = [4 : i32, 2 : i32]} : tensor<4x2xi32>
    %269 = migraphx.constant() {shape = [7 : i32, 7 : i32, 3 : i32, 64 : i32]} : tensor<7x7x3x64xf32>
    %270 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %271 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %272 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %273 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %274 = migraphx.constant() {shape = [64 : i32]} : tensor<64xf32>
    %275 = "tosa.pad"(%arg0, %268) : (tensor<100x224x224x3xf32>, tensor<4x2xi32>) -> tensor<100x230x230x3xf32>
    %276 = "tosa.transpose"(%269, %2) : (tensor<7x7x3x64xf32>, tensor<4xi32>) -> tensor<64x7x7x3xf32>
    %277 = "tosa.conv2d"(%275, %276, %270) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<100x230x230x3xf32>, tensor<64x7x7x3xf32>, tensor<64xf32>) -> tensor<100x112x112x64xf32>
    %278 = "tosa.reshape"(%274) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %279 = "tosa.sub"(%277, %278) : (tensor<100x112x112x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x112x112x64xf32>
    %280 = "tosa.add"(%273, %1) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    %281 = "tosa.rsqrt"(%280) : (tensor<64xf32>) -> tensor<64xf32>
    %282 = "tosa.reshape"(%281) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %283 = "tosa.mul"(%279, %282) {shift = 0 : i32} : (tensor<100x112x112x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x112x112x64xf32>
    %284 = "tosa.reshape"(%272) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %285 = "tosa.mul"(%283, %284) {shift = 0 : i32} : (tensor<100x112x112x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x112x112x64xf32>
    %286 = "tosa.reshape"(%271) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %287 = "tosa.add"(%285, %286) : (tensor<100x112x112x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x112x112x64xf32>
    %288 = "tosa.clamp"(%287) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x112x112x64xf32>) -> tensor<100x112x112x64xf32>
    %289 = "tosa.pad"(%288, %5) : (tensor<100x112x112x64xf32>, tensor<4x2xi32>) -> tensor<100x114x114x64xf32>
    %290 = "tosa.max_pool2d"(%289) {kernel = [3, 3], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<100x114x114x64xf32>) -> tensor<100x56x56x64xf32>
    %291 = "tosa.transpose"(%262, %2) : (tensor<1x1x64x256xf32>, tensor<4xi32>) -> tensor<256x1x1x64xf32>
    %292 = "tosa.conv2d"(%290, %291, %263) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x56x56x64xf32>, tensor<256x1x1x64xf32>, tensor<256xf32>) -> tensor<100x56x56x256xf32>
    %293 = "tosa.reshape"(%267) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %294 = "tosa.sub"(%292, %293) : (tensor<100x56x56x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x56x56x256xf32>
    %295 = "tosa.add"(%266, %1) : (tensor<256xf32>, tensor<1xf32>) -> tensor<256xf32>
    %296 = "tosa.rsqrt"(%295) : (tensor<256xf32>) -> tensor<256xf32>
    %297 = "tosa.reshape"(%296) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %298 = "tosa.mul"(%294, %297) {shift = 0 : i32} : (tensor<100x56x56x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x56x56x256xf32>
    %299 = "tosa.reshape"(%265) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %300 = "tosa.mul"(%298, %299) {shift = 0 : i32} : (tensor<100x56x56x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x56x56x256xf32>
    %301 = "tosa.reshape"(%264) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %302 = "tosa.add"(%300, %301) : (tensor<100x56x56x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x56x56x256xf32>
    %303 = "tosa.transpose"(%256, %2) : (tensor<1x1x64x64xf32>, tensor<4xi32>) -> tensor<64x1x1x64xf32>
    %304 = "tosa.conv2d"(%290, %303, %257) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x56x56x64xf32>, tensor<64x1x1x64xf32>, tensor<64xf32>) -> tensor<100x56x56x64xf32>
    %305 = "tosa.reshape"(%261) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %306 = "tosa.sub"(%304, %305) : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %307 = "tosa.add"(%260, %1) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    %308 = "tosa.rsqrt"(%307) : (tensor<64xf32>) -> tensor<64xf32>
    %309 = "tosa.reshape"(%308) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %310 = "tosa.mul"(%306, %309) {shift = 0 : i32} : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %311 = "tosa.reshape"(%259) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %312 = "tosa.mul"(%310, %311) {shift = 0 : i32} : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %313 = "tosa.reshape"(%258) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %314 = "tosa.add"(%312, %313) : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %315 = "tosa.clamp"(%314) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x56x56x64xf32>) -> tensor<100x56x56x64xf32>
    %316 = "tosa.transpose"(%251, %2) : (tensor<3x3x64x64xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %317 = "tosa.conv2d"(%315, %316, %257) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<100x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<100x56x56x64xf32>
    %318 = "tosa.reshape"(%255) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %319 = "tosa.sub"(%317, %318) : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %320 = "tosa.add"(%254, %1) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    %321 = "tosa.rsqrt"(%320) : (tensor<64xf32>) -> tensor<64xf32>
    %322 = "tosa.reshape"(%321) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %323 = "tosa.mul"(%319, %322) {shift = 0 : i32} : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %324 = "tosa.reshape"(%253) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %325 = "tosa.mul"(%323, %324) {shift = 0 : i32} : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %326 = "tosa.reshape"(%252) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %327 = "tosa.add"(%325, %326) : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %328 = "tosa.clamp"(%327) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x56x56x64xf32>) -> tensor<100x56x56x64xf32>
    %329 = "tosa.transpose"(%247, %2) : (tensor<1x1x64x256xf32>, tensor<4xi32>) -> tensor<256x1x1x64xf32>
    %330 = "tosa.conv2d"(%328, %329, %263) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x56x56x64xf32>, tensor<256x1x1x64xf32>, tensor<256xf32>) -> tensor<100x56x56x256xf32>
    %331 = "tosa.reshape"(%250) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %332 = "tosa.sub"(%330, %331) : (tensor<100x56x56x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x56x56x256xf32>
    %333 = "tosa.add"(%249, %1) : (tensor<256xf32>, tensor<1xf32>) -> tensor<256xf32>
    %334 = "tosa.rsqrt"(%333) : (tensor<256xf32>) -> tensor<256xf32>
    %335 = "tosa.reshape"(%334) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %336 = "tosa.mul"(%332, %335) {shift = 0 : i32} : (tensor<100x56x56x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x56x56x256xf32>
    %337 = "tosa.reshape"(%248) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %338 = "tosa.mul"(%336, %337) {shift = 0 : i32} : (tensor<100x56x56x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x56x56x256xf32>
    %339 = "tosa.reshape"(%264) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %340 = "tosa.add"(%338, %339) : (tensor<100x56x56x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x56x56x256xf32>
    %341 = "tosa.add"(%302, %340) : (tensor<100x56x56x256xf32>, tensor<100x56x56x256xf32>) -> tensor<100x56x56x256xf32>
    %342 = "tosa.clamp"(%341) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x56x56x256xf32>) -> tensor<100x56x56x256xf32>
    %343 = "tosa.transpose"(%242, %2) : (tensor<1x1x256x64xf32>, tensor<4xi32>) -> tensor<64x1x1x256xf32>
    %344 = "tosa.conv2d"(%342, %343, %257) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x56x56x256xf32>, tensor<64x1x1x256xf32>, tensor<64xf32>) -> tensor<100x56x56x64xf32>
    %345 = "tosa.reshape"(%246) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %346 = "tosa.sub"(%344, %345) : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %347 = "tosa.add"(%245, %1) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    %348 = "tosa.rsqrt"(%347) : (tensor<64xf32>) -> tensor<64xf32>
    %349 = "tosa.reshape"(%348) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %350 = "tosa.mul"(%346, %349) {shift = 0 : i32} : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %351 = "tosa.reshape"(%244) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %352 = "tosa.mul"(%350, %351) {shift = 0 : i32} : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %353 = "tosa.reshape"(%243) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %354 = "tosa.add"(%352, %353) : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %355 = "tosa.clamp"(%354) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x56x56x64xf32>) -> tensor<100x56x56x64xf32>
    %356 = "tosa.transpose"(%237, %2) : (tensor<3x3x64x64xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %357 = "tosa.conv2d"(%355, %356, %257) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<100x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<100x56x56x64xf32>
    %358 = "tosa.reshape"(%241) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %359 = "tosa.sub"(%357, %358) : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %360 = "tosa.add"(%240, %1) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    %361 = "tosa.rsqrt"(%360) : (tensor<64xf32>) -> tensor<64xf32>
    %362 = "tosa.reshape"(%361) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %363 = "tosa.mul"(%359, %362) {shift = 0 : i32} : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %364 = "tosa.reshape"(%239) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %365 = "tosa.mul"(%363, %364) {shift = 0 : i32} : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %366 = "tosa.reshape"(%238) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %367 = "tosa.add"(%365, %366) : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %368 = "tosa.clamp"(%367) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x56x56x64xf32>) -> tensor<100x56x56x64xf32>
    %369 = "tosa.transpose"(%232, %2) : (tensor<1x1x64x256xf32>, tensor<4xi32>) -> tensor<256x1x1x64xf32>
    %370 = "tosa.conv2d"(%368, %369, %263) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x56x56x64xf32>, tensor<256x1x1x64xf32>, tensor<256xf32>) -> tensor<100x56x56x256xf32>
    %371 = "tosa.reshape"(%236) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %372 = "tosa.sub"(%370, %371) : (tensor<100x56x56x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x56x56x256xf32>
    %373 = "tosa.add"(%235, %1) : (tensor<256xf32>, tensor<1xf32>) -> tensor<256xf32>
    %374 = "tosa.rsqrt"(%373) : (tensor<256xf32>) -> tensor<256xf32>
    %375 = "tosa.reshape"(%374) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %376 = "tosa.mul"(%372, %375) {shift = 0 : i32} : (tensor<100x56x56x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x56x56x256xf32>
    %377 = "tosa.reshape"(%234) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %378 = "tosa.mul"(%376, %377) {shift = 0 : i32} : (tensor<100x56x56x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x56x56x256xf32>
    %379 = "tosa.reshape"(%233) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %380 = "tosa.add"(%378, %379) : (tensor<100x56x56x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x56x56x256xf32>
    %381 = "tosa.add"(%342, %380) : (tensor<100x56x56x256xf32>, tensor<100x56x56x256xf32>) -> tensor<100x56x56x256xf32>
    %382 = "tosa.clamp"(%381) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x56x56x256xf32>) -> tensor<100x56x56x256xf32>
    %383 = "tosa.transpose"(%227, %2) : (tensor<1x1x256x64xf32>, tensor<4xi32>) -> tensor<64x1x1x256xf32>
    %384 = "tosa.conv2d"(%382, %383, %257) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x56x56x256xf32>, tensor<64x1x1x256xf32>, tensor<64xf32>) -> tensor<100x56x56x64xf32>
    %385 = "tosa.reshape"(%231) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %386 = "tosa.sub"(%384, %385) : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %387 = "tosa.add"(%230, %1) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    %388 = "tosa.rsqrt"(%387) : (tensor<64xf32>) -> tensor<64xf32>
    %389 = "tosa.reshape"(%388) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %390 = "tosa.mul"(%386, %389) {shift = 0 : i32} : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %391 = "tosa.reshape"(%229) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %392 = "tosa.mul"(%390, %391) {shift = 0 : i32} : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %393 = "tosa.reshape"(%228) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %394 = "tosa.add"(%392, %393) : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %395 = "tosa.clamp"(%394) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x56x56x64xf32>) -> tensor<100x56x56x64xf32>
    %396 = "tosa.transpose"(%222, %2) : (tensor<3x3x64x64xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %397 = "tosa.conv2d"(%395, %396, %257) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<100x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<100x56x56x64xf32>
    %398 = "tosa.reshape"(%226) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %399 = "tosa.sub"(%397, %398) : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %400 = "tosa.add"(%225, %1) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    %401 = "tosa.rsqrt"(%400) : (tensor<64xf32>) -> tensor<64xf32>
    %402 = "tosa.reshape"(%401) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %403 = "tosa.mul"(%399, %402) {shift = 0 : i32} : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %404 = "tosa.reshape"(%224) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %405 = "tosa.mul"(%403, %404) {shift = 0 : i32} : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %406 = "tosa.reshape"(%223) {new_shape = [1, 1, 1, 64]} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %407 = "tosa.add"(%405, %406) : (tensor<100x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<100x56x56x64xf32>
    %408 = "tosa.clamp"(%407) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x56x56x64xf32>) -> tensor<100x56x56x64xf32>
    %409 = "tosa.transpose"(%217, %2) : (tensor<1x1x64x256xf32>, tensor<4xi32>) -> tensor<256x1x1x64xf32>
    %410 = "tosa.conv2d"(%408, %409, %263) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x56x56x64xf32>, tensor<256x1x1x64xf32>, tensor<256xf32>) -> tensor<100x56x56x256xf32>
    %411 = "tosa.reshape"(%221) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %412 = "tosa.sub"(%410, %411) : (tensor<100x56x56x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x56x56x256xf32>
    %413 = "tosa.add"(%220, %1) : (tensor<256xf32>, tensor<1xf32>) -> tensor<256xf32>
    %414 = "tosa.rsqrt"(%413) : (tensor<256xf32>) -> tensor<256xf32>
    %415 = "tosa.reshape"(%414) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %416 = "tosa.mul"(%412, %415) {shift = 0 : i32} : (tensor<100x56x56x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x56x56x256xf32>
    %417 = "tosa.reshape"(%219) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %418 = "tosa.mul"(%416, %417) {shift = 0 : i32} : (tensor<100x56x56x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x56x56x256xf32>
    %419 = "tosa.reshape"(%218) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %420 = "tosa.add"(%418, %419) : (tensor<100x56x56x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x56x56x256xf32>
    %421 = "tosa.add"(%382, %420) : (tensor<100x56x56x256xf32>, tensor<100x56x56x256xf32>) -> tensor<100x56x56x256xf32>
    %422 = "tosa.clamp"(%421) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x56x56x256xf32>) -> tensor<100x56x56x256xf32>
    %423 = "tosa.transpose"(%211, %2) : (tensor<1x1x256x512xf32>, tensor<4xi32>) -> tensor<512x1x1x256xf32>
    %424 = "tosa.conv2d"(%422, %423, %212) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<100x56x56x256xf32>, tensor<512x1x1x256xf32>, tensor<512xf32>) -> tensor<100x28x28x512xf32>
    %425 = "tosa.reshape"(%216) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %426 = "tosa.sub"(%424, %425) : (tensor<100x28x28x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x28x28x512xf32>
    %427 = "tosa.add"(%215, %1) : (tensor<512xf32>, tensor<1xf32>) -> tensor<512xf32>
    %428 = "tosa.rsqrt"(%427) : (tensor<512xf32>) -> tensor<512xf32>
    %429 = "tosa.reshape"(%428) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %430 = "tosa.mul"(%426, %429) {shift = 0 : i32} : (tensor<100x28x28x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x28x28x512xf32>
    %431 = "tosa.reshape"(%214) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %432 = "tosa.mul"(%430, %431) {shift = 0 : i32} : (tensor<100x28x28x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x28x28x512xf32>
    %433 = "tosa.reshape"(%213) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %434 = "tosa.add"(%432, %433) : (tensor<100x28x28x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x28x28x512xf32>
    %435 = "tosa.transpose"(%205, %2) : (tensor<1x1x256x128xf32>, tensor<4xi32>) -> tensor<128x1x1x256xf32>
    %436 = "tosa.conv2d"(%422, %435, %206) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<100x56x56x256xf32>, tensor<128x1x1x256xf32>, tensor<128xf32>) -> tensor<100x28x28x128xf32>
    %437 = "tosa.reshape"(%210) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %438 = "tosa.sub"(%436, %437) : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %439 = "tosa.add"(%209, %1) : (tensor<128xf32>, tensor<1xf32>) -> tensor<128xf32>
    %440 = "tosa.rsqrt"(%439) : (tensor<128xf32>) -> tensor<128xf32>
    %441 = "tosa.reshape"(%440) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %442 = "tosa.mul"(%438, %441) {shift = 0 : i32} : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %443 = "tosa.reshape"(%208) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %444 = "tosa.mul"(%442, %443) {shift = 0 : i32} : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %445 = "tosa.reshape"(%207) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %446 = "tosa.add"(%444, %445) : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %447 = "tosa.clamp"(%446) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x28x28x128xf32>) -> tensor<100x28x28x128xf32>
    %448 = "tosa.transpose"(%200, %2) : (tensor<3x3x128x128xf32>, tensor<4xi32>) -> tensor<128x3x3x128xf32>
    %449 = "tosa.conv2d"(%447, %448, %206) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<100x28x28x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<100x28x28x128xf32>
    %450 = "tosa.reshape"(%204) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %451 = "tosa.sub"(%449, %450) : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %452 = "tosa.add"(%203, %1) : (tensor<128xf32>, tensor<1xf32>) -> tensor<128xf32>
    %453 = "tosa.rsqrt"(%452) : (tensor<128xf32>) -> tensor<128xf32>
    %454 = "tosa.reshape"(%453) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %455 = "tosa.mul"(%451, %454) {shift = 0 : i32} : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %456 = "tosa.reshape"(%202) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %457 = "tosa.mul"(%455, %456) {shift = 0 : i32} : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %458 = "tosa.reshape"(%201) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %459 = "tosa.add"(%457, %458) : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %460 = "tosa.clamp"(%459) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x28x28x128xf32>) -> tensor<100x28x28x128xf32>
    %461 = "tosa.transpose"(%196, %2) : (tensor<1x1x128x512xf32>, tensor<4xi32>) -> tensor<512x1x1x128xf32>
    %462 = "tosa.conv2d"(%460, %461, %212) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x28x28x128xf32>, tensor<512x1x1x128xf32>, tensor<512xf32>) -> tensor<100x28x28x512xf32>
    %463 = "tosa.reshape"(%199) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %464 = "tosa.sub"(%462, %463) : (tensor<100x28x28x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x28x28x512xf32>
    %465 = "tosa.add"(%198, %1) : (tensor<512xf32>, tensor<1xf32>) -> tensor<512xf32>
    %466 = "tosa.rsqrt"(%465) : (tensor<512xf32>) -> tensor<512xf32>
    %467 = "tosa.reshape"(%466) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %468 = "tosa.mul"(%464, %467) {shift = 0 : i32} : (tensor<100x28x28x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x28x28x512xf32>
    %469 = "tosa.reshape"(%197) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %470 = "tosa.mul"(%468, %469) {shift = 0 : i32} : (tensor<100x28x28x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x28x28x512xf32>
    %471 = "tosa.reshape"(%213) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %472 = "tosa.add"(%470, %471) : (tensor<100x28x28x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x28x28x512xf32>
    %473 = "tosa.add"(%434, %472) : (tensor<100x28x28x512xf32>, tensor<100x28x28x512xf32>) -> tensor<100x28x28x512xf32>
    %474 = "tosa.clamp"(%473) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x28x28x512xf32>) -> tensor<100x28x28x512xf32>
    %475 = "tosa.transpose"(%191, %2) : (tensor<1x1x512x128xf32>, tensor<4xi32>) -> tensor<128x1x1x512xf32>
    %476 = "tosa.conv2d"(%474, %475, %206) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x28x28x512xf32>, tensor<128x1x1x512xf32>, tensor<128xf32>) -> tensor<100x28x28x128xf32>
    %477 = "tosa.reshape"(%195) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %478 = "tosa.sub"(%476, %477) : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %479 = "tosa.add"(%194, %1) : (tensor<128xf32>, tensor<1xf32>) -> tensor<128xf32>
    %480 = "tosa.rsqrt"(%479) : (tensor<128xf32>) -> tensor<128xf32>
    %481 = "tosa.reshape"(%480) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %482 = "tosa.mul"(%478, %481) {shift = 0 : i32} : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %483 = "tosa.reshape"(%193) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %484 = "tosa.mul"(%482, %483) {shift = 0 : i32} : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %485 = "tosa.reshape"(%192) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %486 = "tosa.add"(%484, %485) : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %487 = "tosa.clamp"(%486) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x28x28x128xf32>) -> tensor<100x28x28x128xf32>
    %488 = "tosa.transpose"(%186, %2) : (tensor<3x3x128x128xf32>, tensor<4xi32>) -> tensor<128x3x3x128xf32>
    %489 = "tosa.conv2d"(%487, %488, %206) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<100x28x28x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<100x28x28x128xf32>
    %490 = "tosa.reshape"(%190) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %491 = "tosa.sub"(%489, %490) : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %492 = "tosa.add"(%189, %1) : (tensor<128xf32>, tensor<1xf32>) -> tensor<128xf32>
    %493 = "tosa.rsqrt"(%492) : (tensor<128xf32>) -> tensor<128xf32>
    %494 = "tosa.reshape"(%493) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %495 = "tosa.mul"(%491, %494) {shift = 0 : i32} : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %496 = "tosa.reshape"(%188) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %497 = "tosa.mul"(%495, %496) {shift = 0 : i32} : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %498 = "tosa.reshape"(%187) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %499 = "tosa.add"(%497, %498) : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %500 = "tosa.clamp"(%499) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x28x28x128xf32>) -> tensor<100x28x28x128xf32>
    %501 = "tosa.transpose"(%181, %2) : (tensor<1x1x128x512xf32>, tensor<4xi32>) -> tensor<512x1x1x128xf32>
    %502 = "tosa.conv2d"(%500, %501, %212) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x28x28x128xf32>, tensor<512x1x1x128xf32>, tensor<512xf32>) -> tensor<100x28x28x512xf32>
    %503 = "tosa.reshape"(%185) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %504 = "tosa.sub"(%502, %503) : (tensor<100x28x28x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x28x28x512xf32>
    %505 = "tosa.add"(%184, %1) : (tensor<512xf32>, tensor<1xf32>) -> tensor<512xf32>
    %506 = "tosa.rsqrt"(%505) : (tensor<512xf32>) -> tensor<512xf32>
    %507 = "tosa.reshape"(%506) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %508 = "tosa.mul"(%504, %507) {shift = 0 : i32} : (tensor<100x28x28x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x28x28x512xf32>
    %509 = "tosa.reshape"(%183) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %510 = "tosa.mul"(%508, %509) {shift = 0 : i32} : (tensor<100x28x28x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x28x28x512xf32>
    %511 = "tosa.reshape"(%182) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %512 = "tosa.add"(%510, %511) : (tensor<100x28x28x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x28x28x512xf32>
    %513 = "tosa.add"(%474, %512) : (tensor<100x28x28x512xf32>, tensor<100x28x28x512xf32>) -> tensor<100x28x28x512xf32>
    %514 = "tosa.clamp"(%513) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x28x28x512xf32>) -> tensor<100x28x28x512xf32>
    %515 = "tosa.transpose"(%176, %2) : (tensor<1x1x512x128xf32>, tensor<4xi32>) -> tensor<128x1x1x512xf32>
    %516 = "tosa.conv2d"(%514, %515, %206) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x28x28x512xf32>, tensor<128x1x1x512xf32>, tensor<128xf32>) -> tensor<100x28x28x128xf32>
    %517 = "tosa.reshape"(%180) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %518 = "tosa.sub"(%516, %517) : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %519 = "tosa.add"(%179, %1) : (tensor<128xf32>, tensor<1xf32>) -> tensor<128xf32>
    %520 = "tosa.rsqrt"(%519) : (tensor<128xf32>) -> tensor<128xf32>
    %521 = "tosa.reshape"(%520) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %522 = "tosa.mul"(%518, %521) {shift = 0 : i32} : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %523 = "tosa.reshape"(%178) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %524 = "tosa.mul"(%522, %523) {shift = 0 : i32} : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %525 = "tosa.reshape"(%177) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %526 = "tosa.add"(%524, %525) : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %527 = "tosa.clamp"(%526) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x28x28x128xf32>) -> tensor<100x28x28x128xf32>
    %528 = "tosa.transpose"(%171, %2) : (tensor<3x3x128x128xf32>, tensor<4xi32>) -> tensor<128x3x3x128xf32>
    %529 = "tosa.conv2d"(%527, %528, %206) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<100x28x28x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<100x28x28x128xf32>
    %530 = "tosa.reshape"(%175) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %531 = "tosa.sub"(%529, %530) : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %532 = "tosa.add"(%174, %1) : (tensor<128xf32>, tensor<1xf32>) -> tensor<128xf32>
    %533 = "tosa.rsqrt"(%532) : (tensor<128xf32>) -> tensor<128xf32>
    %534 = "tosa.reshape"(%533) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %535 = "tosa.mul"(%531, %534) {shift = 0 : i32} : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %536 = "tosa.reshape"(%173) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %537 = "tosa.mul"(%535, %536) {shift = 0 : i32} : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %538 = "tosa.reshape"(%172) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %539 = "tosa.add"(%537, %538) : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %540 = "tosa.clamp"(%539) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x28x28x128xf32>) -> tensor<100x28x28x128xf32>
    %541 = "tosa.transpose"(%166, %2) : (tensor<1x1x128x512xf32>, tensor<4xi32>) -> tensor<512x1x1x128xf32>
    %542 = "tosa.conv2d"(%540, %541, %212) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x28x28x128xf32>, tensor<512x1x1x128xf32>, tensor<512xf32>) -> tensor<100x28x28x512xf32>
    %543 = "tosa.reshape"(%170) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %544 = "tosa.sub"(%542, %543) : (tensor<100x28x28x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x28x28x512xf32>
    %545 = "tosa.add"(%169, %1) : (tensor<512xf32>, tensor<1xf32>) -> tensor<512xf32>
    %546 = "tosa.rsqrt"(%545) : (tensor<512xf32>) -> tensor<512xf32>
    %547 = "tosa.reshape"(%546) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %548 = "tosa.mul"(%544, %547) {shift = 0 : i32} : (tensor<100x28x28x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x28x28x512xf32>
    %549 = "tosa.reshape"(%168) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %550 = "tosa.mul"(%548, %549) {shift = 0 : i32} : (tensor<100x28x28x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x28x28x512xf32>
    %551 = "tosa.reshape"(%167) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %552 = "tosa.add"(%550, %551) : (tensor<100x28x28x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x28x28x512xf32>
    %553 = "tosa.add"(%514, %552) : (tensor<100x28x28x512xf32>, tensor<100x28x28x512xf32>) -> tensor<100x28x28x512xf32>
    %554 = "tosa.clamp"(%553) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x28x28x512xf32>) -> tensor<100x28x28x512xf32>
    %555 = "tosa.transpose"(%161, %2) : (tensor<1x1x512x128xf32>, tensor<4xi32>) -> tensor<128x1x1x512xf32>
    %556 = "tosa.conv2d"(%554, %555, %206) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x28x28x512xf32>, tensor<128x1x1x512xf32>, tensor<128xf32>) -> tensor<100x28x28x128xf32>
    %557 = "tosa.reshape"(%165) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %558 = "tosa.sub"(%556, %557) : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %559 = "tosa.add"(%164, %1) : (tensor<128xf32>, tensor<1xf32>) -> tensor<128xf32>
    %560 = "tosa.rsqrt"(%559) : (tensor<128xf32>) -> tensor<128xf32>
    %561 = "tosa.reshape"(%560) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %562 = "tosa.mul"(%558, %561) {shift = 0 : i32} : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %563 = "tosa.reshape"(%163) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %564 = "tosa.mul"(%562, %563) {shift = 0 : i32} : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %565 = "tosa.reshape"(%162) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %566 = "tosa.add"(%564, %565) : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %567 = "tosa.clamp"(%566) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x28x28x128xf32>) -> tensor<100x28x28x128xf32>
    %568 = "tosa.transpose"(%156, %2) : (tensor<3x3x128x128xf32>, tensor<4xi32>) -> tensor<128x3x3x128xf32>
    %569 = "tosa.conv2d"(%567, %568, %206) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<100x28x28x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<100x28x28x128xf32>
    %570 = "tosa.reshape"(%160) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %571 = "tosa.sub"(%569, %570) : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %572 = "tosa.add"(%159, %1) : (tensor<128xf32>, tensor<1xf32>) -> tensor<128xf32>
    %573 = "tosa.rsqrt"(%572) : (tensor<128xf32>) -> tensor<128xf32>
    %574 = "tosa.reshape"(%573) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %575 = "tosa.mul"(%571, %574) {shift = 0 : i32} : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %576 = "tosa.reshape"(%158) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %577 = "tosa.mul"(%575, %576) {shift = 0 : i32} : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %578 = "tosa.reshape"(%157) {new_shape = [1, 1, 1, 128]} : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %579 = "tosa.add"(%577, %578) : (tensor<100x28x28x128xf32>, tensor<1x1x1x128xf32>) -> tensor<100x28x28x128xf32>
    %580 = "tosa.clamp"(%579) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x28x28x128xf32>) -> tensor<100x28x28x128xf32>
    %581 = "tosa.transpose"(%151, %2) : (tensor<1x1x128x512xf32>, tensor<4xi32>) -> tensor<512x1x1x128xf32>
    %582 = "tosa.conv2d"(%580, %581, %212) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x28x28x128xf32>, tensor<512x1x1x128xf32>, tensor<512xf32>) -> tensor<100x28x28x512xf32>
    %583 = "tosa.reshape"(%155) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %584 = "tosa.sub"(%582, %583) : (tensor<100x28x28x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x28x28x512xf32>
    %585 = "tosa.add"(%154, %1) : (tensor<512xf32>, tensor<1xf32>) -> tensor<512xf32>
    %586 = "tosa.rsqrt"(%585) : (tensor<512xf32>) -> tensor<512xf32>
    %587 = "tosa.reshape"(%586) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %588 = "tosa.mul"(%584, %587) {shift = 0 : i32} : (tensor<100x28x28x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x28x28x512xf32>
    %589 = "tosa.reshape"(%153) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %590 = "tosa.mul"(%588, %589) {shift = 0 : i32} : (tensor<100x28x28x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x28x28x512xf32>
    %591 = "tosa.reshape"(%152) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %592 = "tosa.add"(%590, %591) : (tensor<100x28x28x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x28x28x512xf32>
    %593 = "tosa.add"(%554, %592) : (tensor<100x28x28x512xf32>, tensor<100x28x28x512xf32>) -> tensor<100x28x28x512xf32>
    %594 = "tosa.clamp"(%593) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x28x28x512xf32>) -> tensor<100x28x28x512xf32>
    %595 = "tosa.transpose"(%145, %2) : (tensor<1x1x512x1024xf32>, tensor<4xi32>) -> tensor<1024x1x1x512xf32>
    %596 = "tosa.conv2d"(%594, %595, %146) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<100x28x28x512xf32>, tensor<1024x1x1x512xf32>, tensor<1024xf32>) -> tensor<100x14x14x1024xf32>
    %597 = "tosa.reshape"(%150) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %598 = "tosa.sub"(%596, %597) : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %599 = "tosa.add"(%149, %1) : (tensor<1024xf32>, tensor<1xf32>) -> tensor<1024xf32>
    %600 = "tosa.rsqrt"(%599) : (tensor<1024xf32>) -> tensor<1024xf32>
    %601 = "tosa.reshape"(%600) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %602 = "tosa.mul"(%598, %601) {shift = 0 : i32} : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %603 = "tosa.reshape"(%148) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %604 = "tosa.mul"(%602, %603) {shift = 0 : i32} : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %605 = "tosa.reshape"(%147) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %606 = "tosa.add"(%604, %605) : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %607 = "tosa.transpose"(%140, %2) : (tensor<1x1x512x256xf32>, tensor<4xi32>) -> tensor<256x1x1x512xf32>
    %608 = "tosa.conv2d"(%594, %607, %263) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<100x28x28x512xf32>, tensor<256x1x1x512xf32>, tensor<256xf32>) -> tensor<100x14x14x256xf32>
    %609 = "tosa.reshape"(%144) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %610 = "tosa.sub"(%608, %609) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %611 = "tosa.add"(%143, %1) : (tensor<256xf32>, tensor<1xf32>) -> tensor<256xf32>
    %612 = "tosa.rsqrt"(%611) : (tensor<256xf32>) -> tensor<256xf32>
    %613 = "tosa.reshape"(%612) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %614 = "tosa.mul"(%610, %613) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %615 = "tosa.reshape"(%142) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %616 = "tosa.mul"(%614, %615) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %617 = "tosa.reshape"(%141) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %618 = "tosa.add"(%616, %617) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %619 = "tosa.clamp"(%618) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x14x14x256xf32>) -> tensor<100x14x14x256xf32>
    %620 = "tosa.transpose"(%135, %2) : (tensor<3x3x256x256xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %621 = "tosa.conv2d"(%619, %620, %263) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<100x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<100x14x14x256xf32>
    %622 = "tosa.reshape"(%139) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %623 = "tosa.sub"(%621, %622) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %624 = "tosa.add"(%138, %1) : (tensor<256xf32>, tensor<1xf32>) -> tensor<256xf32>
    %625 = "tosa.rsqrt"(%624) : (tensor<256xf32>) -> tensor<256xf32>
    %626 = "tosa.reshape"(%625) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %627 = "tosa.mul"(%623, %626) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %628 = "tosa.reshape"(%137) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %629 = "tosa.mul"(%627, %628) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %630 = "tosa.reshape"(%136) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %631 = "tosa.add"(%629, %630) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %632 = "tosa.clamp"(%631) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x14x14x256xf32>) -> tensor<100x14x14x256xf32>
    %633 = "tosa.transpose"(%131, %2) : (tensor<1x1x256x1024xf32>, tensor<4xi32>) -> tensor<1024x1x1x256xf32>
    %634 = "tosa.conv2d"(%632, %633, %146) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x14x14x256xf32>, tensor<1024x1x1x256xf32>, tensor<1024xf32>) -> tensor<100x14x14x1024xf32>
    %635 = "tosa.reshape"(%134) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %636 = "tosa.sub"(%634, %635) : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %637 = "tosa.add"(%133, %1) : (tensor<1024xf32>, tensor<1xf32>) -> tensor<1024xf32>
    %638 = "tosa.rsqrt"(%637) : (tensor<1024xf32>) -> tensor<1024xf32>
    %639 = "tosa.reshape"(%638) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %640 = "tosa.mul"(%636, %639) {shift = 0 : i32} : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %641 = "tosa.reshape"(%132) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %642 = "tosa.mul"(%640, %641) {shift = 0 : i32} : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %643 = "tosa.reshape"(%147) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %644 = "tosa.add"(%642, %643) : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %645 = "tosa.add"(%606, %644) : (tensor<100x14x14x1024xf32>, tensor<100x14x14x1024xf32>) -> tensor<100x14x14x1024xf32>
    %646 = "tosa.clamp"(%645) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x14x14x1024xf32>) -> tensor<100x14x14x1024xf32>
    %647 = "tosa.transpose"(%126, %2) : (tensor<1x1x1024x256xf32>, tensor<4xi32>) -> tensor<256x1x1x1024xf32>
    %648 = "tosa.conv2d"(%646, %647, %263) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x14x14x1024xf32>, tensor<256x1x1x1024xf32>, tensor<256xf32>) -> tensor<100x14x14x256xf32>
    %649 = "tosa.reshape"(%130) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %650 = "tosa.sub"(%648, %649) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %651 = "tosa.add"(%129, %1) : (tensor<256xf32>, tensor<1xf32>) -> tensor<256xf32>
    %652 = "tosa.rsqrt"(%651) : (tensor<256xf32>) -> tensor<256xf32>
    %653 = "tosa.reshape"(%652) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %654 = "tosa.mul"(%650, %653) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %655 = "tosa.reshape"(%128) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %656 = "tosa.mul"(%654, %655) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %657 = "tosa.reshape"(%127) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %658 = "tosa.add"(%656, %657) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %659 = "tosa.clamp"(%658) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x14x14x256xf32>) -> tensor<100x14x14x256xf32>
    %660 = "tosa.transpose"(%121, %2) : (tensor<3x3x256x256xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %661 = "tosa.conv2d"(%659, %660, %263) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<100x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<100x14x14x256xf32>
    %662 = "tosa.reshape"(%125) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %663 = "tosa.sub"(%661, %662) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %664 = "tosa.add"(%124, %1) : (tensor<256xf32>, tensor<1xf32>) -> tensor<256xf32>
    %665 = "tosa.rsqrt"(%664) : (tensor<256xf32>) -> tensor<256xf32>
    %666 = "tosa.reshape"(%665) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %667 = "tosa.mul"(%663, %666) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %668 = "tosa.reshape"(%123) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %669 = "tosa.mul"(%667, %668) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %670 = "tosa.reshape"(%122) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %671 = "tosa.add"(%669, %670) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %672 = "tosa.clamp"(%671) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x14x14x256xf32>) -> tensor<100x14x14x256xf32>
    %673 = "tosa.transpose"(%116, %2) : (tensor<1x1x256x1024xf32>, tensor<4xi32>) -> tensor<1024x1x1x256xf32>
    %674 = "tosa.conv2d"(%672, %673, %146) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x14x14x256xf32>, tensor<1024x1x1x256xf32>, tensor<1024xf32>) -> tensor<100x14x14x1024xf32>
    %675 = "tosa.reshape"(%120) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %676 = "tosa.sub"(%674, %675) : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %677 = "tosa.add"(%119, %1) : (tensor<1024xf32>, tensor<1xf32>) -> tensor<1024xf32>
    %678 = "tosa.rsqrt"(%677) : (tensor<1024xf32>) -> tensor<1024xf32>
    %679 = "tosa.reshape"(%678) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %680 = "tosa.mul"(%676, %679) {shift = 0 : i32} : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %681 = "tosa.reshape"(%118) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %682 = "tosa.mul"(%680, %681) {shift = 0 : i32} : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %683 = "tosa.reshape"(%117) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %684 = "tosa.add"(%682, %683) : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %685 = "tosa.add"(%646, %684) : (tensor<100x14x14x1024xf32>, tensor<100x14x14x1024xf32>) -> tensor<100x14x14x1024xf32>
    %686 = "tosa.clamp"(%685) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x14x14x1024xf32>) -> tensor<100x14x14x1024xf32>
    %687 = "tosa.transpose"(%111, %2) : (tensor<1x1x1024x256xf32>, tensor<4xi32>) -> tensor<256x1x1x1024xf32>
    %688 = "tosa.conv2d"(%686, %687, %263) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x14x14x1024xf32>, tensor<256x1x1x1024xf32>, tensor<256xf32>) -> tensor<100x14x14x256xf32>
    %689 = "tosa.reshape"(%115) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %690 = "tosa.sub"(%688, %689) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %691 = "tosa.add"(%114, %1) : (tensor<256xf32>, tensor<1xf32>) -> tensor<256xf32>
    %692 = "tosa.rsqrt"(%691) : (tensor<256xf32>) -> tensor<256xf32>
    %693 = "tosa.reshape"(%692) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %694 = "tosa.mul"(%690, %693) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %695 = "tosa.reshape"(%113) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %696 = "tosa.mul"(%694, %695) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %697 = "tosa.reshape"(%112) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %698 = "tosa.add"(%696, %697) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %699 = "tosa.clamp"(%698) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x14x14x256xf32>) -> tensor<100x14x14x256xf32>
    %700 = "tosa.transpose"(%106, %2) : (tensor<3x3x256x256xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %701 = "tosa.conv2d"(%699, %700, %263) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<100x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<100x14x14x256xf32>
    %702 = "tosa.reshape"(%110) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %703 = "tosa.sub"(%701, %702) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %704 = "tosa.add"(%109, %1) : (tensor<256xf32>, tensor<1xf32>) -> tensor<256xf32>
    %705 = "tosa.rsqrt"(%704) : (tensor<256xf32>) -> tensor<256xf32>
    %706 = "tosa.reshape"(%705) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %707 = "tosa.mul"(%703, %706) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %708 = "tosa.reshape"(%108) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %709 = "tosa.mul"(%707, %708) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %710 = "tosa.reshape"(%107) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %711 = "tosa.add"(%709, %710) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %712 = "tosa.clamp"(%711) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x14x14x256xf32>) -> tensor<100x14x14x256xf32>
    %713 = "tosa.transpose"(%101, %2) : (tensor<1x1x256x1024xf32>, tensor<4xi32>) -> tensor<1024x1x1x256xf32>
    %714 = "tosa.conv2d"(%712, %713, %146) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x14x14x256xf32>, tensor<1024x1x1x256xf32>, tensor<1024xf32>) -> tensor<100x14x14x1024xf32>
    %715 = "tosa.reshape"(%105) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %716 = "tosa.sub"(%714, %715) : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %717 = "tosa.add"(%104, %1) : (tensor<1024xf32>, tensor<1xf32>) -> tensor<1024xf32>
    %718 = "tosa.rsqrt"(%717) : (tensor<1024xf32>) -> tensor<1024xf32>
    %719 = "tosa.reshape"(%718) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %720 = "tosa.mul"(%716, %719) {shift = 0 : i32} : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %721 = "tosa.reshape"(%103) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %722 = "tosa.mul"(%720, %721) {shift = 0 : i32} : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %723 = "tosa.reshape"(%102) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %724 = "tosa.add"(%722, %723) : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %725 = "tosa.add"(%686, %724) : (tensor<100x14x14x1024xf32>, tensor<100x14x14x1024xf32>) -> tensor<100x14x14x1024xf32>
    %726 = "tosa.clamp"(%725) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x14x14x1024xf32>) -> tensor<100x14x14x1024xf32>
    %727 = "tosa.transpose"(%96, %2) : (tensor<1x1x1024x256xf32>, tensor<4xi32>) -> tensor<256x1x1x1024xf32>
    %728 = "tosa.conv2d"(%726, %727, %263) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x14x14x1024xf32>, tensor<256x1x1x1024xf32>, tensor<256xf32>) -> tensor<100x14x14x256xf32>
    %729 = "tosa.reshape"(%100) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %730 = "tosa.sub"(%728, %729) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %731 = "tosa.add"(%99, %1) : (tensor<256xf32>, tensor<1xf32>) -> tensor<256xf32>
    %732 = "tosa.rsqrt"(%731) : (tensor<256xf32>) -> tensor<256xf32>
    %733 = "tosa.reshape"(%732) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %734 = "tosa.mul"(%730, %733) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %735 = "tosa.reshape"(%98) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %736 = "tosa.mul"(%734, %735) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %737 = "tosa.reshape"(%97) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %738 = "tosa.add"(%736, %737) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %739 = "tosa.clamp"(%738) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x14x14x256xf32>) -> tensor<100x14x14x256xf32>
    %740 = "tosa.transpose"(%91, %2) : (tensor<3x3x256x256xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %741 = "tosa.conv2d"(%739, %740, %263) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<100x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<100x14x14x256xf32>
    %742 = "tosa.reshape"(%95) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %743 = "tosa.sub"(%741, %742) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %744 = "tosa.add"(%94, %1) : (tensor<256xf32>, tensor<1xf32>) -> tensor<256xf32>
    %745 = "tosa.rsqrt"(%744) : (tensor<256xf32>) -> tensor<256xf32>
    %746 = "tosa.reshape"(%745) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %747 = "tosa.mul"(%743, %746) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %748 = "tosa.reshape"(%93) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %749 = "tosa.mul"(%747, %748) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %750 = "tosa.reshape"(%92) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %751 = "tosa.add"(%749, %750) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %752 = "tosa.clamp"(%751) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x14x14x256xf32>) -> tensor<100x14x14x256xf32>
    %753 = "tosa.transpose"(%86, %2) : (tensor<1x1x256x1024xf32>, tensor<4xi32>) -> tensor<1024x1x1x256xf32>
    %754 = "tosa.conv2d"(%752, %753, %146) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x14x14x256xf32>, tensor<1024x1x1x256xf32>, tensor<1024xf32>) -> tensor<100x14x14x1024xf32>
    %755 = "tosa.reshape"(%90) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %756 = "tosa.sub"(%754, %755) : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %757 = "tosa.add"(%89, %1) : (tensor<1024xf32>, tensor<1xf32>) -> tensor<1024xf32>
    %758 = "tosa.rsqrt"(%757) : (tensor<1024xf32>) -> tensor<1024xf32>
    %759 = "tosa.reshape"(%758) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %760 = "tosa.mul"(%756, %759) {shift = 0 : i32} : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %761 = "tosa.reshape"(%88) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %762 = "tosa.mul"(%760, %761) {shift = 0 : i32} : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %763 = "tosa.reshape"(%87) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %764 = "tosa.add"(%762, %763) : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %765 = "tosa.add"(%726, %764) : (tensor<100x14x14x1024xf32>, tensor<100x14x14x1024xf32>) -> tensor<100x14x14x1024xf32>
    %766 = "tosa.clamp"(%765) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x14x14x1024xf32>) -> tensor<100x14x14x1024xf32>
    %767 = "tosa.transpose"(%81, %2) : (tensor<1x1x1024x256xf32>, tensor<4xi32>) -> tensor<256x1x1x1024xf32>
    %768 = "tosa.conv2d"(%766, %767, %263) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x14x14x1024xf32>, tensor<256x1x1x1024xf32>, tensor<256xf32>) -> tensor<100x14x14x256xf32>
    %769 = "tosa.reshape"(%85) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %770 = "tosa.sub"(%768, %769) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %771 = "tosa.add"(%84, %1) : (tensor<256xf32>, tensor<1xf32>) -> tensor<256xf32>
    %772 = "tosa.rsqrt"(%771) : (tensor<256xf32>) -> tensor<256xf32>
    %773 = "tosa.reshape"(%772) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %774 = "tosa.mul"(%770, %773) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %775 = "tosa.reshape"(%83) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %776 = "tosa.mul"(%774, %775) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %777 = "tosa.reshape"(%82) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %778 = "tosa.add"(%776, %777) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %779 = "tosa.clamp"(%778) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x14x14x256xf32>) -> tensor<100x14x14x256xf32>
    %780 = "tosa.transpose"(%76, %2) : (tensor<3x3x256x256xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %781 = "tosa.conv2d"(%779, %780, %263) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<100x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<100x14x14x256xf32>
    %782 = "tosa.reshape"(%80) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %783 = "tosa.sub"(%781, %782) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %784 = "tosa.add"(%79, %1) : (tensor<256xf32>, tensor<1xf32>) -> tensor<256xf32>
    %785 = "tosa.rsqrt"(%784) : (tensor<256xf32>) -> tensor<256xf32>
    %786 = "tosa.reshape"(%785) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %787 = "tosa.mul"(%783, %786) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %788 = "tosa.reshape"(%78) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %789 = "tosa.mul"(%787, %788) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %790 = "tosa.reshape"(%77) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %791 = "tosa.add"(%789, %790) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %792 = "tosa.clamp"(%791) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x14x14x256xf32>) -> tensor<100x14x14x256xf32>
    %793 = "tosa.transpose"(%71, %2) : (tensor<1x1x256x1024xf32>, tensor<4xi32>) -> tensor<1024x1x1x256xf32>
    %794 = "tosa.conv2d"(%792, %793, %146) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x14x14x256xf32>, tensor<1024x1x1x256xf32>, tensor<1024xf32>) -> tensor<100x14x14x1024xf32>
    %795 = "tosa.reshape"(%75) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %796 = "tosa.sub"(%794, %795) : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %797 = "tosa.add"(%74, %1) : (tensor<1024xf32>, tensor<1xf32>) -> tensor<1024xf32>
    %798 = "tosa.rsqrt"(%797) : (tensor<1024xf32>) -> tensor<1024xf32>
    %799 = "tosa.reshape"(%798) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %800 = "tosa.mul"(%796, %799) {shift = 0 : i32} : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %801 = "tosa.reshape"(%73) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %802 = "tosa.mul"(%800, %801) {shift = 0 : i32} : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %803 = "tosa.reshape"(%72) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %804 = "tosa.add"(%802, %803) : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %805 = "tosa.add"(%766, %804) : (tensor<100x14x14x1024xf32>, tensor<100x14x14x1024xf32>) -> tensor<100x14x14x1024xf32>
    %806 = "tosa.clamp"(%805) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x14x14x1024xf32>) -> tensor<100x14x14x1024xf32>
    %807 = "tosa.transpose"(%66, %2) : (tensor<1x1x1024x256xf32>, tensor<4xi32>) -> tensor<256x1x1x1024xf32>
    %808 = "tosa.conv2d"(%806, %807, %263) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x14x14x1024xf32>, tensor<256x1x1x1024xf32>, tensor<256xf32>) -> tensor<100x14x14x256xf32>
    %809 = "tosa.reshape"(%70) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %810 = "tosa.sub"(%808, %809) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %811 = "tosa.add"(%69, %1) : (tensor<256xf32>, tensor<1xf32>) -> tensor<256xf32>
    %812 = "tosa.rsqrt"(%811) : (tensor<256xf32>) -> tensor<256xf32>
    %813 = "tosa.reshape"(%812) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %814 = "tosa.mul"(%810, %813) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %815 = "tosa.reshape"(%68) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %816 = "tosa.mul"(%814, %815) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %817 = "tosa.reshape"(%67) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %818 = "tosa.add"(%816, %817) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %819 = "tosa.clamp"(%818) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x14x14x256xf32>) -> tensor<100x14x14x256xf32>
    %820 = "tosa.transpose"(%61, %2) : (tensor<3x3x256x256xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %821 = "tosa.conv2d"(%819, %820, %263) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<100x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<100x14x14x256xf32>
    %822 = "tosa.reshape"(%65) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %823 = "tosa.sub"(%821, %822) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %824 = "tosa.add"(%64, %1) : (tensor<256xf32>, tensor<1xf32>) -> tensor<256xf32>
    %825 = "tosa.rsqrt"(%824) : (tensor<256xf32>) -> tensor<256xf32>
    %826 = "tosa.reshape"(%825) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %827 = "tosa.mul"(%823, %826) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %828 = "tosa.reshape"(%63) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %829 = "tosa.mul"(%827, %828) {shift = 0 : i32} : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %830 = "tosa.reshape"(%62) {new_shape = [1, 1, 1, 256]} : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %831 = "tosa.add"(%829, %830) : (tensor<100x14x14x256xf32>, tensor<1x1x1x256xf32>) -> tensor<100x14x14x256xf32>
    %832 = "tosa.clamp"(%831) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x14x14x256xf32>) -> tensor<100x14x14x256xf32>
    %833 = "tosa.transpose"(%56, %2) : (tensor<1x1x256x1024xf32>, tensor<4xi32>) -> tensor<1024x1x1x256xf32>
    %834 = "tosa.conv2d"(%832, %833, %146) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x14x14x256xf32>, tensor<1024x1x1x256xf32>, tensor<1024xf32>) -> tensor<100x14x14x1024xf32>
    %835 = "tosa.reshape"(%60) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %836 = "tosa.sub"(%834, %835) : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %837 = "tosa.add"(%59, %1) : (tensor<1024xf32>, tensor<1xf32>) -> tensor<1024xf32>
    %838 = "tosa.rsqrt"(%837) : (tensor<1024xf32>) -> tensor<1024xf32>
    %839 = "tosa.reshape"(%838) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %840 = "tosa.mul"(%836, %839) {shift = 0 : i32} : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %841 = "tosa.reshape"(%58) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %842 = "tosa.mul"(%840, %841) {shift = 0 : i32} : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %843 = "tosa.reshape"(%57) {new_shape = [1, 1, 1, 1024]} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %844 = "tosa.add"(%842, %843) : (tensor<100x14x14x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<100x14x14x1024xf32>
    %845 = "tosa.add"(%806, %844) : (tensor<100x14x14x1024xf32>, tensor<100x14x14x1024xf32>) -> tensor<100x14x14x1024xf32>
    %846 = "tosa.clamp"(%845) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x14x14x1024xf32>) -> tensor<100x14x14x1024xf32>
    %847 = "tosa.transpose"(%50, %2) : (tensor<1x1x1024x2048xf32>, tensor<4xi32>) -> tensor<2048x1x1x1024xf32>
    %848 = "tosa.conv2d"(%846, %847, %51) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<100x14x14x1024xf32>, tensor<2048x1x1x1024xf32>, tensor<2048xf32>) -> tensor<100x7x7x2048xf32>
    %849 = "tosa.reshape"(%55) {new_shape = [1, 1, 1, 2048]} : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %850 = "tosa.sub"(%848, %849) : (tensor<100x7x7x2048xf32>, tensor<1x1x1x2048xf32>) -> tensor<100x7x7x2048xf32>
    %851 = "tosa.add"(%54, %1) : (tensor<2048xf32>, tensor<1xf32>) -> tensor<2048xf32>
    %852 = "tosa.rsqrt"(%851) : (tensor<2048xf32>) -> tensor<2048xf32>
    %853 = "tosa.reshape"(%852) {new_shape = [1, 1, 1, 2048]} : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %854 = "tosa.mul"(%850, %853) {shift = 0 : i32} : (tensor<100x7x7x2048xf32>, tensor<1x1x1x2048xf32>) -> tensor<100x7x7x2048xf32>
    %855 = "tosa.reshape"(%53) {new_shape = [1, 1, 1, 2048]} : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %856 = "tosa.mul"(%854, %855) {shift = 0 : i32} : (tensor<100x7x7x2048xf32>, tensor<1x1x1x2048xf32>) -> tensor<100x7x7x2048xf32>
    %857 = "tosa.reshape"(%52) {new_shape = [1, 1, 1, 2048]} : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %858 = "tosa.add"(%856, %857) : (tensor<100x7x7x2048xf32>, tensor<1x1x1x2048xf32>) -> tensor<100x7x7x2048xf32>
    %859 = "tosa.transpose"(%45, %2) : (tensor<1x1x1024x512xf32>, tensor<4xi32>) -> tensor<512x1x1x1024xf32>
    %860 = "tosa.conv2d"(%846, %859, %212) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<100x14x14x1024xf32>, tensor<512x1x1x1024xf32>, tensor<512xf32>) -> tensor<100x7x7x512xf32>
    %861 = "tosa.reshape"(%49) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %862 = "tosa.sub"(%860, %861) : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %863 = "tosa.add"(%48, %1) : (tensor<512xf32>, tensor<1xf32>) -> tensor<512xf32>
    %864 = "tosa.rsqrt"(%863) : (tensor<512xf32>) -> tensor<512xf32>
    %865 = "tosa.reshape"(%864) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %866 = "tosa.mul"(%862, %865) {shift = 0 : i32} : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %867 = "tosa.reshape"(%47) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %868 = "tosa.mul"(%866, %867) {shift = 0 : i32} : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %869 = "tosa.reshape"(%46) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %870 = "tosa.add"(%868, %869) : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %871 = "tosa.clamp"(%870) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x7x7x512xf32>) -> tensor<100x7x7x512xf32>
    %872 = "tosa.transpose"(%40, %2) : (tensor<3x3x512x512xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %873 = "tosa.conv2d"(%871, %872, %212) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<100x7x7x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<100x7x7x512xf32>
    %874 = "tosa.reshape"(%44) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %875 = "tosa.sub"(%873, %874) : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %876 = "tosa.add"(%43, %1) : (tensor<512xf32>, tensor<1xf32>) -> tensor<512xf32>
    %877 = "tosa.rsqrt"(%876) : (tensor<512xf32>) -> tensor<512xf32>
    %878 = "tosa.reshape"(%877) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %879 = "tosa.mul"(%875, %878) {shift = 0 : i32} : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %880 = "tosa.reshape"(%42) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %881 = "tosa.mul"(%879, %880) {shift = 0 : i32} : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %882 = "tosa.reshape"(%41) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %883 = "tosa.add"(%881, %882) : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %884 = "tosa.clamp"(%883) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x7x7x512xf32>) -> tensor<100x7x7x512xf32>
    %885 = "tosa.transpose"(%36, %2) : (tensor<1x1x512x2048xf32>, tensor<4xi32>) -> tensor<2048x1x1x512xf32>
    %886 = "tosa.conv2d"(%884, %885, %51) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x7x7x512xf32>, tensor<2048x1x1x512xf32>, tensor<2048xf32>) -> tensor<100x7x7x2048xf32>
    %887 = "tosa.reshape"(%39) {new_shape = [1, 1, 1, 2048]} : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %888 = "tosa.sub"(%886, %887) : (tensor<100x7x7x2048xf32>, tensor<1x1x1x2048xf32>) -> tensor<100x7x7x2048xf32>
    %889 = "tosa.add"(%38, %1) : (tensor<2048xf32>, tensor<1xf32>) -> tensor<2048xf32>
    %890 = "tosa.rsqrt"(%889) : (tensor<2048xf32>) -> tensor<2048xf32>
    %891 = "tosa.reshape"(%890) {new_shape = [1, 1, 1, 2048]} : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %892 = "tosa.mul"(%888, %891) {shift = 0 : i32} : (tensor<100x7x7x2048xf32>, tensor<1x1x1x2048xf32>) -> tensor<100x7x7x2048xf32>
    %893 = "tosa.reshape"(%37) {new_shape = [1, 1, 1, 2048]} : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %894 = "tosa.mul"(%892, %893) {shift = 0 : i32} : (tensor<100x7x7x2048xf32>, tensor<1x1x1x2048xf32>) -> tensor<100x7x7x2048xf32>
    %895 = "tosa.reshape"(%52) {new_shape = [1, 1, 1, 2048]} : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %896 = "tosa.add"(%894, %895) : (tensor<100x7x7x2048xf32>, tensor<1x1x1x2048xf32>) -> tensor<100x7x7x2048xf32>
    %897 = "tosa.add"(%858, %896) : (tensor<100x7x7x2048xf32>, tensor<100x7x7x2048xf32>) -> tensor<100x7x7x2048xf32>
    %898 = "tosa.clamp"(%897) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x7x7x2048xf32>) -> tensor<100x7x7x2048xf32>
    %899 = "tosa.transpose"(%31, %2) : (tensor<1x1x2048x512xf32>, tensor<4xi32>) -> tensor<512x1x1x2048xf32>
    %900 = "tosa.conv2d"(%898, %899, %212) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x7x7x2048xf32>, tensor<512x1x1x2048xf32>, tensor<512xf32>) -> tensor<100x7x7x512xf32>
    %901 = "tosa.reshape"(%35) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %902 = "tosa.sub"(%900, %901) : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %903 = "tosa.add"(%34, %1) : (tensor<512xf32>, tensor<1xf32>) -> tensor<512xf32>
    %904 = "tosa.rsqrt"(%903) : (tensor<512xf32>) -> tensor<512xf32>
    %905 = "tosa.reshape"(%904) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %906 = "tosa.mul"(%902, %905) {shift = 0 : i32} : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %907 = "tosa.reshape"(%33) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %908 = "tosa.mul"(%906, %907) {shift = 0 : i32} : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %909 = "tosa.reshape"(%32) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %910 = "tosa.add"(%908, %909) : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %911 = "tosa.clamp"(%910) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x7x7x512xf32>) -> tensor<100x7x7x512xf32>
    %912 = "tosa.transpose"(%26, %2) : (tensor<3x3x512x512xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %913 = "tosa.conv2d"(%911, %912, %212) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<100x7x7x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<100x7x7x512xf32>
    %914 = "tosa.reshape"(%30) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %915 = "tosa.sub"(%913, %914) : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %916 = "tosa.add"(%29, %1) : (tensor<512xf32>, tensor<1xf32>) -> tensor<512xf32>
    %917 = "tosa.rsqrt"(%916) : (tensor<512xf32>) -> tensor<512xf32>
    %918 = "tosa.reshape"(%917) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %919 = "tosa.mul"(%915, %918) {shift = 0 : i32} : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %920 = "tosa.reshape"(%28) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %921 = "tosa.mul"(%919, %920) {shift = 0 : i32} : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %922 = "tosa.reshape"(%27) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %923 = "tosa.add"(%921, %922) : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %924 = "tosa.clamp"(%923) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x7x7x512xf32>) -> tensor<100x7x7x512xf32>
    %925 = "tosa.transpose"(%21, %2) : (tensor<1x1x512x2048xf32>, tensor<4xi32>) -> tensor<2048x1x1x512xf32>
    %926 = "tosa.conv2d"(%924, %925, %51) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x7x7x512xf32>, tensor<2048x1x1x512xf32>, tensor<2048xf32>) -> tensor<100x7x7x2048xf32>
    %927 = "tosa.reshape"(%25) {new_shape = [1, 1, 1, 2048]} : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %928 = "tosa.sub"(%926, %927) : (tensor<100x7x7x2048xf32>, tensor<1x1x1x2048xf32>) -> tensor<100x7x7x2048xf32>
    %929 = "tosa.add"(%24, %1) : (tensor<2048xf32>, tensor<1xf32>) -> tensor<2048xf32>
    %930 = "tosa.rsqrt"(%929) : (tensor<2048xf32>) -> tensor<2048xf32>
    %931 = "tosa.reshape"(%930) {new_shape = [1, 1, 1, 2048]} : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %932 = "tosa.mul"(%928, %931) {shift = 0 : i32} : (tensor<100x7x7x2048xf32>, tensor<1x1x1x2048xf32>) -> tensor<100x7x7x2048xf32>
    %933 = "tosa.reshape"(%23) {new_shape = [1, 1, 1, 2048]} : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %934 = "tosa.mul"(%932, %933) {shift = 0 : i32} : (tensor<100x7x7x2048xf32>, tensor<1x1x1x2048xf32>) -> tensor<100x7x7x2048xf32>
    %935 = "tosa.reshape"(%22) {new_shape = [1, 1, 1, 2048]} : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %936 = "tosa.add"(%934, %935) : (tensor<100x7x7x2048xf32>, tensor<1x1x1x2048xf32>) -> tensor<100x7x7x2048xf32>
    %937 = "tosa.add"(%898, %936) : (tensor<100x7x7x2048xf32>, tensor<100x7x7x2048xf32>) -> tensor<100x7x7x2048xf32>
    %938 = "tosa.clamp"(%937) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x7x7x2048xf32>) -> tensor<100x7x7x2048xf32>
    %939 = "tosa.transpose"(%16, %2) : (tensor<1x1x2048x512xf32>, tensor<4xi32>) -> tensor<512x1x1x2048xf32>
    %940 = "tosa.conv2d"(%938, %939, %212) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x7x7x2048xf32>, tensor<512x1x1x2048xf32>, tensor<512xf32>) -> tensor<100x7x7x512xf32>
    %941 = "tosa.reshape"(%20) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %942 = "tosa.sub"(%940, %941) : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %943 = "tosa.add"(%19, %1) : (tensor<512xf32>, tensor<1xf32>) -> tensor<512xf32>
    %944 = "tosa.rsqrt"(%943) : (tensor<512xf32>) -> tensor<512xf32>
    %945 = "tosa.reshape"(%944) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %946 = "tosa.mul"(%942, %945) {shift = 0 : i32} : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %947 = "tosa.reshape"(%18) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %948 = "tosa.mul"(%946, %947) {shift = 0 : i32} : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %949 = "tosa.reshape"(%17) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %950 = "tosa.add"(%948, %949) : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %951 = "tosa.clamp"(%950) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x7x7x512xf32>) -> tensor<100x7x7x512xf32>
    %952 = "tosa.transpose"(%11, %2) : (tensor<3x3x512x512xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %953 = "tosa.conv2d"(%951, %952, %212) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<100x7x7x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<100x7x7x512xf32>
    %954 = "tosa.reshape"(%15) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %955 = "tosa.sub"(%953, %954) : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %956 = "tosa.add"(%14, %1) : (tensor<512xf32>, tensor<1xf32>) -> tensor<512xf32>
    %957 = "tosa.rsqrt"(%956) : (tensor<512xf32>) -> tensor<512xf32>
    %958 = "tosa.reshape"(%957) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %959 = "tosa.mul"(%955, %958) {shift = 0 : i32} : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %960 = "tosa.reshape"(%13) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %961 = "tosa.mul"(%959, %960) {shift = 0 : i32} : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %962 = "tosa.reshape"(%12) {new_shape = [1, 1, 1, 512]} : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %963 = "tosa.add"(%961, %962) : (tensor<100x7x7x512xf32>, tensor<1x1x1x512xf32>) -> tensor<100x7x7x512xf32>
    %964 = "tosa.clamp"(%963) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x7x7x512xf32>) -> tensor<100x7x7x512xf32>
    %965 = "tosa.transpose"(%6, %2) : (tensor<1x1x512x2048xf32>, tensor<4xi32>) -> tensor<2048x1x1x512xf32>
    %966 = "tosa.conv2d"(%964, %965, %51) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<100x7x7x512xf32>, tensor<2048x1x1x512xf32>, tensor<2048xf32>) -> tensor<100x7x7x2048xf32>
    %967 = "tosa.reshape"(%10) {new_shape = [1, 1, 1, 2048]} : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %968 = "tosa.sub"(%966, %967) : (tensor<100x7x7x2048xf32>, tensor<1x1x1x2048xf32>) -> tensor<100x7x7x2048xf32>
    %969 = "tosa.add"(%9, %1) : (tensor<2048xf32>, tensor<1xf32>) -> tensor<2048xf32>
    %970 = "tosa.rsqrt"(%969) : (tensor<2048xf32>) -> tensor<2048xf32>
    %971 = "tosa.reshape"(%970) {new_shape = [1, 1, 1, 2048]} : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %972 = "tosa.mul"(%968, %971) {shift = 0 : i32} : (tensor<100x7x7x2048xf32>, tensor<1x1x1x2048xf32>) -> tensor<100x7x7x2048xf32>
    %973 = "tosa.reshape"(%8) {new_shape = [1, 1, 1, 2048]} : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %974 = "tosa.mul"(%972, %973) {shift = 0 : i32} : (tensor<100x7x7x2048xf32>, tensor<1x1x1x2048xf32>) -> tensor<100x7x7x2048xf32>
    %975 = "tosa.reshape"(%7) {new_shape = [1, 1, 1, 2048]} : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %976 = "tosa.add"(%974, %975) : (tensor<100x7x7x2048xf32>, tensor<1x1x1x2048xf32>) -> tensor<100x7x7x2048xf32>
    %977 = "tosa.add"(%938, %976) : (tensor<100x7x7x2048xf32>, tensor<100x7x7x2048xf32>) -> tensor<100x7x7x2048xf32>
    %978 = "tosa.clamp"(%977) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<100x7x7x2048xf32>) -> tensor<100x7x7x2048xf32>
    %979 = "tosa.reduce_sum"(%978) {axis = 1 : i64} : (tensor<100x7x7x2048xf32>) -> tensor<100x1x7x2048xf32>
    %980 = "tosa.reduce_sum"(%979) {axis = 2 : i64} : (tensor<100x1x7x2048xf32>) -> tensor<100x1x1x2048xf32>
    %981 = "tosa.reshape"(%980) {new_shape = [100, 2048]} : (tensor<100x1x1x2048xf32>) -> tensor<100x2048xf32>
    %982 = "tosa.reshape"(%0) {new_shape = [1, 1]} : (tensor<f32>) -> tensor<1x1xf32>
    %983 = "tosa.mul"(%981, %982) {shift = 0 : i32} : (tensor<100x2048xf32>, tensor<1x1xf32>) -> tensor<100x2048xf32>
    %984 = "tosa.matmul"(%983, %3) : (tensor<100x2048xf32>, tensor<2048x1000xf32>) -> tensor<100x1000xf32>
    %985 = "tosa.reshape"(%4) {new_shape = [1, 1000]} : (tensor<1000xf32>) -> tensor<1x1000xf32>
    %986 = "tosa.add"(%984, %985) : (tensor<100x1000xf32>, tensor<1x1000xf32>) -> tensor<100x1000xf32>
    %987 = "tosa.exp"(%986) : (tensor<100x1000xf32>) -> tensor<100x1000xf32>
    %988 = "tosa.reduce_sum"(%987) {axis = 1 : i64} : (tensor<100x1000xf32>) -> tensor<100x1xf32>
    %989 = "tosa.reciprocal"(%988) : (tensor<100x1xf32>) -> tensor<100x1xf32>
    %990 = "tosa.mul"(%987, %989) {shift = 0 : i32} : (tensor<100x1000xf32>, tensor<100x1xf32>) -> tensor<100x1000xf32>
    return %990 : tensor<100x1000xf32>
  }
}

