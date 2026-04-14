module {
  func.func @main_graph(%arg0: !migraphx.shaped<1x10000xf16, 10000x1>) -> !migraphx.shaped<1x6xf16, 6x1> {
    %0 = migraphx.literal(dense<0.000000e+00> : tensor<512x1x10xf16>) : <512x1x10xf16, 10x10x1>
    %1 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %2 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %3 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %4 = migraphx.literal(dense<0.000000e+00> : tensor<512x512x3xf16>) : <512x512x3xf16, 1536x3x1>
    %5 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %6 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %7 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %8 = migraphx.literal(dense<0.000000e+00> : tensor<512x512x3xf16>) : <512x512x3xf16, 1536x3x1>
    %9 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %10 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %11 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %12 = migraphx.literal(dense<0.000000e+00> : tensor<512x512x3xf16>) : <512x512x3xf16, 1536x3x1>
    %13 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %14 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %15 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %16 = migraphx.literal(dense<0.000000e+00> : tensor<512x512x3xf16>) : <512x512x3xf16, 1536x3x1>
    %17 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %18 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %19 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %20 = migraphx.literal(dense<0.000000e+00> : tensor<512x512x2xf16>) : <512x512x2xf16, 1024x2x1>
    %21 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %22 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %23 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %24 = migraphx.literal(dense<0.000000e+00> : tensor<512x512x2xf16>) : <512x512x2xf16, 1024x2x1>
    %25 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %26 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %27 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %28 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %29 = migraphx.literal(dense<0.000000e+00> : tensor<512xf16>) : <512xf16, 1>
    %30 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %31 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %32 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %33 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %34 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %35 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %36 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %37 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %38 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %39 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %40 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %41 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %42 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %43 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %44 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %45 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %46 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %47 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %48 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %49 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %50 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %51 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %52 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %53 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %54 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %55 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %56 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %57 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %58 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %59 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %60 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %61 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %62 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %63 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %64 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %65 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %66 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %67 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %68 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %69 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %70 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %71 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %72 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %73 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %74 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %75 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %76 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %77 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %78 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %79 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %80 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %81 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %82 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %83 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %84 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %85 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %86 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %87 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %88 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %89 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %90 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %91 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %92 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %93 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %94 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %95 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %96 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %97 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %98 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %99 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %100 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %101 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %102 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %103 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %104 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %105 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %106 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %107 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %108 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %109 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %110 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %111 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %112 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %113 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %114 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %115 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %116 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %117 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %118 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %119 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %120 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %121 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %122 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %123 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %124 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %125 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %126 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %127 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %128 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %129 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %130 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %131 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %132 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %133 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %134 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %135 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %136 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %137 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %138 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %139 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %140 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %141 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %142 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %143 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %144 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %145 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %146 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %147 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %148 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %149 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %150 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %151 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %152 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %153 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %154 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %155 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %156 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %157 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %158 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %159 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %160 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %161 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %162 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %163 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %164 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %165 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %166 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %167 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %168 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %169 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %170 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %171 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %172 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %173 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %174 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %175 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %176 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %177 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %178 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %179 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %180 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %181 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %182 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %183 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %184 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %185 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %186 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %187 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %188 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %189 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %190 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %191 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %192 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %193 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %194 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %195 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %196 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %197 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %198 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %199 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %200 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %201 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %202 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %203 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %204 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %205 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %206 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %207 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %208 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %209 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %210 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %211 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %212 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %213 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %214 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %215 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %216 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %217 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %218 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %219 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %220 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %221 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %222 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %223 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %224 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %225 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %226 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %227 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %228 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %229 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %230 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %231 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %232 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %233 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %234 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %235 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %236 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %237 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %238 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %239 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %240 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %241 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %242 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %243 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %244 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %245 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %246 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %247 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %248 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %249 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %250 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %251 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %252 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %253 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %254 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %255 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %256 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %257 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %258 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %259 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %260 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %261 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %262 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %263 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %264 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %265 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %266 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %267 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %268 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %269 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %270 = migraphx.literal(dense<0.000000e+00> : tensor<4096xf16>) : <4096xf16, 1>
    %271 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %272 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %273 = migraphx.literal(dense<0.000000e+00> : tensor<1024xf16>) : <1024xf16, 1>
    %274 = migraphx.literal(dense<0.000000e+00> : tensor<512x1024xf16>) : <512x1024xf16, 1024x1>
    %275 = migraphx.literal(dense<0.000000e+00> : tensor<1024x64x128xf16>) : <1024x64x128xf16, 8192x128x1>
    %276 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %277 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %278 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %279 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %280 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %281 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %282 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %283 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %284 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %285 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %286 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %287 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %288 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %289 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %290 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %291 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %292 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %293 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %294 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %295 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %296 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %297 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %298 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %299 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %300 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %301 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %302 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %303 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %304 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %305 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %306 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %307 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %308 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %309 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %310 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %311 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %312 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %313 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %314 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %315 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %316 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %317 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %318 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %319 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %320 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %321 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %322 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %323 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %324 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %325 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %326 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %327 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %328 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %329 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %330 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %331 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %332 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %333 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %334 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %335 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %336 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %337 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %338 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %339 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %340 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %341 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %342 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %343 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %344 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %345 = migraphx.literal(dense<0.000000e+00> : tensor<1024x1024xf16>) : <1024x1024xf16, 1024x1>
    %346 = migraphx.literal(dense<0.000000e+00> : tensor<1024x4096xf16>) : <1024x4096xf16, 4096x1>
    %347 = migraphx.literal(dense<0.000000e+00> : tensor<4096x1024xf16>) : <4096x1024xf16, 1024x1>
    %348 = migraphx.literal(dense<0.000000e+00> : tensor<1024x6xf16>) : <1024x6xf16, 6x1>
    %349 = migraphx.literal(dense<0.000000e+00> : tensor<1xf16>) : <1xf16, 1>
    %350 = migraphx.literal(dense<0.000000e+00> : tensor<1xf16>) : <1xf16, 1>
    %351 = migraphx.literal(dense<0.000000e+00> : tensor<1xf16>) : <1xf16, 1>
    %352 = migraphx.literal(dense<0.000000e+00> : tensor<1xf16>) : <1xf16, 1>
    %353 = migraphx.literal(dense<0.000000e+00> : tensor<1xf16>) : <1xf16, 1>
    %354 = migraphx.literal(dense<0.000000e+00> : tensor<1xf16>) : <1xf16, 1>
    %355 = migraphx.literal(dense<0.000000e+00> : tensor<1xf16>) : <1xf16, 1>
    %356 = migraphx.literal(dense<0.000000e+00> : tensor<1xf16>) : <1xf16, 1>
    %357 = migraphx.literal(dense<0.000000e+00> : tensor<1xf16>) : <1xf16, 1>
    %358 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %359 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %360 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %361 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %362 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %363 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %364 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %365 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %366 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %367 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %368 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %369 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %370 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %371 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %372 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %373 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %374 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %375 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %376 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %377 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %378 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %379 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %380 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %381 = migraphx.literal(dense<0.000000e+00> : tensor<1024x3072xf16>) : <1024x3072xf16, 3072x1>
    %382 = migraphx.reduce_mean %arg0 {axes = [1]} : <1x10000xf16, 10000x1> -> <1x1xf16, 1x1>
    %383 = migraphx.sub %arg0, %382 : <1x10000xf16, 10000x1>, <1x1xf16, 1x1> -> <1x10000xf16, 10000x1>
    %384 = migraphx.mul %383, %383 : <1x10000xf16, 10000x1>, <1x10000xf16, 10000x1> -> <1x10000xf16, 10000x1>
    %385 = migraphx.reduce_mean %384 {axes = [1]} : <1x10000xf16, 10000x1> -> <1x1xf16, 1x1>
    %386 = migraphx.mul %385, %356 : <1x1xf16, 1x1>, <1xf16, 1> -> <1x1xf16, 1x1>
    %387 = migraphx.div %386, %357 : <1x1xf16, 1x1>, <1xf16, 1> -> <1x1xf16, 1x1>
    %388 = migraphx.sqrt %387 : <1x1xf16, 1x1> -> <1x1xf16, 1x1>
    %389 = migraphx.add %388, %350 : <1x1xf16, 1x1>, <1xf16, 1> -> <1x1xf16, 1x1>
    %390 = migraphx.div %383, %389 : <1x10000xf16, 10000x1>, <1x1xf16, 1x1> -> <1x10000xf16, 10000x1>
    %391 = migraphx.reshape %390 {dims = [1, 2, 5000]} : <1x10000xf16, 10000x1> -> <1x2x5000xf16, 10000x5000x1>
    %392 = migraphx.slice %391 {axes = [1], ends = [-1], starts = [0]} : <1x2x5000xf16, 10000x5000x1> -> <1x1x5000xf16, 5000x5000x1>
    %393 = migraphx.slice %391 {axes = [1], ends = [9223372036854775807], starts = [1]} : <1x2x5000xf16, 10000x5000x1> -> <1x1x5000xf16, 5000x5000x1>
    %394 = migraphx.pad %392 {mode = 0 : i32, pads = [0, 0, 0, 0, 0, 5000], value = 0.000000e+00 : f32} : <1x1x5000xf16, 5000x5000x1> -> <1x1x10000xf16, 10000x10000x1>
    %395 = migraphx.pad %393 {mode = 0 : i32, pads = [0, 0, 5000, 0, 0, 0], value = 0.000000e+00 : f32} : <1x1x5000xf16, 5000x5000x1> -> <1x1x10000xf16, 10000x10000x1>
    %396 = migraphx.add %394, %395 : <1x1x10000xf16, 10000x10000x1>, <1x1x10000xf16, 10000x10000x1> -> <1x1x10000xf16, 10000x10000x1>
    %397 = migraphx.reshape %396 {dims = [1, 10000]} : <1x1x10000xf16, 10000x10000x1> -> <1x10000xf16, 10000x1>
    %398 = migraphx.reshape %397 {dims = [1, 1, 10000]} : <1x10000xf16, 10000x1> -> <1x1x10000xf16, 10000x10000x1>
    %399 = migraphx.convolution %398, %0 {dilation = [1], group = 1 : i64, padding = [0, 0], stride = [5]} : <1x1x10000xf16, 10000x10000x1>, <512x1x10xf16, 10x10x1> -> <1x512x1999xf16, 1023488x1999x1>
    %400 = migraphx.multibroadcast %1 {out_lens = [1, 512, 1999]} : <512xf16, 1> -> <1x512x1999xf16, 0x1x0>
    %401 = migraphx.add %399, %400 : <1x512x1999xf16, 1023488x1999x1>, <1x512x1999xf16, 0x1x0> -> <1x512x1999xf16, 1023488x1999x1>
    %402 = migraphx.transpose %401 {permutation = [0, 2, 1]} : <1x512x1999xf16, 1023488x1999x1> -> <1x1999x512xf16, 1023488x512x1>
    %403 = migraphx.reduce_mean %402 {axes = [-1]} : <1x1999x512xf16, 1023488x512x1> -> <1x1999x1xf16, 1999x1x1>
    %404 = migraphx.sub %402, %403 : <1x1999x512xf16, 1023488x512x1>, <1x1999x1xf16, 1999x1x1> -> <1x1999x512xf16, 1023488x512x1>
    %405 = migraphx.pow %404, %351 : <1x1999x512xf16, 1023488x512x1>, <1xf16, 1> -> <1x1999x512xf16, 1023488x512x1>
    %406 = migraphx.reduce_mean %405 {axes = [-1]} : <1x1999x512xf16, 1023488x512x1> -> <1x1999x1xf16, 1999x1x1>
    %407 = migraphx.add %406, %352 : <1x1999x1xf16, 1999x1x1>, <1xf16, 1> -> <1x1999x1xf16, 1999x1x1>
    %408 = migraphx.sqrt %407 : <1x1999x1xf16, 1999x1x1> -> <1x1999x1xf16, 1999x1x1>
    %409 = migraphx.div %404, %408 : <1x1999x512xf16, 1023488x512x1>, <1x1999x1xf16, 1999x1x1> -> <1x1999x512xf16, 1023488x512x1>
    %410 = migraphx.mul %409, %2 : <1x1999x512xf16, 1023488x512x1>, <512xf16, 1> -> <1x1999x512xf16, 1023488x512x1>
    %411 = migraphx.add %410, %3 : <1x1999x512xf16, 1023488x512x1>, <512xf16, 1> -> <1x1999x512xf16, 1023488x512x1>
    %412 = migraphx.transpose %411 {permutation = [0, 2, 1]} : <1x1999x512xf16, 1023488x512x1> -> <1x512x1999xf16, 1023488x1999x1>
    %413 = migraphx.div %412, %353 : <1x512x1999xf16, 1023488x1999x1>, <1xf16, 1> -> <1x512x1999xf16, 1023488x1999x1>
    %414 = migraphx.erf %413 : <1x512x1999xf16, 1023488x1999x1> -> <1x512x1999xf16, 1023488x1999x1>
    %415 = migraphx.add %414, %349 : <1x512x1999xf16, 1023488x1999x1>, <1xf16, 1> -> <1x512x1999xf16, 1023488x1999x1>
    %416 = migraphx.mul %412, %415 : <1x512x1999xf16, 1023488x1999x1>, <1x512x1999xf16, 1023488x1999x1> -> <1x512x1999xf16, 1023488x1999x1>
    %417 = migraphx.mul %416, %354 : <1x512x1999xf16, 1023488x1999x1>, <1xf16, 1> -> <1x512x1999xf16, 1023488x1999x1>
    %418 = migraphx.convolution %417, %4 {dilation = [1], group = 1 : i64, padding = [0, 0], stride = [2]} : <1x512x1999xf16, 1023488x1999x1>, <512x512x3xf16, 1536x3x1> -> <1x512x999xf16, 511488x999x1>
    %419 = migraphx.multibroadcast %5 {out_lens = [1, 512, 999]} : <512xf16, 1> -> <1x512x999xf16, 0x1x0>
    %420 = migraphx.add %418, %419 : <1x512x999xf16, 511488x999x1>, <1x512x999xf16, 0x1x0> -> <1x512x999xf16, 511488x999x1>
    %421 = migraphx.transpose %420 {permutation = [0, 2, 1]} : <1x512x999xf16, 511488x999x1> -> <1x999x512xf16, 511488x512x1>
    %422 = migraphx.reduce_mean %421 {axes = [-1]} : <1x999x512xf16, 511488x512x1> -> <1x999x1xf16, 999x1x1>
    %423 = migraphx.sub %421, %422 : <1x999x512xf16, 511488x512x1>, <1x999x1xf16, 999x1x1> -> <1x999x512xf16, 511488x512x1>
    %424 = migraphx.pow %423, %351 : <1x999x512xf16, 511488x512x1>, <1xf16, 1> -> <1x999x512xf16, 511488x512x1>
    %425 = migraphx.reduce_mean %424 {axes = [-1]} : <1x999x512xf16, 511488x512x1> -> <1x999x1xf16, 999x1x1>
    %426 = migraphx.add %425, %352 : <1x999x1xf16, 999x1x1>, <1xf16, 1> -> <1x999x1xf16, 999x1x1>
    %427 = migraphx.sqrt %426 : <1x999x1xf16, 999x1x1> -> <1x999x1xf16, 999x1x1>
    %428 = migraphx.div %423, %427 : <1x999x512xf16, 511488x512x1>, <1x999x1xf16, 999x1x1> -> <1x999x512xf16, 511488x512x1>
    %429 = migraphx.mul %428, %6 : <1x999x512xf16, 511488x512x1>, <512xf16, 1> -> <1x999x512xf16, 511488x512x1>
    %430 = migraphx.add %429, %7 : <1x999x512xf16, 511488x512x1>, <512xf16, 1> -> <1x999x512xf16, 511488x512x1>
    %431 = migraphx.transpose %430 {permutation = [0, 2, 1]} : <1x999x512xf16, 511488x512x1> -> <1x512x999xf16, 511488x999x1>
    %432 = migraphx.div %431, %353 : <1x512x999xf16, 511488x999x1>, <1xf16, 1> -> <1x512x999xf16, 511488x999x1>
    %433 = migraphx.erf %432 : <1x512x999xf16, 511488x999x1> -> <1x512x999xf16, 511488x999x1>
    %434 = migraphx.add %433, %349 : <1x512x999xf16, 511488x999x1>, <1xf16, 1> -> <1x512x999xf16, 511488x999x1>
    %435 = migraphx.mul %431, %434 : <1x512x999xf16, 511488x999x1>, <1x512x999xf16, 511488x999x1> -> <1x512x999xf16, 511488x999x1>
    %436 = migraphx.mul %435, %354 : <1x512x999xf16, 511488x999x1>, <1xf16, 1> -> <1x512x999xf16, 511488x999x1>
    %437 = migraphx.convolution %436, %8 {dilation = [1], group = 1 : i64, padding = [0, 0], stride = [2]} : <1x512x999xf16, 511488x999x1>, <512x512x3xf16, 1536x3x1> -> <1x512x499xf16, 255488x499x1>
    %438 = migraphx.multibroadcast %9 {out_lens = [1, 512, 499]} : <512xf16, 1> -> <1x512x499xf16, 0x1x0>
    %439 = migraphx.add %437, %438 : <1x512x499xf16, 255488x499x1>, <1x512x499xf16, 0x1x0> -> <1x512x499xf16, 255488x499x1>
    %440 = migraphx.transpose %439 {permutation = [0, 2, 1]} : <1x512x499xf16, 255488x499x1> -> <1x499x512xf16, 255488x512x1>
    %441 = migraphx.reduce_mean %440 {axes = [-1]} : <1x499x512xf16, 255488x512x1> -> <1x499x1xf16, 499x1x1>
    %442 = migraphx.sub %440, %441 : <1x499x512xf16, 255488x512x1>, <1x499x1xf16, 499x1x1> -> <1x499x512xf16, 255488x512x1>
    %443 = migraphx.pow %442, %351 : <1x499x512xf16, 255488x512x1>, <1xf16, 1> -> <1x499x512xf16, 255488x512x1>
    %444 = migraphx.reduce_mean %443 {axes = [-1]} : <1x499x512xf16, 255488x512x1> -> <1x499x1xf16, 499x1x1>
    %445 = migraphx.add %444, %352 : <1x499x1xf16, 499x1x1>, <1xf16, 1> -> <1x499x1xf16, 499x1x1>
    %446 = migraphx.sqrt %445 : <1x499x1xf16, 499x1x1> -> <1x499x1xf16, 499x1x1>
    %447 = migraphx.div %442, %446 : <1x499x512xf16, 255488x512x1>, <1x499x1xf16, 499x1x1> -> <1x499x512xf16, 255488x512x1>
    %448 = migraphx.mul %447, %10 : <1x499x512xf16, 255488x512x1>, <512xf16, 1> -> <1x499x512xf16, 255488x512x1>
    %449 = migraphx.add %448, %11 : <1x499x512xf16, 255488x512x1>, <512xf16, 1> -> <1x499x512xf16, 255488x512x1>
    %450 = migraphx.transpose %449 {permutation = [0, 2, 1]} : <1x499x512xf16, 255488x512x1> -> <1x512x499xf16, 255488x499x1>
    %451 = migraphx.div %450, %353 : <1x512x499xf16, 255488x499x1>, <1xf16, 1> -> <1x512x499xf16, 255488x499x1>
    %452 = migraphx.erf %451 : <1x512x499xf16, 255488x499x1> -> <1x512x499xf16, 255488x499x1>
    %453 = migraphx.add %452, %349 : <1x512x499xf16, 255488x499x1>, <1xf16, 1> -> <1x512x499xf16, 255488x499x1>
    %454 = migraphx.mul %450, %453 : <1x512x499xf16, 255488x499x1>, <1x512x499xf16, 255488x499x1> -> <1x512x499xf16, 255488x499x1>
    %455 = migraphx.mul %454, %354 : <1x512x499xf16, 255488x499x1>, <1xf16, 1> -> <1x512x499xf16, 255488x499x1>
    %456 = migraphx.convolution %455, %12 {dilation = [1], group = 1 : i64, padding = [0, 0], stride = [2]} : <1x512x499xf16, 255488x499x1>, <512x512x3xf16, 1536x3x1> -> <1x512x249xf16, 127488x249x1>
    %457 = migraphx.multibroadcast %13 {out_lens = [1, 512, 249]} : <512xf16, 1> -> <1x512x249xf16, 0x1x0>
    %458 = migraphx.add %456, %457 : <1x512x249xf16, 127488x249x1>, <1x512x249xf16, 0x1x0> -> <1x512x249xf16, 127488x249x1>
    %459 = migraphx.transpose %458 {permutation = [0, 2, 1]} : <1x512x249xf16, 127488x249x1> -> <1x249x512xf16, 127488x512x1>
    %460 = migraphx.reduce_mean %459 {axes = [-1]} : <1x249x512xf16, 127488x512x1> -> <1x249x1xf16, 249x1x1>
    %461 = migraphx.sub %459, %460 : <1x249x512xf16, 127488x512x1>, <1x249x1xf16, 249x1x1> -> <1x249x512xf16, 127488x512x1>
    %462 = migraphx.pow %461, %351 : <1x249x512xf16, 127488x512x1>, <1xf16, 1> -> <1x249x512xf16, 127488x512x1>
    %463 = migraphx.reduce_mean %462 {axes = [-1]} : <1x249x512xf16, 127488x512x1> -> <1x249x1xf16, 249x1x1>
    %464 = migraphx.add %463, %352 : <1x249x1xf16, 249x1x1>, <1xf16, 1> -> <1x249x1xf16, 249x1x1>
    %465 = migraphx.sqrt %464 : <1x249x1xf16, 249x1x1> -> <1x249x1xf16, 249x1x1>
    %466 = migraphx.div %461, %465 : <1x249x512xf16, 127488x512x1>, <1x249x1xf16, 249x1x1> -> <1x249x512xf16, 127488x512x1>
    %467 = migraphx.mul %466, %14 : <1x249x512xf16, 127488x512x1>, <512xf16, 1> -> <1x249x512xf16, 127488x512x1>
    %468 = migraphx.add %467, %15 : <1x249x512xf16, 127488x512x1>, <512xf16, 1> -> <1x249x512xf16, 127488x512x1>
    %469 = migraphx.transpose %468 {permutation = [0, 2, 1]} : <1x249x512xf16, 127488x512x1> -> <1x512x249xf16, 127488x249x1>
    %470 = migraphx.div %469, %353 : <1x512x249xf16, 127488x249x1>, <1xf16, 1> -> <1x512x249xf16, 127488x249x1>
    %471 = migraphx.erf %470 : <1x512x249xf16, 127488x249x1> -> <1x512x249xf16, 127488x249x1>
    %472 = migraphx.add %471, %349 : <1x512x249xf16, 127488x249x1>, <1xf16, 1> -> <1x512x249xf16, 127488x249x1>
    %473 = migraphx.mul %469, %472 : <1x512x249xf16, 127488x249x1>, <1x512x249xf16, 127488x249x1> -> <1x512x249xf16, 127488x249x1>
    %474 = migraphx.mul %473, %354 : <1x512x249xf16, 127488x249x1>, <1xf16, 1> -> <1x512x249xf16, 127488x249x1>
    %475 = migraphx.convolution %474, %16 {dilation = [1], group = 1 : i64, padding = [0, 0], stride = [2]} : <1x512x249xf16, 127488x249x1>, <512x512x3xf16, 1536x3x1> -> <1x512x124xf16, 63488x124x1>
    %476 = migraphx.multibroadcast %17 {out_lens = [1, 512, 124]} : <512xf16, 1> -> <1x512x124xf16, 0x1x0>
    %477 = migraphx.add %475, %476 : <1x512x124xf16, 63488x124x1>, <1x512x124xf16, 0x1x0> -> <1x512x124xf16, 63488x124x1>
    %478 = migraphx.transpose %477 {permutation = [0, 2, 1]} : <1x512x124xf16, 63488x124x1> -> <1x124x512xf16, 63488x512x1>
    %479 = migraphx.reduce_mean %478 {axes = [-1]} : <1x124x512xf16, 63488x512x1> -> <1x124x1xf16, 124x1x1>
    %480 = migraphx.sub %478, %479 : <1x124x512xf16, 63488x512x1>, <1x124x1xf16, 124x1x1> -> <1x124x512xf16, 63488x512x1>
    %481 = migraphx.pow %480, %351 : <1x124x512xf16, 63488x512x1>, <1xf16, 1> -> <1x124x512xf16, 63488x512x1>
    %482 = migraphx.reduce_mean %481 {axes = [-1]} : <1x124x512xf16, 63488x512x1> -> <1x124x1xf16, 124x1x1>
    %483 = migraphx.add %482, %352 : <1x124x1xf16, 124x1x1>, <1xf16, 1> -> <1x124x1xf16, 124x1x1>
    %484 = migraphx.sqrt %483 : <1x124x1xf16, 124x1x1> -> <1x124x1xf16, 124x1x1>
    %485 = migraphx.div %480, %484 : <1x124x512xf16, 63488x512x1>, <1x124x1xf16, 124x1x1> -> <1x124x512xf16, 63488x512x1>
    %486 = migraphx.mul %485, %18 : <1x124x512xf16, 63488x512x1>, <512xf16, 1> -> <1x124x512xf16, 63488x512x1>
    %487 = migraphx.add %486, %19 : <1x124x512xf16, 63488x512x1>, <512xf16, 1> -> <1x124x512xf16, 63488x512x1>
    %488 = migraphx.transpose %487 {permutation = [0, 2, 1]} : <1x124x512xf16, 63488x512x1> -> <1x512x124xf16, 63488x124x1>
    %489 = migraphx.div %488, %353 : <1x512x124xf16, 63488x124x1>, <1xf16, 1> -> <1x512x124xf16, 63488x124x1>
    %490 = migraphx.erf %489 : <1x512x124xf16, 63488x124x1> -> <1x512x124xf16, 63488x124x1>
    %491 = migraphx.add %490, %349 : <1x512x124xf16, 63488x124x1>, <1xf16, 1> -> <1x512x124xf16, 63488x124x1>
    %492 = migraphx.mul %488, %491 : <1x512x124xf16, 63488x124x1>, <1x512x124xf16, 63488x124x1> -> <1x512x124xf16, 63488x124x1>
    %493 = migraphx.mul %492, %354 : <1x512x124xf16, 63488x124x1>, <1xf16, 1> -> <1x512x124xf16, 63488x124x1>
    %494 = migraphx.convolution %493, %20 {dilation = [1], group = 1 : i64, padding = [0, 0], stride = [2]} : <1x512x124xf16, 63488x124x1>, <512x512x2xf16, 1024x2x1> -> <1x512x62xf16, 31744x62x1>
    %495 = migraphx.multibroadcast %21 {out_lens = [1, 512, 62]} : <512xf16, 1> -> <1x512x62xf16, 0x1x0>
    %496 = migraphx.add %494, %495 : <1x512x62xf16, 31744x62x1>, <1x512x62xf16, 0x1x0> -> <1x512x62xf16, 31744x62x1>
    %497 = migraphx.transpose %496 {permutation = [0, 2, 1]} : <1x512x62xf16, 31744x62x1> -> <1x62x512xf16, 31744x512x1>
    %498 = migraphx.reduce_mean %497 {axes = [-1]} : <1x62x512xf16, 31744x512x1> -> <1x62x1xf16, 62x1x1>
    %499 = migraphx.sub %497, %498 : <1x62x512xf16, 31744x512x1>, <1x62x1xf16, 62x1x1> -> <1x62x512xf16, 31744x512x1>
    %500 = migraphx.pow %499, %351 : <1x62x512xf16, 31744x512x1>, <1xf16, 1> -> <1x62x512xf16, 31744x512x1>
    %501 = migraphx.reduce_mean %500 {axes = [-1]} : <1x62x512xf16, 31744x512x1> -> <1x62x1xf16, 62x1x1>
    %502 = migraphx.add %501, %352 : <1x62x1xf16, 62x1x1>, <1xf16, 1> -> <1x62x1xf16, 62x1x1>
    %503 = migraphx.sqrt %502 : <1x62x1xf16, 62x1x1> -> <1x62x1xf16, 62x1x1>
    %504 = migraphx.div %499, %503 : <1x62x512xf16, 31744x512x1>, <1x62x1xf16, 62x1x1> -> <1x62x512xf16, 31744x512x1>
    %505 = migraphx.mul %504, %22 : <1x62x512xf16, 31744x512x1>, <512xf16, 1> -> <1x62x512xf16, 31744x512x1>
    %506 = migraphx.add %505, %23 : <1x62x512xf16, 31744x512x1>, <512xf16, 1> -> <1x62x512xf16, 31744x512x1>
    %507 = migraphx.transpose %506 {permutation = [0, 2, 1]} : <1x62x512xf16, 31744x512x1> -> <1x512x62xf16, 31744x62x1>
    %508 = migraphx.div %507, %353 : <1x512x62xf16, 31744x62x1>, <1xf16, 1> -> <1x512x62xf16, 31744x62x1>
    %509 = migraphx.erf %508 : <1x512x62xf16, 31744x62x1> -> <1x512x62xf16, 31744x62x1>
    %510 = migraphx.add %509, %349 : <1x512x62xf16, 31744x62x1>, <1xf16, 1> -> <1x512x62xf16, 31744x62x1>
    %511 = migraphx.mul %507, %510 : <1x512x62xf16, 31744x62x1>, <1x512x62xf16, 31744x62x1> -> <1x512x62xf16, 31744x62x1>
    %512 = migraphx.mul %511, %354 : <1x512x62xf16, 31744x62x1>, <1xf16, 1> -> <1x512x62xf16, 31744x62x1>
    %513 = migraphx.convolution %512, %24 {dilation = [1], group = 1 : i64, padding = [0, 0], stride = [2]} : <1x512x62xf16, 31744x62x1>, <512x512x2xf16, 1024x2x1> -> <1x512x31xf16, 15872x31x1>
    %514 = migraphx.multibroadcast %25 {out_lens = [1, 512, 31]} : <512xf16, 1> -> <1x512x31xf16, 0x1x0>
    %515 = migraphx.add %513, %514 : <1x512x31xf16, 15872x31x1>, <1x512x31xf16, 0x1x0> -> <1x512x31xf16, 15872x31x1>
    %516 = migraphx.transpose %515 {permutation = [0, 2, 1]} : <1x512x31xf16, 15872x31x1> -> <1x31x512xf16, 15872x512x1>
    %517 = migraphx.reduce_mean %516 {axes = [-1]} : <1x31x512xf16, 15872x512x1> -> <1x31x1xf16, 31x1x1>
    %518 = migraphx.sub %516, %517 : <1x31x512xf16, 15872x512x1>, <1x31x1xf16, 31x1x1> -> <1x31x512xf16, 15872x512x1>
    %519 = migraphx.pow %518, %351 : <1x31x512xf16, 15872x512x1>, <1xf16, 1> -> <1x31x512xf16, 15872x512x1>
    %520 = migraphx.reduce_mean %519 {axes = [-1]} : <1x31x512xf16, 15872x512x1> -> <1x31x1xf16, 31x1x1>
    %521 = migraphx.add %520, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %522 = migraphx.sqrt %521 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %523 = migraphx.div %518, %522 : <1x31x512xf16, 15872x512x1>, <1x31x1xf16, 31x1x1> -> <1x31x512xf16, 15872x512x1>
    %524 = migraphx.mul %523, %26 : <1x31x512xf16, 15872x512x1>, <512xf16, 1> -> <1x31x512xf16, 15872x512x1>
    %525 = migraphx.add %524, %27 : <1x31x512xf16, 15872x512x1>, <512xf16, 1> -> <1x31x512xf16, 15872x512x1>
    %526 = migraphx.transpose %525 {permutation = [0, 2, 1]} : <1x31x512xf16, 15872x512x1> -> <1x512x31xf16, 15872x31x1>
    %527 = migraphx.div %526, %353 : <1x512x31xf16, 15872x31x1>, <1xf16, 1> -> <1x512x31xf16, 15872x31x1>
    %528 = migraphx.erf %527 : <1x512x31xf16, 15872x31x1> -> <1x512x31xf16, 15872x31x1>
    %529 = migraphx.add %528, %349 : <1x512x31xf16, 15872x31x1>, <1xf16, 1> -> <1x512x31xf16, 15872x31x1>
    %530 = migraphx.mul %526, %529 : <1x512x31xf16, 15872x31x1>, <1x512x31xf16, 15872x31x1> -> <1x512x31xf16, 15872x31x1>
    %531 = migraphx.mul %530, %354 : <1x512x31xf16, 15872x31x1>, <1xf16, 1> -> <1x512x31xf16, 15872x31x1>
    %532 = migraphx.transpose %531 {permutation = [0, 2, 1]} : <1x512x31xf16, 15872x31x1> -> <1x31x512xf16, 15872x512x1>
    %533 = migraphx.reduce_mean %532 {axes = [-1]} : <1x31x512xf16, 15872x512x1> -> <1x31x1xf16, 31x1x1>
    %534 = migraphx.sub %532, %533 : <1x31x512xf16, 15872x512x1>, <1x31x1xf16, 31x1x1> -> <1x31x512xf16, 15872x512x1>
    %535 = migraphx.pow %534, %351 : <1x31x512xf16, 15872x512x1>, <1xf16, 1> -> <1x31x512xf16, 15872x512x1>
    %536 = migraphx.reduce_mean %535 {axes = [-1]} : <1x31x512xf16, 15872x512x1> -> <1x31x1xf16, 31x1x1>
    %537 = migraphx.add %536, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %538 = migraphx.sqrt %537 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %539 = migraphx.div %534, %538 : <1x31x512xf16, 15872x512x1>, <1x31x1xf16, 31x1x1> -> <1x31x512xf16, 15872x512x1>
    %540 = migraphx.mul %539, %28 : <1x31x512xf16, 15872x512x1>, <512xf16, 1> -> <1x31x512xf16, 15872x512x1>
    %541 = migraphx.add %540, %29 : <1x31x512xf16, 15872x512x1>, <512xf16, 1> -> <1x31x512xf16, 15872x512x1>
    %542 = migraphx.dot %541, %274 : <1x31x512xf16, 15872x512x1>, <512x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %543 = migraphx.add %30, %542 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %544 = migraphx.transpose %543 {permutation = [0, 2, 1]} : <1x31x1024xf16, 31744x1024x1> -> <1x1024x31xf16, 31744x31x1>
    %545 = migraphx.convolution %544, %275 {dilation = [1], group = 16 : i64, padding = [64, 64], stride = [1]} : <1x1024x31xf16, 31744x31x1>, <1024x64x128xf16, 8192x128x1> -> <1x1024x32xf16, 32768x32x1>
    %546 = migraphx.multibroadcast %31 {out_lens = [1, 1024, 32]} : <1024xf16, 1> -> <1x1024x32xf16, 0x1x0>
    %547 = migraphx.add %545, %546 : <1x1024x32xf16, 32768x32x1>, <1x1024x32xf16, 0x1x0> -> <1x1024x32xf16, 32768x32x1>
    %548 = migraphx.slice %547 {axes = [2], ends = [-1], starts = [0]} : <1x1024x32xf16, 32768x32x1> -> <1x1024x31xf16, 31744x31x1>
    %549 = migraphx.div %548, %353 : <1x1024x31xf16, 31744x31x1>, <1xf16, 1> -> <1x1024x31xf16, 31744x31x1>
    %550 = migraphx.erf %549 : <1x1024x31xf16, 31744x31x1> -> <1x1024x31xf16, 31744x31x1>
    %551 = migraphx.add %550, %349 : <1x1024x31xf16, 31744x31x1>, <1xf16, 1> -> <1x1024x31xf16, 31744x31x1>
    %552 = migraphx.mul %548, %551 : <1x1024x31xf16, 31744x31x1>, <1x1024x31xf16, 31744x31x1> -> <1x1024x31xf16, 31744x31x1>
    %553 = migraphx.mul %552, %354 : <1x1024x31xf16, 31744x31x1>, <1xf16, 1> -> <1x1024x31xf16, 31744x31x1>
    %554 = migraphx.transpose %553 {permutation = [0, 2, 1]} : <1x1024x31xf16, 31744x31x1> -> <1x31x1024xf16, 31744x1024x1>
    %555 = migraphx.add %543, %554 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %556 = migraphx.reduce_mean %555 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %557 = migraphx.sub %555, %556 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %558 = migraphx.pow %557, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %559 = migraphx.reduce_mean %558 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %560 = migraphx.add %559, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %561 = migraphx.sqrt %560 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %562 = migraphx.div %557, %561 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %563 = migraphx.mul %562, %38 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %564 = migraphx.add %563, %39 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %565 = migraphx.dot %564, %358 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %566 = migraphx.slice %565 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %567 = migraphx.slice %565 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %568 = migraphx.slice %565 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %569 = migraphx.add %36, %566 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %570 = migraphx.mul %569, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %571 = migraphx.add %34, %567 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %572 = migraphx.reshape %571 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %573 = migraphx.transpose %572 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %574 = migraphx.add %35, %568 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %575 = migraphx.reshape %574 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %576 = migraphx.transpose %575 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %577 = migraphx.reshape %570 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %578 = migraphx.transpose %577 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %579 = migraphx.reshape %578 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %580 = migraphx.reshape %573 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %581 = migraphx.reshape %576 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %582 = migraphx.transpose %580 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %583 = migraphx.dot %579, %582 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %584 = migraphx.softmax %583 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %585 = migraphx.dot %584, %581 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %586 = migraphx.reshape %585 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %587 = migraphx.transpose %586 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %588 = migraphx.reshape %587 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %589 = migraphx.dot %588, %276 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %590 = migraphx.add %37, %589 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %591 = migraphx.add %555, %590 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %592 = migraphx.reduce_mean %591 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %593 = migraphx.sub %591, %592 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %594 = migraphx.pow %593, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %595 = migraphx.reduce_mean %594 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %596 = migraphx.add %595, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %597 = migraphx.sqrt %596 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %598 = migraphx.div %593, %597 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %599 = migraphx.mul %598, %42 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %600 = migraphx.add %599, %43 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %601 = migraphx.dot %600, %277 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %602 = migraphx.add %40, %601 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %603 = migraphx.div %602, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %604 = migraphx.erf %603 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %605 = migraphx.add %604, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %606 = migraphx.mul %602, %605 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %607 = migraphx.mul %606, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %608 = migraphx.dot %607, %278 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %609 = migraphx.add %41, %608 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %610 = migraphx.add %591, %609 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %611 = migraphx.reduce_mean %610 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %612 = migraphx.sub %610, %611 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %613 = migraphx.pow %612, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %614 = migraphx.reduce_mean %613 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %615 = migraphx.add %614, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %616 = migraphx.sqrt %615 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %617 = migraphx.div %612, %616 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %618 = migraphx.mul %617, %48 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %619 = migraphx.add %618, %49 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %620 = migraphx.dot %619, %359 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %621 = migraphx.slice %620 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %622 = migraphx.slice %620 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %623 = migraphx.slice %620 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %624 = migraphx.add %46, %621 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %625 = migraphx.mul %624, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %626 = migraphx.add %44, %622 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %627 = migraphx.reshape %626 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %628 = migraphx.transpose %627 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %629 = migraphx.add %45, %623 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %630 = migraphx.reshape %629 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %631 = migraphx.transpose %630 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %632 = migraphx.reshape %625 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %633 = migraphx.transpose %632 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %634 = migraphx.reshape %633 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %635 = migraphx.reshape %628 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %636 = migraphx.reshape %631 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %637 = migraphx.transpose %635 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %638 = migraphx.dot %634, %637 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %639 = migraphx.softmax %638 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %640 = migraphx.dot %639, %636 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %641 = migraphx.reshape %640 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %642 = migraphx.transpose %641 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %643 = migraphx.reshape %642 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %644 = migraphx.dot %643, %279 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %645 = migraphx.add %47, %644 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %646 = migraphx.add %610, %645 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %647 = migraphx.reduce_mean %646 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %648 = migraphx.sub %646, %647 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %649 = migraphx.pow %648, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %650 = migraphx.reduce_mean %649 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %651 = migraphx.add %650, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %652 = migraphx.sqrt %651 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %653 = migraphx.div %648, %652 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %654 = migraphx.mul %653, %52 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %655 = migraphx.add %654, %53 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %656 = migraphx.dot %655, %280 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %657 = migraphx.add %50, %656 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %658 = migraphx.div %657, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %659 = migraphx.erf %658 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %660 = migraphx.add %659, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %661 = migraphx.mul %657, %660 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %662 = migraphx.mul %661, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %663 = migraphx.dot %662, %281 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %664 = migraphx.add %51, %663 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %665 = migraphx.add %646, %664 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %666 = migraphx.reduce_mean %665 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %667 = migraphx.sub %665, %666 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %668 = migraphx.pow %667, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %669 = migraphx.reduce_mean %668 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %670 = migraphx.add %669, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %671 = migraphx.sqrt %670 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %672 = migraphx.div %667, %671 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %673 = migraphx.mul %672, %58 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %674 = migraphx.add %673, %59 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %675 = migraphx.dot %674, %360 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %676 = migraphx.slice %675 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %677 = migraphx.slice %675 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %678 = migraphx.slice %675 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %679 = migraphx.add %56, %676 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %680 = migraphx.mul %679, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %681 = migraphx.add %54, %677 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %682 = migraphx.reshape %681 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %683 = migraphx.transpose %682 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %684 = migraphx.add %55, %678 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %685 = migraphx.reshape %684 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %686 = migraphx.transpose %685 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %687 = migraphx.reshape %680 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %688 = migraphx.transpose %687 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %689 = migraphx.reshape %688 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %690 = migraphx.reshape %683 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %691 = migraphx.reshape %686 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %692 = migraphx.transpose %690 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %693 = migraphx.dot %689, %692 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %694 = migraphx.softmax %693 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %695 = migraphx.dot %694, %691 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %696 = migraphx.reshape %695 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %697 = migraphx.transpose %696 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %698 = migraphx.reshape %697 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %699 = migraphx.dot %698, %282 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %700 = migraphx.add %57, %699 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %701 = migraphx.add %665, %700 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %702 = migraphx.reduce_mean %701 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %703 = migraphx.sub %701, %702 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %704 = migraphx.pow %703, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %705 = migraphx.reduce_mean %704 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %706 = migraphx.add %705, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %707 = migraphx.sqrt %706 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %708 = migraphx.div %703, %707 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %709 = migraphx.mul %708, %62 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %710 = migraphx.add %709, %63 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %711 = migraphx.dot %710, %283 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %712 = migraphx.add %60, %711 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %713 = migraphx.div %712, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %714 = migraphx.erf %713 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %715 = migraphx.add %714, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %716 = migraphx.mul %712, %715 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %717 = migraphx.mul %716, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %718 = migraphx.dot %717, %284 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %719 = migraphx.add %61, %718 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %720 = migraphx.add %701, %719 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %721 = migraphx.reduce_mean %720 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %722 = migraphx.sub %720, %721 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %723 = migraphx.pow %722, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %724 = migraphx.reduce_mean %723 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %725 = migraphx.add %724, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %726 = migraphx.sqrt %725 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %727 = migraphx.div %722, %726 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %728 = migraphx.mul %727, %68 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %729 = migraphx.add %728, %69 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %730 = migraphx.dot %729, %361 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %731 = migraphx.slice %730 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %732 = migraphx.slice %730 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %733 = migraphx.slice %730 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %734 = migraphx.add %66, %731 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %735 = migraphx.mul %734, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %736 = migraphx.add %64, %732 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %737 = migraphx.reshape %736 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %738 = migraphx.transpose %737 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %739 = migraphx.add %65, %733 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %740 = migraphx.reshape %739 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %741 = migraphx.transpose %740 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %742 = migraphx.reshape %735 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %743 = migraphx.transpose %742 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %744 = migraphx.reshape %743 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %745 = migraphx.reshape %738 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %746 = migraphx.reshape %741 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %747 = migraphx.transpose %745 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %748 = migraphx.dot %744, %747 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %749 = migraphx.softmax %748 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %750 = migraphx.dot %749, %746 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %751 = migraphx.reshape %750 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %752 = migraphx.transpose %751 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %753 = migraphx.reshape %752 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %754 = migraphx.dot %753, %285 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %755 = migraphx.add %67, %754 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %756 = migraphx.add %720, %755 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %757 = migraphx.reduce_mean %756 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %758 = migraphx.sub %756, %757 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %759 = migraphx.pow %758, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %760 = migraphx.reduce_mean %759 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %761 = migraphx.add %760, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %762 = migraphx.sqrt %761 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %763 = migraphx.div %758, %762 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %764 = migraphx.mul %763, %72 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %765 = migraphx.add %764, %73 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %766 = migraphx.dot %765, %286 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %767 = migraphx.add %70, %766 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %768 = migraphx.div %767, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %769 = migraphx.erf %768 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %770 = migraphx.add %769, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %771 = migraphx.mul %767, %770 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %772 = migraphx.mul %771, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %773 = migraphx.dot %772, %287 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %774 = migraphx.add %71, %773 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %775 = migraphx.add %756, %774 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %776 = migraphx.reduce_mean %775 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %777 = migraphx.sub %775, %776 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %778 = migraphx.pow %777, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %779 = migraphx.reduce_mean %778 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %780 = migraphx.add %779, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %781 = migraphx.sqrt %780 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %782 = migraphx.div %777, %781 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %783 = migraphx.mul %782, %78 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %784 = migraphx.add %783, %79 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %785 = migraphx.dot %784, %362 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %786 = migraphx.slice %785 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %787 = migraphx.slice %785 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %788 = migraphx.slice %785 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %789 = migraphx.add %76, %786 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %790 = migraphx.mul %789, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %791 = migraphx.add %74, %787 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %792 = migraphx.reshape %791 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %793 = migraphx.transpose %792 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %794 = migraphx.add %75, %788 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %795 = migraphx.reshape %794 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %796 = migraphx.transpose %795 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %797 = migraphx.reshape %790 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %798 = migraphx.transpose %797 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %799 = migraphx.reshape %798 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %800 = migraphx.reshape %793 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %801 = migraphx.reshape %796 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %802 = migraphx.transpose %800 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %803 = migraphx.dot %799, %802 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %804 = migraphx.softmax %803 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %805 = migraphx.dot %804, %801 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %806 = migraphx.reshape %805 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %807 = migraphx.transpose %806 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %808 = migraphx.reshape %807 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %809 = migraphx.dot %808, %288 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %810 = migraphx.add %77, %809 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %811 = migraphx.add %775, %810 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %812 = migraphx.reduce_mean %811 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %813 = migraphx.sub %811, %812 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %814 = migraphx.pow %813, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %815 = migraphx.reduce_mean %814 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %816 = migraphx.add %815, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %817 = migraphx.sqrt %816 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %818 = migraphx.div %813, %817 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %819 = migraphx.mul %818, %82 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %820 = migraphx.add %819, %83 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %821 = migraphx.dot %820, %289 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %822 = migraphx.add %80, %821 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %823 = migraphx.div %822, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %824 = migraphx.erf %823 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %825 = migraphx.add %824, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %826 = migraphx.mul %822, %825 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %827 = migraphx.mul %826, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %828 = migraphx.dot %827, %290 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %829 = migraphx.add %81, %828 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %830 = migraphx.add %811, %829 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %831 = migraphx.reduce_mean %830 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %832 = migraphx.sub %830, %831 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %833 = migraphx.pow %832, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %834 = migraphx.reduce_mean %833 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %835 = migraphx.add %834, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %836 = migraphx.sqrt %835 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %837 = migraphx.div %832, %836 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %838 = migraphx.mul %837, %88 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %839 = migraphx.add %838, %89 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %840 = migraphx.dot %839, %363 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %841 = migraphx.slice %840 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %842 = migraphx.slice %840 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %843 = migraphx.slice %840 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %844 = migraphx.add %86, %841 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %845 = migraphx.mul %844, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %846 = migraphx.add %84, %842 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %847 = migraphx.reshape %846 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %848 = migraphx.transpose %847 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %849 = migraphx.add %85, %843 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %850 = migraphx.reshape %849 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %851 = migraphx.transpose %850 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %852 = migraphx.reshape %845 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %853 = migraphx.transpose %852 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %854 = migraphx.reshape %853 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %855 = migraphx.reshape %848 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %856 = migraphx.reshape %851 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %857 = migraphx.transpose %855 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %858 = migraphx.dot %854, %857 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %859 = migraphx.softmax %858 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %860 = migraphx.dot %859, %856 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %861 = migraphx.reshape %860 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %862 = migraphx.transpose %861 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %863 = migraphx.reshape %862 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %864 = migraphx.dot %863, %291 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %865 = migraphx.add %87, %864 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %866 = migraphx.add %830, %865 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %867 = migraphx.reduce_mean %866 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %868 = migraphx.sub %866, %867 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %869 = migraphx.pow %868, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %870 = migraphx.reduce_mean %869 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %871 = migraphx.add %870, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %872 = migraphx.sqrt %871 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %873 = migraphx.div %868, %872 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %874 = migraphx.mul %873, %92 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %875 = migraphx.add %874, %93 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %876 = migraphx.dot %875, %292 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %877 = migraphx.add %90, %876 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %878 = migraphx.div %877, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %879 = migraphx.erf %878 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %880 = migraphx.add %879, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %881 = migraphx.mul %877, %880 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %882 = migraphx.mul %881, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %883 = migraphx.dot %882, %293 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %884 = migraphx.add %91, %883 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %885 = migraphx.add %866, %884 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %886 = migraphx.reduce_mean %885 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %887 = migraphx.sub %885, %886 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %888 = migraphx.pow %887, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %889 = migraphx.reduce_mean %888 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %890 = migraphx.add %889, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %891 = migraphx.sqrt %890 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %892 = migraphx.div %887, %891 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %893 = migraphx.mul %892, %98 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %894 = migraphx.add %893, %99 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %895 = migraphx.dot %894, %364 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %896 = migraphx.slice %895 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %897 = migraphx.slice %895 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %898 = migraphx.slice %895 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %899 = migraphx.add %96, %896 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %900 = migraphx.mul %899, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %901 = migraphx.add %94, %897 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %902 = migraphx.reshape %901 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %903 = migraphx.transpose %902 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %904 = migraphx.add %95, %898 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %905 = migraphx.reshape %904 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %906 = migraphx.transpose %905 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %907 = migraphx.reshape %900 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %908 = migraphx.transpose %907 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %909 = migraphx.reshape %908 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %910 = migraphx.reshape %903 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %911 = migraphx.reshape %906 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %912 = migraphx.transpose %910 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %913 = migraphx.dot %909, %912 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %914 = migraphx.softmax %913 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %915 = migraphx.dot %914, %911 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %916 = migraphx.reshape %915 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %917 = migraphx.transpose %916 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %918 = migraphx.reshape %917 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %919 = migraphx.dot %918, %294 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %920 = migraphx.add %97, %919 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %921 = migraphx.add %885, %920 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %922 = migraphx.reduce_mean %921 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %923 = migraphx.sub %921, %922 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %924 = migraphx.pow %923, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %925 = migraphx.reduce_mean %924 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %926 = migraphx.add %925, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %927 = migraphx.sqrt %926 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %928 = migraphx.div %923, %927 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %929 = migraphx.mul %928, %102 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %930 = migraphx.add %929, %103 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %931 = migraphx.dot %930, %295 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %932 = migraphx.add %100, %931 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %933 = migraphx.div %932, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %934 = migraphx.erf %933 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %935 = migraphx.add %934, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %936 = migraphx.mul %932, %935 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %937 = migraphx.mul %936, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %938 = migraphx.dot %937, %296 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %939 = migraphx.add %101, %938 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %940 = migraphx.add %921, %939 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %941 = migraphx.reduce_mean %940 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %942 = migraphx.sub %940, %941 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %943 = migraphx.pow %942, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %944 = migraphx.reduce_mean %943 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %945 = migraphx.add %944, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %946 = migraphx.sqrt %945 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %947 = migraphx.div %942, %946 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %948 = migraphx.mul %947, %108 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %949 = migraphx.add %948, %109 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %950 = migraphx.dot %949, %365 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %951 = migraphx.slice %950 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %952 = migraphx.slice %950 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %953 = migraphx.slice %950 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %954 = migraphx.add %106, %951 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %955 = migraphx.mul %954, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %956 = migraphx.add %104, %952 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %957 = migraphx.reshape %956 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %958 = migraphx.transpose %957 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %959 = migraphx.add %105, %953 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %960 = migraphx.reshape %959 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %961 = migraphx.transpose %960 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %962 = migraphx.reshape %955 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %963 = migraphx.transpose %962 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %964 = migraphx.reshape %963 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %965 = migraphx.reshape %958 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %966 = migraphx.reshape %961 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %967 = migraphx.transpose %965 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %968 = migraphx.dot %964, %967 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %969 = migraphx.softmax %968 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %970 = migraphx.dot %969, %966 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %971 = migraphx.reshape %970 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %972 = migraphx.transpose %971 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %973 = migraphx.reshape %972 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %974 = migraphx.dot %973, %297 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %975 = migraphx.add %107, %974 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %976 = migraphx.add %940, %975 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %977 = migraphx.reduce_mean %976 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %978 = migraphx.sub %976, %977 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %979 = migraphx.pow %978, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %980 = migraphx.reduce_mean %979 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %981 = migraphx.add %980, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %982 = migraphx.sqrt %981 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %983 = migraphx.div %978, %982 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %984 = migraphx.mul %983, %112 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %985 = migraphx.add %984, %113 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %986 = migraphx.dot %985, %298 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %987 = migraphx.add %110, %986 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %988 = migraphx.div %987, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %989 = migraphx.erf %988 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %990 = migraphx.add %989, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %991 = migraphx.mul %987, %990 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %992 = migraphx.mul %991, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %993 = migraphx.dot %992, %299 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %994 = migraphx.add %111, %993 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %995 = migraphx.add %976, %994 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %996 = migraphx.reduce_mean %995 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %997 = migraphx.sub %995, %996 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %998 = migraphx.pow %997, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %999 = migraphx.reduce_mean %998 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1000 = migraphx.add %999, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1001 = migraphx.sqrt %1000 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1002 = migraphx.div %997, %1001 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1003 = migraphx.mul %1002, %118 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1004 = migraphx.add %1003, %119 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1005 = migraphx.dot %1004, %366 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %1006 = migraphx.slice %1005 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1007 = migraphx.slice %1005 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1008 = migraphx.slice %1005 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1009 = migraphx.add %116, %1006 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1010 = migraphx.mul %1009, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1011 = migraphx.add %114, %1007 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1012 = migraphx.reshape %1011 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1013 = migraphx.transpose %1012 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1014 = migraphx.add %115, %1008 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1015 = migraphx.reshape %1014 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1016 = migraphx.transpose %1015 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1017 = migraphx.reshape %1010 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1018 = migraphx.transpose %1017 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1019 = migraphx.reshape %1018 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1020 = migraphx.reshape %1013 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1021 = migraphx.reshape %1016 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1022 = migraphx.transpose %1020 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %1023 = migraphx.dot %1019, %1022 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %1024 = migraphx.softmax %1023 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %1025 = migraphx.dot %1024, %1021 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1026 = migraphx.reshape %1025 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1027 = migraphx.transpose %1026 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1028 = migraphx.reshape %1027 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %1029 = migraphx.dot %1028, %300 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1030 = migraphx.add %117, %1029 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1031 = migraphx.add %995, %1030 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1032 = migraphx.reduce_mean %1031 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1033 = migraphx.sub %1031, %1032 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1034 = migraphx.pow %1033, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1035 = migraphx.reduce_mean %1034 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1036 = migraphx.add %1035, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1037 = migraphx.sqrt %1036 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1038 = migraphx.div %1033, %1037 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1039 = migraphx.mul %1038, %122 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1040 = migraphx.add %1039, %123 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1041 = migraphx.dot %1040, %301 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1042 = migraphx.add %120, %1041 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1043 = migraphx.div %1042, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1044 = migraphx.erf %1043 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1045 = migraphx.add %1044, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1046 = migraphx.mul %1042, %1045 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1047 = migraphx.mul %1046, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1048 = migraphx.dot %1047, %302 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1049 = migraphx.add %121, %1048 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1050 = migraphx.add %1031, %1049 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1051 = migraphx.reduce_mean %1050 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1052 = migraphx.sub %1050, %1051 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1053 = migraphx.pow %1052, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1054 = migraphx.reduce_mean %1053 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1055 = migraphx.add %1054, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1056 = migraphx.sqrt %1055 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1057 = migraphx.div %1052, %1056 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1058 = migraphx.mul %1057, %128 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1059 = migraphx.add %1058, %129 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1060 = migraphx.dot %1059, %367 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %1061 = migraphx.slice %1060 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1062 = migraphx.slice %1060 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1063 = migraphx.slice %1060 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1064 = migraphx.add %126, %1061 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1065 = migraphx.mul %1064, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1066 = migraphx.add %124, %1062 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1067 = migraphx.reshape %1066 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1068 = migraphx.transpose %1067 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1069 = migraphx.add %125, %1063 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1070 = migraphx.reshape %1069 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1071 = migraphx.transpose %1070 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1072 = migraphx.reshape %1065 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1073 = migraphx.transpose %1072 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1074 = migraphx.reshape %1073 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1075 = migraphx.reshape %1068 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1076 = migraphx.reshape %1071 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1077 = migraphx.transpose %1075 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %1078 = migraphx.dot %1074, %1077 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %1079 = migraphx.softmax %1078 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %1080 = migraphx.dot %1079, %1076 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1081 = migraphx.reshape %1080 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1082 = migraphx.transpose %1081 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1083 = migraphx.reshape %1082 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %1084 = migraphx.dot %1083, %303 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1085 = migraphx.add %127, %1084 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1086 = migraphx.add %1050, %1085 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1087 = migraphx.reduce_mean %1086 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1088 = migraphx.sub %1086, %1087 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1089 = migraphx.pow %1088, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1090 = migraphx.reduce_mean %1089 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1091 = migraphx.add %1090, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1092 = migraphx.sqrt %1091 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1093 = migraphx.div %1088, %1092 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1094 = migraphx.mul %1093, %132 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1095 = migraphx.add %1094, %133 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1096 = migraphx.dot %1095, %304 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1097 = migraphx.add %130, %1096 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1098 = migraphx.div %1097, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1099 = migraphx.erf %1098 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1100 = migraphx.add %1099, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1101 = migraphx.mul %1097, %1100 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1102 = migraphx.mul %1101, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1103 = migraphx.dot %1102, %305 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1104 = migraphx.add %131, %1103 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1105 = migraphx.add %1086, %1104 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1106 = migraphx.reduce_mean %1105 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1107 = migraphx.sub %1105, %1106 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1108 = migraphx.pow %1107, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1109 = migraphx.reduce_mean %1108 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1110 = migraphx.add %1109, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1111 = migraphx.sqrt %1110 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1112 = migraphx.div %1107, %1111 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1113 = migraphx.mul %1112, %138 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1114 = migraphx.add %1113, %139 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1115 = migraphx.dot %1114, %368 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %1116 = migraphx.slice %1115 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1117 = migraphx.slice %1115 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1118 = migraphx.slice %1115 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1119 = migraphx.add %136, %1116 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1120 = migraphx.mul %1119, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1121 = migraphx.add %134, %1117 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1122 = migraphx.reshape %1121 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1123 = migraphx.transpose %1122 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1124 = migraphx.add %135, %1118 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1125 = migraphx.reshape %1124 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1126 = migraphx.transpose %1125 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1127 = migraphx.reshape %1120 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1128 = migraphx.transpose %1127 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1129 = migraphx.reshape %1128 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1130 = migraphx.reshape %1123 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1131 = migraphx.reshape %1126 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1132 = migraphx.transpose %1130 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %1133 = migraphx.dot %1129, %1132 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %1134 = migraphx.softmax %1133 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %1135 = migraphx.dot %1134, %1131 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1136 = migraphx.reshape %1135 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1137 = migraphx.transpose %1136 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1138 = migraphx.reshape %1137 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %1139 = migraphx.dot %1138, %306 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1140 = migraphx.add %137, %1139 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1141 = migraphx.add %1105, %1140 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1142 = migraphx.reduce_mean %1141 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1143 = migraphx.sub %1141, %1142 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1144 = migraphx.pow %1143, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1145 = migraphx.reduce_mean %1144 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1146 = migraphx.add %1145, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1147 = migraphx.sqrt %1146 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1148 = migraphx.div %1143, %1147 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1149 = migraphx.mul %1148, %142 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1150 = migraphx.add %1149, %143 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1151 = migraphx.dot %1150, %307 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1152 = migraphx.add %140, %1151 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1153 = migraphx.div %1152, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1154 = migraphx.erf %1153 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1155 = migraphx.add %1154, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1156 = migraphx.mul %1152, %1155 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1157 = migraphx.mul %1156, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1158 = migraphx.dot %1157, %308 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1159 = migraphx.add %141, %1158 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1160 = migraphx.add %1141, %1159 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1161 = migraphx.reduce_mean %1160 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1162 = migraphx.sub %1160, %1161 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1163 = migraphx.pow %1162, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1164 = migraphx.reduce_mean %1163 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1165 = migraphx.add %1164, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1166 = migraphx.sqrt %1165 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1167 = migraphx.div %1162, %1166 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1168 = migraphx.mul %1167, %148 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1169 = migraphx.add %1168, %149 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1170 = migraphx.dot %1169, %369 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %1171 = migraphx.slice %1170 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1172 = migraphx.slice %1170 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1173 = migraphx.slice %1170 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1174 = migraphx.add %146, %1171 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1175 = migraphx.mul %1174, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1176 = migraphx.add %144, %1172 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1177 = migraphx.reshape %1176 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1178 = migraphx.transpose %1177 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1179 = migraphx.add %145, %1173 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1180 = migraphx.reshape %1179 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1181 = migraphx.transpose %1180 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1182 = migraphx.reshape %1175 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1183 = migraphx.transpose %1182 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1184 = migraphx.reshape %1183 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1185 = migraphx.reshape %1178 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1186 = migraphx.reshape %1181 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1187 = migraphx.transpose %1185 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %1188 = migraphx.dot %1184, %1187 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %1189 = migraphx.softmax %1188 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %1190 = migraphx.dot %1189, %1186 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1191 = migraphx.reshape %1190 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1192 = migraphx.transpose %1191 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1193 = migraphx.reshape %1192 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %1194 = migraphx.dot %1193, %309 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1195 = migraphx.add %147, %1194 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1196 = migraphx.add %1160, %1195 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1197 = migraphx.reduce_mean %1196 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1198 = migraphx.sub %1196, %1197 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1199 = migraphx.pow %1198, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1200 = migraphx.reduce_mean %1199 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1201 = migraphx.add %1200, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1202 = migraphx.sqrt %1201 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1203 = migraphx.div %1198, %1202 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1204 = migraphx.mul %1203, %152 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1205 = migraphx.add %1204, %153 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1206 = migraphx.dot %1205, %310 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1207 = migraphx.add %150, %1206 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1208 = migraphx.div %1207, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1209 = migraphx.erf %1208 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1210 = migraphx.add %1209, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1211 = migraphx.mul %1207, %1210 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1212 = migraphx.mul %1211, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1213 = migraphx.dot %1212, %311 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1214 = migraphx.add %151, %1213 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1215 = migraphx.add %1196, %1214 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1216 = migraphx.reduce_mean %1215 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1217 = migraphx.sub %1215, %1216 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1218 = migraphx.pow %1217, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1219 = migraphx.reduce_mean %1218 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1220 = migraphx.add %1219, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1221 = migraphx.sqrt %1220 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1222 = migraphx.div %1217, %1221 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1223 = migraphx.mul %1222, %158 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1224 = migraphx.add %1223, %159 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1225 = migraphx.dot %1224, %370 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %1226 = migraphx.slice %1225 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1227 = migraphx.slice %1225 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1228 = migraphx.slice %1225 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1229 = migraphx.add %156, %1226 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1230 = migraphx.mul %1229, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1231 = migraphx.add %154, %1227 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1232 = migraphx.reshape %1231 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1233 = migraphx.transpose %1232 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1234 = migraphx.add %155, %1228 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1235 = migraphx.reshape %1234 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1236 = migraphx.transpose %1235 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1237 = migraphx.reshape %1230 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1238 = migraphx.transpose %1237 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1239 = migraphx.reshape %1238 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1240 = migraphx.reshape %1233 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1241 = migraphx.reshape %1236 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1242 = migraphx.transpose %1240 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %1243 = migraphx.dot %1239, %1242 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %1244 = migraphx.softmax %1243 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %1245 = migraphx.dot %1244, %1241 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1246 = migraphx.reshape %1245 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1247 = migraphx.transpose %1246 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1248 = migraphx.reshape %1247 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %1249 = migraphx.dot %1248, %312 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1250 = migraphx.add %157, %1249 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1251 = migraphx.add %1215, %1250 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1252 = migraphx.reduce_mean %1251 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1253 = migraphx.sub %1251, %1252 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1254 = migraphx.pow %1253, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1255 = migraphx.reduce_mean %1254 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1256 = migraphx.add %1255, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1257 = migraphx.sqrt %1256 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1258 = migraphx.div %1253, %1257 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1259 = migraphx.mul %1258, %162 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1260 = migraphx.add %1259, %163 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1261 = migraphx.dot %1260, %313 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1262 = migraphx.add %160, %1261 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1263 = migraphx.div %1262, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1264 = migraphx.erf %1263 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1265 = migraphx.add %1264, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1266 = migraphx.mul %1262, %1265 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1267 = migraphx.mul %1266, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1268 = migraphx.dot %1267, %314 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1269 = migraphx.add %161, %1268 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1270 = migraphx.add %1251, %1269 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1271 = migraphx.reduce_mean %1270 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1272 = migraphx.sub %1270, %1271 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1273 = migraphx.pow %1272, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1274 = migraphx.reduce_mean %1273 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1275 = migraphx.add %1274, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1276 = migraphx.sqrt %1275 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1277 = migraphx.div %1272, %1276 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1278 = migraphx.mul %1277, %168 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1279 = migraphx.add %1278, %169 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1280 = migraphx.dot %1279, %371 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %1281 = migraphx.slice %1280 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1282 = migraphx.slice %1280 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1283 = migraphx.slice %1280 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1284 = migraphx.add %166, %1281 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1285 = migraphx.mul %1284, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1286 = migraphx.add %164, %1282 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1287 = migraphx.reshape %1286 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1288 = migraphx.transpose %1287 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1289 = migraphx.add %165, %1283 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1290 = migraphx.reshape %1289 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1291 = migraphx.transpose %1290 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1292 = migraphx.reshape %1285 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1293 = migraphx.transpose %1292 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1294 = migraphx.reshape %1293 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1295 = migraphx.reshape %1288 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1296 = migraphx.reshape %1291 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1297 = migraphx.transpose %1295 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %1298 = migraphx.dot %1294, %1297 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %1299 = migraphx.softmax %1298 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %1300 = migraphx.dot %1299, %1296 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1301 = migraphx.reshape %1300 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1302 = migraphx.transpose %1301 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1303 = migraphx.reshape %1302 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %1304 = migraphx.dot %1303, %315 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1305 = migraphx.add %167, %1304 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1306 = migraphx.add %1270, %1305 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1307 = migraphx.reduce_mean %1306 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1308 = migraphx.sub %1306, %1307 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1309 = migraphx.pow %1308, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1310 = migraphx.reduce_mean %1309 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1311 = migraphx.add %1310, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1312 = migraphx.sqrt %1311 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1313 = migraphx.div %1308, %1312 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1314 = migraphx.mul %1313, %172 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1315 = migraphx.add %1314, %173 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1316 = migraphx.dot %1315, %316 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1317 = migraphx.add %170, %1316 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1318 = migraphx.div %1317, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1319 = migraphx.erf %1318 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1320 = migraphx.add %1319, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1321 = migraphx.mul %1317, %1320 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1322 = migraphx.mul %1321, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1323 = migraphx.dot %1322, %317 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1324 = migraphx.add %171, %1323 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1325 = migraphx.add %1306, %1324 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1326 = migraphx.reduce_mean %1325 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1327 = migraphx.sub %1325, %1326 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1328 = migraphx.pow %1327, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1329 = migraphx.reduce_mean %1328 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1330 = migraphx.add %1329, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1331 = migraphx.sqrt %1330 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1332 = migraphx.div %1327, %1331 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1333 = migraphx.mul %1332, %178 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1334 = migraphx.add %1333, %179 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1335 = migraphx.dot %1334, %372 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %1336 = migraphx.slice %1335 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1337 = migraphx.slice %1335 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1338 = migraphx.slice %1335 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1339 = migraphx.add %176, %1336 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1340 = migraphx.mul %1339, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1341 = migraphx.add %174, %1337 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1342 = migraphx.reshape %1341 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1343 = migraphx.transpose %1342 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1344 = migraphx.add %175, %1338 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1345 = migraphx.reshape %1344 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1346 = migraphx.transpose %1345 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1347 = migraphx.reshape %1340 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1348 = migraphx.transpose %1347 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1349 = migraphx.reshape %1348 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1350 = migraphx.reshape %1343 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1351 = migraphx.reshape %1346 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1352 = migraphx.transpose %1350 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %1353 = migraphx.dot %1349, %1352 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %1354 = migraphx.softmax %1353 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %1355 = migraphx.dot %1354, %1351 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1356 = migraphx.reshape %1355 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1357 = migraphx.transpose %1356 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1358 = migraphx.reshape %1357 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %1359 = migraphx.dot %1358, %318 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1360 = migraphx.add %177, %1359 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1361 = migraphx.add %1325, %1360 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1362 = migraphx.reduce_mean %1361 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1363 = migraphx.sub %1361, %1362 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1364 = migraphx.pow %1363, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1365 = migraphx.reduce_mean %1364 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1366 = migraphx.add %1365, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1367 = migraphx.sqrt %1366 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1368 = migraphx.div %1363, %1367 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1369 = migraphx.mul %1368, %182 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1370 = migraphx.add %1369, %183 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1371 = migraphx.dot %1370, %319 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1372 = migraphx.add %180, %1371 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1373 = migraphx.div %1372, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1374 = migraphx.erf %1373 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1375 = migraphx.add %1374, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1376 = migraphx.mul %1372, %1375 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1377 = migraphx.mul %1376, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1378 = migraphx.dot %1377, %320 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1379 = migraphx.add %181, %1378 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1380 = migraphx.add %1361, %1379 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1381 = migraphx.reduce_mean %1380 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1382 = migraphx.sub %1380, %1381 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1383 = migraphx.pow %1382, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1384 = migraphx.reduce_mean %1383 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1385 = migraphx.add %1384, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1386 = migraphx.sqrt %1385 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1387 = migraphx.div %1382, %1386 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1388 = migraphx.mul %1387, %188 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1389 = migraphx.add %1388, %189 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1390 = migraphx.dot %1389, %373 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %1391 = migraphx.slice %1390 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1392 = migraphx.slice %1390 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1393 = migraphx.slice %1390 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1394 = migraphx.add %186, %1391 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1395 = migraphx.mul %1394, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1396 = migraphx.add %184, %1392 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1397 = migraphx.reshape %1396 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1398 = migraphx.transpose %1397 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1399 = migraphx.add %185, %1393 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1400 = migraphx.reshape %1399 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1401 = migraphx.transpose %1400 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1402 = migraphx.reshape %1395 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1403 = migraphx.transpose %1402 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1404 = migraphx.reshape %1403 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1405 = migraphx.reshape %1398 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1406 = migraphx.reshape %1401 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1407 = migraphx.transpose %1405 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %1408 = migraphx.dot %1404, %1407 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %1409 = migraphx.softmax %1408 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %1410 = migraphx.dot %1409, %1406 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1411 = migraphx.reshape %1410 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1412 = migraphx.transpose %1411 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1413 = migraphx.reshape %1412 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %1414 = migraphx.dot %1413, %321 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1415 = migraphx.add %187, %1414 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1416 = migraphx.add %1380, %1415 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1417 = migraphx.reduce_mean %1416 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1418 = migraphx.sub %1416, %1417 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1419 = migraphx.pow %1418, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1420 = migraphx.reduce_mean %1419 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1421 = migraphx.add %1420, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1422 = migraphx.sqrt %1421 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1423 = migraphx.div %1418, %1422 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1424 = migraphx.mul %1423, %192 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1425 = migraphx.add %1424, %193 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1426 = migraphx.dot %1425, %322 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1427 = migraphx.add %190, %1426 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1428 = migraphx.div %1427, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1429 = migraphx.erf %1428 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1430 = migraphx.add %1429, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1431 = migraphx.mul %1427, %1430 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1432 = migraphx.mul %1431, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1433 = migraphx.dot %1432, %323 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1434 = migraphx.add %191, %1433 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1435 = migraphx.add %1416, %1434 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1436 = migraphx.reduce_mean %1435 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1437 = migraphx.sub %1435, %1436 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1438 = migraphx.pow %1437, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1439 = migraphx.reduce_mean %1438 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1440 = migraphx.add %1439, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1441 = migraphx.sqrt %1440 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1442 = migraphx.div %1437, %1441 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1443 = migraphx.mul %1442, %198 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1444 = migraphx.add %1443, %199 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1445 = migraphx.dot %1444, %374 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %1446 = migraphx.slice %1445 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1447 = migraphx.slice %1445 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1448 = migraphx.slice %1445 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1449 = migraphx.add %196, %1446 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1450 = migraphx.mul %1449, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1451 = migraphx.add %194, %1447 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1452 = migraphx.reshape %1451 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1453 = migraphx.transpose %1452 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1454 = migraphx.add %195, %1448 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1455 = migraphx.reshape %1454 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1456 = migraphx.transpose %1455 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1457 = migraphx.reshape %1450 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1458 = migraphx.transpose %1457 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1459 = migraphx.reshape %1458 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1460 = migraphx.reshape %1453 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1461 = migraphx.reshape %1456 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1462 = migraphx.transpose %1460 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %1463 = migraphx.dot %1459, %1462 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %1464 = migraphx.softmax %1463 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %1465 = migraphx.dot %1464, %1461 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1466 = migraphx.reshape %1465 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1467 = migraphx.transpose %1466 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1468 = migraphx.reshape %1467 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %1469 = migraphx.dot %1468, %324 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1470 = migraphx.add %197, %1469 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1471 = migraphx.add %1435, %1470 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1472 = migraphx.reduce_mean %1471 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1473 = migraphx.sub %1471, %1472 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1474 = migraphx.pow %1473, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1475 = migraphx.reduce_mean %1474 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1476 = migraphx.add %1475, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1477 = migraphx.sqrt %1476 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1478 = migraphx.div %1473, %1477 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1479 = migraphx.mul %1478, %202 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1480 = migraphx.add %1479, %203 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1481 = migraphx.dot %1480, %325 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1482 = migraphx.add %200, %1481 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1483 = migraphx.div %1482, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1484 = migraphx.erf %1483 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1485 = migraphx.add %1484, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1486 = migraphx.mul %1482, %1485 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1487 = migraphx.mul %1486, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1488 = migraphx.dot %1487, %326 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1489 = migraphx.add %201, %1488 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1490 = migraphx.add %1471, %1489 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1491 = migraphx.reduce_mean %1490 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1492 = migraphx.sub %1490, %1491 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1493 = migraphx.pow %1492, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1494 = migraphx.reduce_mean %1493 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1495 = migraphx.add %1494, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1496 = migraphx.sqrt %1495 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1497 = migraphx.div %1492, %1496 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1498 = migraphx.mul %1497, %208 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1499 = migraphx.add %1498, %209 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1500 = migraphx.dot %1499, %375 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %1501 = migraphx.slice %1500 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1502 = migraphx.slice %1500 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1503 = migraphx.slice %1500 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1504 = migraphx.add %206, %1501 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1505 = migraphx.mul %1504, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1506 = migraphx.add %204, %1502 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1507 = migraphx.reshape %1506 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1508 = migraphx.transpose %1507 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1509 = migraphx.add %205, %1503 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1510 = migraphx.reshape %1509 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1511 = migraphx.transpose %1510 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1512 = migraphx.reshape %1505 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1513 = migraphx.transpose %1512 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1514 = migraphx.reshape %1513 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1515 = migraphx.reshape %1508 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1516 = migraphx.reshape %1511 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1517 = migraphx.transpose %1515 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %1518 = migraphx.dot %1514, %1517 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %1519 = migraphx.softmax %1518 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %1520 = migraphx.dot %1519, %1516 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1521 = migraphx.reshape %1520 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1522 = migraphx.transpose %1521 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1523 = migraphx.reshape %1522 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %1524 = migraphx.dot %1523, %327 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1525 = migraphx.add %207, %1524 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1526 = migraphx.add %1490, %1525 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1527 = migraphx.reduce_mean %1526 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1528 = migraphx.sub %1526, %1527 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1529 = migraphx.pow %1528, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1530 = migraphx.reduce_mean %1529 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1531 = migraphx.add %1530, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1532 = migraphx.sqrt %1531 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1533 = migraphx.div %1528, %1532 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1534 = migraphx.mul %1533, %212 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1535 = migraphx.add %1534, %213 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1536 = migraphx.dot %1535, %328 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1537 = migraphx.add %210, %1536 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1538 = migraphx.div %1537, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1539 = migraphx.erf %1538 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1540 = migraphx.add %1539, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1541 = migraphx.mul %1537, %1540 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1542 = migraphx.mul %1541, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1543 = migraphx.dot %1542, %329 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1544 = migraphx.add %211, %1543 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1545 = migraphx.add %1526, %1544 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1546 = migraphx.reduce_mean %1545 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1547 = migraphx.sub %1545, %1546 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1548 = migraphx.pow %1547, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1549 = migraphx.reduce_mean %1548 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1550 = migraphx.add %1549, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1551 = migraphx.sqrt %1550 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1552 = migraphx.div %1547, %1551 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1553 = migraphx.mul %1552, %218 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1554 = migraphx.add %1553, %219 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1555 = migraphx.dot %1554, %376 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %1556 = migraphx.slice %1555 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1557 = migraphx.slice %1555 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1558 = migraphx.slice %1555 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1559 = migraphx.add %216, %1556 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1560 = migraphx.mul %1559, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1561 = migraphx.add %214, %1557 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1562 = migraphx.reshape %1561 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1563 = migraphx.transpose %1562 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1564 = migraphx.add %215, %1558 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1565 = migraphx.reshape %1564 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1566 = migraphx.transpose %1565 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1567 = migraphx.reshape %1560 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1568 = migraphx.transpose %1567 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1569 = migraphx.reshape %1568 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1570 = migraphx.reshape %1563 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1571 = migraphx.reshape %1566 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1572 = migraphx.transpose %1570 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %1573 = migraphx.dot %1569, %1572 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %1574 = migraphx.softmax %1573 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %1575 = migraphx.dot %1574, %1571 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1576 = migraphx.reshape %1575 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1577 = migraphx.transpose %1576 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1578 = migraphx.reshape %1577 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %1579 = migraphx.dot %1578, %330 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1580 = migraphx.add %217, %1579 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1581 = migraphx.add %1545, %1580 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1582 = migraphx.reduce_mean %1581 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1583 = migraphx.sub %1581, %1582 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1584 = migraphx.pow %1583, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1585 = migraphx.reduce_mean %1584 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1586 = migraphx.add %1585, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1587 = migraphx.sqrt %1586 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1588 = migraphx.div %1583, %1587 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1589 = migraphx.mul %1588, %222 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1590 = migraphx.add %1589, %223 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1591 = migraphx.dot %1590, %331 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1592 = migraphx.add %220, %1591 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1593 = migraphx.div %1592, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1594 = migraphx.erf %1593 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1595 = migraphx.add %1594, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1596 = migraphx.mul %1592, %1595 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1597 = migraphx.mul %1596, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1598 = migraphx.dot %1597, %332 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1599 = migraphx.add %221, %1598 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1600 = migraphx.add %1581, %1599 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1601 = migraphx.reduce_mean %1600 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1602 = migraphx.sub %1600, %1601 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1603 = migraphx.pow %1602, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1604 = migraphx.reduce_mean %1603 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1605 = migraphx.add %1604, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1606 = migraphx.sqrt %1605 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1607 = migraphx.div %1602, %1606 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1608 = migraphx.mul %1607, %228 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1609 = migraphx.add %1608, %229 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1610 = migraphx.dot %1609, %377 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %1611 = migraphx.slice %1610 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1612 = migraphx.slice %1610 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1613 = migraphx.slice %1610 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1614 = migraphx.add %226, %1611 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1615 = migraphx.mul %1614, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1616 = migraphx.add %224, %1612 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1617 = migraphx.reshape %1616 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1618 = migraphx.transpose %1617 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1619 = migraphx.add %225, %1613 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1620 = migraphx.reshape %1619 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1621 = migraphx.transpose %1620 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1622 = migraphx.reshape %1615 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1623 = migraphx.transpose %1622 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1624 = migraphx.reshape %1623 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1625 = migraphx.reshape %1618 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1626 = migraphx.reshape %1621 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1627 = migraphx.transpose %1625 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %1628 = migraphx.dot %1624, %1627 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %1629 = migraphx.softmax %1628 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %1630 = migraphx.dot %1629, %1626 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1631 = migraphx.reshape %1630 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1632 = migraphx.transpose %1631 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1633 = migraphx.reshape %1632 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %1634 = migraphx.dot %1633, %333 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1635 = migraphx.add %227, %1634 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1636 = migraphx.add %1600, %1635 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1637 = migraphx.reduce_mean %1636 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1638 = migraphx.sub %1636, %1637 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1639 = migraphx.pow %1638, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1640 = migraphx.reduce_mean %1639 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1641 = migraphx.add %1640, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1642 = migraphx.sqrt %1641 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1643 = migraphx.div %1638, %1642 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1644 = migraphx.mul %1643, %232 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1645 = migraphx.add %1644, %233 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1646 = migraphx.dot %1645, %334 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1647 = migraphx.add %230, %1646 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1648 = migraphx.div %1647, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1649 = migraphx.erf %1648 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1650 = migraphx.add %1649, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1651 = migraphx.mul %1647, %1650 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1652 = migraphx.mul %1651, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1653 = migraphx.dot %1652, %335 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1654 = migraphx.add %231, %1653 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1655 = migraphx.add %1636, %1654 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1656 = migraphx.reduce_mean %1655 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1657 = migraphx.sub %1655, %1656 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1658 = migraphx.pow %1657, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1659 = migraphx.reduce_mean %1658 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1660 = migraphx.add %1659, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1661 = migraphx.sqrt %1660 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1662 = migraphx.div %1657, %1661 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1663 = migraphx.mul %1662, %238 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1664 = migraphx.add %1663, %239 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1665 = migraphx.dot %1664, %378 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %1666 = migraphx.slice %1665 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1667 = migraphx.slice %1665 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1668 = migraphx.slice %1665 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1669 = migraphx.add %236, %1666 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1670 = migraphx.mul %1669, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1671 = migraphx.add %234, %1667 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1672 = migraphx.reshape %1671 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1673 = migraphx.transpose %1672 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1674 = migraphx.add %235, %1668 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1675 = migraphx.reshape %1674 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1676 = migraphx.transpose %1675 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1677 = migraphx.reshape %1670 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1678 = migraphx.transpose %1677 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1679 = migraphx.reshape %1678 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1680 = migraphx.reshape %1673 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1681 = migraphx.reshape %1676 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1682 = migraphx.transpose %1680 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %1683 = migraphx.dot %1679, %1682 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %1684 = migraphx.softmax %1683 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %1685 = migraphx.dot %1684, %1681 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1686 = migraphx.reshape %1685 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1687 = migraphx.transpose %1686 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1688 = migraphx.reshape %1687 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %1689 = migraphx.dot %1688, %336 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1690 = migraphx.add %237, %1689 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1691 = migraphx.add %1655, %1690 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1692 = migraphx.reduce_mean %1691 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1693 = migraphx.sub %1691, %1692 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1694 = migraphx.pow %1693, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1695 = migraphx.reduce_mean %1694 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1696 = migraphx.add %1695, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1697 = migraphx.sqrt %1696 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1698 = migraphx.div %1693, %1697 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1699 = migraphx.mul %1698, %242 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1700 = migraphx.add %1699, %243 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1701 = migraphx.dot %1700, %337 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1702 = migraphx.add %240, %1701 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1703 = migraphx.div %1702, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1704 = migraphx.erf %1703 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1705 = migraphx.add %1704, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1706 = migraphx.mul %1702, %1705 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1707 = migraphx.mul %1706, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1708 = migraphx.dot %1707, %338 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1709 = migraphx.add %241, %1708 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1710 = migraphx.add %1691, %1709 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1711 = migraphx.reduce_mean %1710 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1712 = migraphx.sub %1710, %1711 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1713 = migraphx.pow %1712, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1714 = migraphx.reduce_mean %1713 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1715 = migraphx.add %1714, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1716 = migraphx.sqrt %1715 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1717 = migraphx.div %1712, %1716 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1718 = migraphx.mul %1717, %248 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1719 = migraphx.add %1718, %249 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1720 = migraphx.dot %1719, %379 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %1721 = migraphx.slice %1720 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1722 = migraphx.slice %1720 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1723 = migraphx.slice %1720 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1724 = migraphx.add %246, %1721 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1725 = migraphx.mul %1724, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1726 = migraphx.add %244, %1722 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1727 = migraphx.reshape %1726 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1728 = migraphx.transpose %1727 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1729 = migraphx.add %245, %1723 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1730 = migraphx.reshape %1729 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1731 = migraphx.transpose %1730 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1732 = migraphx.reshape %1725 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1733 = migraphx.transpose %1732 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1734 = migraphx.reshape %1733 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1735 = migraphx.reshape %1728 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1736 = migraphx.reshape %1731 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1737 = migraphx.transpose %1735 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %1738 = migraphx.dot %1734, %1737 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %1739 = migraphx.softmax %1738 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %1740 = migraphx.dot %1739, %1736 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1741 = migraphx.reshape %1740 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1742 = migraphx.transpose %1741 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1743 = migraphx.reshape %1742 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %1744 = migraphx.dot %1743, %339 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1745 = migraphx.add %247, %1744 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1746 = migraphx.add %1710, %1745 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1747 = migraphx.reduce_mean %1746 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1748 = migraphx.sub %1746, %1747 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1749 = migraphx.pow %1748, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1750 = migraphx.reduce_mean %1749 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1751 = migraphx.add %1750, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1752 = migraphx.sqrt %1751 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1753 = migraphx.div %1748, %1752 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1754 = migraphx.mul %1753, %252 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1755 = migraphx.add %1754, %253 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1756 = migraphx.dot %1755, %340 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1757 = migraphx.add %250, %1756 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1758 = migraphx.div %1757, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1759 = migraphx.erf %1758 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1760 = migraphx.add %1759, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1761 = migraphx.mul %1757, %1760 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1762 = migraphx.mul %1761, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1763 = migraphx.dot %1762, %341 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1764 = migraphx.add %251, %1763 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1765 = migraphx.add %1746, %1764 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1766 = migraphx.reduce_mean %1765 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1767 = migraphx.sub %1765, %1766 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1768 = migraphx.pow %1767, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1769 = migraphx.reduce_mean %1768 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1770 = migraphx.add %1769, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1771 = migraphx.sqrt %1770 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1772 = migraphx.div %1767, %1771 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1773 = migraphx.mul %1772, %258 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1774 = migraphx.add %1773, %259 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1775 = migraphx.dot %1774, %380 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %1776 = migraphx.slice %1775 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1777 = migraphx.slice %1775 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1778 = migraphx.slice %1775 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1779 = migraphx.add %256, %1776 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1780 = migraphx.mul %1779, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1781 = migraphx.add %254, %1777 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1782 = migraphx.reshape %1781 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1783 = migraphx.transpose %1782 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1784 = migraphx.add %255, %1778 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1785 = migraphx.reshape %1784 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1786 = migraphx.transpose %1785 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1787 = migraphx.reshape %1780 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1788 = migraphx.transpose %1787 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1789 = migraphx.reshape %1788 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1790 = migraphx.reshape %1783 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1791 = migraphx.reshape %1786 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1792 = migraphx.transpose %1790 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %1793 = migraphx.dot %1789, %1792 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %1794 = migraphx.softmax %1793 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %1795 = migraphx.dot %1794, %1791 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1796 = migraphx.reshape %1795 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1797 = migraphx.transpose %1796 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1798 = migraphx.reshape %1797 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %1799 = migraphx.dot %1798, %342 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1800 = migraphx.add %257, %1799 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1801 = migraphx.add %1765, %1800 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1802 = migraphx.reduce_mean %1801 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1803 = migraphx.sub %1801, %1802 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1804 = migraphx.pow %1803, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1805 = migraphx.reduce_mean %1804 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1806 = migraphx.add %1805, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1807 = migraphx.sqrt %1806 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1808 = migraphx.div %1803, %1807 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1809 = migraphx.mul %1808, %262 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1810 = migraphx.add %1809, %263 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1811 = migraphx.dot %1810, %343 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1812 = migraphx.add %260, %1811 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1813 = migraphx.div %1812, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1814 = migraphx.erf %1813 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1815 = migraphx.add %1814, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1816 = migraphx.mul %1812, %1815 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1817 = migraphx.mul %1816, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1818 = migraphx.dot %1817, %344 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1819 = migraphx.add %261, %1818 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1820 = migraphx.add %1801, %1819 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1821 = migraphx.reduce_mean %1820 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1822 = migraphx.sub %1820, %1821 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1823 = migraphx.pow %1822, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1824 = migraphx.reduce_mean %1823 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1825 = migraphx.add %1824, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1826 = migraphx.sqrt %1825 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1827 = migraphx.div %1822, %1826 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1828 = migraphx.mul %1827, %268 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1829 = migraphx.add %1828, %269 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1830 = migraphx.dot %1829, %381 : <1x31x1024xf16, 31744x1024x1>, <1024x3072xf16, 3072x1> -> <1x31x3072xf16, 95232x3072x1>
    %1831 = migraphx.slice %1830 {axes = [2], ends = [1024], starts = [0]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1832 = migraphx.slice %1830 {axes = [2], ends = [2048], starts = [1024]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1833 = migraphx.slice %1830 {axes = [2], ends = [3072], starts = [2048]} : <1x31x3072xf16, 95232x3072x1> -> <1x31x1024xf16, 31744x1024x1>
    %1834 = migraphx.add %266, %1831 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1835 = migraphx.mul %1834, %355 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1836 = migraphx.add %264, %1832 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1837 = migraphx.reshape %1836 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1838 = migraphx.transpose %1837 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1839 = migraphx.add %265, %1833 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1840 = migraphx.reshape %1839 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1841 = migraphx.transpose %1840 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1842 = migraphx.reshape %1835 {dims = [1, 31, 16, 64]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1843 = migraphx.transpose %1842 {permutation = [0, 2, 1, 3]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1844 = migraphx.reshape %1843 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1845 = migraphx.reshape %1838 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1846 = migraphx.reshape %1841 {dims = [16, 31, 64]} : <1x16x31x64xf16, 31744x1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1847 = migraphx.transpose %1845 {permutation = [0, 2, 1]} : <16x31x64xf16, 1984x64x1> -> <16x64x31xf16, 1984x31x1>
    %1848 = migraphx.dot %1844, %1847 : <16x31x64xf16, 1984x64x1>, <16x64x31xf16, 1984x31x1> -> <16x31x31xf16, 961x31x1>
    %1849 = migraphx.softmax %1848 {axis = 2 : i64} : <16x31x31xf16, 961x31x1> -> <16x31x31xf16, 961x31x1>
    %1850 = migraphx.dot %1849, %1846 : <16x31x31xf16, 961x31x1>, <16x31x64xf16, 1984x64x1> -> <16x31x64xf16, 1984x64x1>
    %1851 = migraphx.reshape %1850 {dims = [1, 16, 31, 64]} : <16x31x64xf16, 1984x64x1> -> <1x16x31x64xf16, 31744x1984x64x1>
    %1852 = migraphx.transpose %1851 {permutation = [0, 2, 1, 3]} : <1x16x31x64xf16, 31744x1984x64x1> -> <1x31x16x64xf16, 31744x1024x64x1>
    %1853 = migraphx.reshape %1852 {dims = [1, 31, 1024]} : <1x31x16x64xf16, 31744x1024x64x1> -> <1x31x1024xf16, 31744x1024x1>
    %1854 = migraphx.dot %1853, %345 : <1x31x1024xf16, 31744x1024x1>, <1024x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1855 = migraphx.add %267, %1854 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1856 = migraphx.add %1820, %1855 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1857 = migraphx.reduce_mean %1856 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1858 = migraphx.sub %1856, %1857 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1859 = migraphx.pow %1858, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1860 = migraphx.reduce_mean %1859 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1861 = migraphx.add %1860, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1862 = migraphx.sqrt %1861 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1863 = migraphx.div %1858, %1862 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1864 = migraphx.mul %1863, %272 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1865 = migraphx.add %1864, %273 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1866 = migraphx.dot %1865, %346 : <1x31x1024xf16, 31744x1024x1>, <1024x4096xf16, 4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1867 = migraphx.add %270, %1866 : <4096xf16, 1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1868 = migraphx.div %1867, %353 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1869 = migraphx.erf %1868 : <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1870 = migraphx.add %1869, %349 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1871 = migraphx.mul %1867, %1870 : <1x31x4096xf16, 126976x4096x1>, <1x31x4096xf16, 126976x4096x1> -> <1x31x4096xf16, 126976x4096x1>
    %1872 = migraphx.mul %1871, %354 : <1x31x4096xf16, 126976x4096x1>, <1xf16, 1> -> <1x31x4096xf16, 126976x4096x1>
    %1873 = migraphx.dot %1872, %347 : <1x31x4096xf16, 126976x4096x1>, <4096x1024xf16, 1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1874 = migraphx.add %271, %1873 : <1024xf16, 1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1875 = migraphx.add %1856, %1874 : <1x31x1024xf16, 31744x1024x1>, <1x31x1024xf16, 31744x1024x1> -> <1x31x1024xf16, 31744x1024x1>
    %1876 = migraphx.reduce_mean %1875 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1877 = migraphx.sub %1875, %1876 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1878 = migraphx.pow %1877, %351 : <1x31x1024xf16, 31744x1024x1>, <1xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1879 = migraphx.reduce_mean %1878 {axes = [-1]} : <1x31x1024xf16, 31744x1024x1> -> <1x31x1xf16, 31x1x1>
    %1880 = migraphx.add %1879, %352 : <1x31x1xf16, 31x1x1>, <1xf16, 1> -> <1x31x1xf16, 31x1x1>
    %1881 = migraphx.sqrt %1880 : <1x31x1xf16, 31x1x1> -> <1x31x1xf16, 31x1x1>
    %1882 = migraphx.div %1877, %1881 : <1x31x1024xf16, 31744x1024x1>, <1x31x1xf16, 31x1x1> -> <1x31x1024xf16, 31744x1024x1>
    %1883 = migraphx.mul %1882, %32 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1884 = migraphx.add %1883, %33 : <1x31x1024xf16, 31744x1024x1>, <1024xf16, 1> -> <1x31x1024xf16, 31744x1024x1>
    %1885 = migraphx.reduce_mean %1884 {axes = [1]} : <1x31x1024xf16, 31744x1024x1> -> <1x1024xf16, 1024x1>
    %1886 = migraphx.dot %1885, %348 : <1x1024xf16, 1024x1>, <1024x6xf16, 6x1> -> <1x6xf16, 6x1>
    %1887 = migraphx.reshape %1886 {dims = [1, 1, 6]} : <1x6xf16, 6x1> -> <1x1x6xf16, 6x6x1>
    %1888 = migraphx.reduce_mean %1887 {axes = [1]} : <1x1x6xf16, 6x6x1> -> <1x6xf16, 6x1>
    return %1888 : !migraphx.shaped<1x6xf16, 6x1>
  }
}
