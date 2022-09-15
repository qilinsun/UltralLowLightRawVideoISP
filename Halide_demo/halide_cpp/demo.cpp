#include <stdio.h>

#include "Halide.h"

using namespace std;
using namespace Halide;

int main(int argc, char **argv)
{
    // tutorial 1
    // g++ lesson_01*.cpp -g -I <path/to/Halide.h> -L <path/to/libHalide.so> -lHalide -o lesson_01 -std=c++17
    // DYLD_LIBRARY_PATH=<path/to/libHalide.dylib>
    // Halide::Func gradient;
    // Halide::Var x, y;
    // Halide::Expr e = x + y;
    // gradient(x,y) = e;
    // Halide::Buffer<int32_t> output = gradient.realize({400,300}); //w,h
    // for (int j = 0; j < output.height(); j++) {
    //     for (int i = 0; i < output.width(); i++) {
    //         printf("Pixel %d, %d was supposed to be %d, fortunately it !is! %d\n",i, j, i + j, output(i, j));
    //     }
    // }


    // tutorial 2
    // #include "halide_image_io.h"
    // using namespace Halide::Tools;
    // g++ lesson_02*.cpp -g -I <path/to/Halide.h> -I <path/to/tools/halide_image_io.h> -L <path/to/libHalide.so> -lHalide `libpng-config --cflags --ldflags` -ljpeg -o lesson_02 -std=c++17
    // DYLD_LIBRARY_PATH=<path/to/libHalide.dylib> ./lesson_02
    // Halide::Buffer<uint8_t> input = load_image("/Users/gravitychen/Desktop/Picture1.jpg"); //w,h
    // save_image(input,"demo.png");


    // tutorial 3 : generate HTML to debug
    // g++ lesson_03*.cpp -g -I <path/to/Halide.h> -L <path/to/libHalide.so> -lHalide -o lesson_03 -std=c++17
    // Halide::Func gradient("Hello_I_am_a_debugger");
    // Halide::Var x("x"), y("y");
    // gradient(x,y) = x+y;
    // Halide::Buffer<int> output = gradient.realize({40,40});
    // gradient.compile_to_lowered_stmt("demo.html",{},Halide::HTML);


    // tutorial 4 : Debugging with tracing, print, and print_when
    // tutorial 4-1 : Debugging with tracing,
    // g++ lesson_04*.cpp -g -I <path/to/Halide.h> -L <path/to/libHalide.so> -lHalide -o lesson_04 -std=c++17
    // ===========
    // Var w("w"),h("h");
    // Func gradient("gradient");
    // gradient.trace_stores();
    // gradient(w,h) = w+h;
    // Buffer<int32_t> output = gradient.realize({4,2}); //w,h

    // Output:
    // Begin pipeline gradient.0()
    // Tag gradient.0() tag = "func_type_and_dim: 1 0 32 1 2 0 4 0 2"
    // Store gradient.0(0, 0) = 0// Store gradient.0(1, 0) = 1// Store gradient.0(2, 0) = 2
    // Store gradient.0(3, 0) = 3// Store gradient.0(0, 1) = 1// Store gradient.0(1, 1) = 2
    // Store gradient.0(2, 1) = 3// Store gradient.0(3, 1) = 4
    // End pipeline gradient.0()
    // halide works
    // ==========
    // tutorial 4-2 : Debugging with tracing (parallel)
    // Var w("w"),h("h");
    // Func parallel_gradient("parallel_gradient");
    // parallel_gradient.trace_stores();
    // parallel_gradient(w,h) = w+h;
    // //      Now we tell Halide to use a parallel for loop over the y
    // //      coordinate. On Linux we run this using a thread pool and a task
    // //      queue. On OS X we call into grand central dispatch, which does // =========== wtf is grand central dispatch https://justinyan.me/post/2420
    // //      the same thing for us.
    // parallel_gradient.parallel(h);
    // parallel_gradient.realize({4,4}); //w,h  赋值的话不能平行

    // Output:
    // Store parallel_gradient.0(0, 0) = 0// Store parallel_gradient.0(1, 0) = 1// Store parallel_gradient.0(2, 0) = 2
    // Store parallel_gradient.0(3, 0) = 3// Store parallel_gradient.0(0, 2) = 2// Store parallel_gradient.0(1, 2) = 3
    // Store parallel_gradient.0(0, 1) = 1// Store parallel_gradient.0(2, 2) = 4// Store parallel_gradient.0(1, 1) = 2
    // Store parallel_gradient.0(3, 2) = 5// Store parallel_gradient.0(2, 1) = 3// Store parallel_gradient.0(0, 3) = 3
    // Store parallel_gradient.0(3, 1) = 4// Store parallel_gradient.0(1, 3) = 4// Store parallel_gradient.0(2, 3) = 5
    // Store parallel_gradient.0(3, 3) = 6
    // End pipeline parallel_gradient.0()
    // halide works


    // ===========
    // tutorial 4-2 : print individual exprs
    Var w("w"),h("h");
    Func f;
    f(w, h) = sin(w) + h;
    // If we want to inspect just one of the terms, we can wrap
    Func g;
    g(w, h) = sin(w) + print(h);
    g.parallel(h);
    printf("\nEvaluating sin(w) + cos(h), and just printing cos(h)\n");
    g.realize({4, 4});
    
    // Output:
    // 0  2  3  1  2  3  1  2  1  3
    // 2  1  3  0  0  0

    
    // ===========
    // tutorial 4-2 : print individual exprs

    // Output:


    // ===========
    // tutorial 4-2 : print individual exprs

    // Output:


    // ===========
    // tutorial 4-2 : print individual exprs

    // Output:


    // ===========
    // tutorial 4-2 : print individual exprs

    // Output:
    cout << "halide works";
    return 0;
}

// C++编译之提示ld: can‘t open output file for writing: test1, errno=21 for architecture x86_64
// solution : 文件夹和文件重名了

// 1 step build and run in vscode c++ mac , 
// edit tasks.json --> edit keybindings.json --> {"key": "alt+r","command": "workbench.action.tasks.runTask","args": "Build_run"}
// 
