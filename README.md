# Computer Vision Repository
The repository contains sample programs written in C++ in domains of Computer Vision. This is a part of a larger repository containing my dev projects. Projects are purely for educational purposes and meant mostly for personal use.

## Projects 

### Mandelbrot 
The program generates a mandelbrot plot and allows the user to benchmark between different methods of optimization. The user may implement their own methods for this purpose.  
The console window outputs the generation time in seconds. It also captures the number of iterations, the scale and the current method of generation.  

> ### Controls
> - **Pan:** Click and drag  
> - **Zoom:** q to zoom in, e to zoom out
> - **Number of iterations:** n to decrease, m to increase  
> - **Toggle b/w methods of generation:** t   


### Billiards 
The program takes an image of a white ball representing the initial cue ball position and a red line representing the direction of the cue stick. On running the program, based on the image, the cue ball moves around the table. We can input the number of collisions after which the program terminates.



## Build

All the source files need to be linked to the respective libraries and should also point at the directory where it will look for header files. Each project is in its own folder. Currently, each project consists of a single source file. For complicated projects, I'll add specific instructions.  
Currently it's just opencv libraries and header files and the Timer.h file for some projects. 
  
> ### Using g++:
> Use the g++ command and add include directories and library directory and files. You may also have them in your environment variables or compiler paths.  
> **Note:** The command would look like the following. lib1, lib2 etc are names of the library files (ending with .a or .so). When linking, don't add the extension. 

    g++ filename.cpp -I/path/headerfiles/ -L/path/library/directory -llib1 -llib2 -llib3 
   


 > ### Using IDEs:
 > Look into the corresponding IDE documentation to find out how to add include paths and link.
    

