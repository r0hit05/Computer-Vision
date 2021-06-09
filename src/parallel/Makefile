CPPFLAGS = $(shell pkg-config --cflags opencv4)
LDLIBS = $(shell pkg-config --libs opencv4)

FLAGS = 

%: %.cpp
	g++ -o $@.o $< $(CPPFLAGS) -I/home/r0hit2005/OpenCV-Sample-Codes/src $(LDLIBS) $(FLAGS)
