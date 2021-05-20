CPPFLAGS = $(shell pkg-config --cflags opencv4)
LDLIBS = $(shell pkg-config --libs opencv4)

FLAGS = 

%: %.cpp
	g++   -o $@ $< $(CPPFLAGS) $(LDLIBS) $(FLAGS)
