CC := g++ -std=c++11
CC_OPTS := -O3
CFLAGS := -c -Wall -funroll-loops
INC := -I${HOME}/local/include -I./include
LIB := -L${HOME}/local/lib
DEBUG_INFO :=

ifndef BLAS_MODE
BLAS_MODE := openblas
endif

ifeq ($(BLAS_MODE), mkl)
BLASLIB := -mkl
CC := icc -std=c++11
DEBUG_INFO := $(DEBUG_INFO) -DUSE_MKL
else ifeq ($(BLAS_MODE), openblas)
BLASLIB := -lopenblas
else
$(error invalid option: BLAS_MODE=$(BLAS_MODE))
endif

LDFLAGS := -lboost_thread -lboost_system -lboost_random -lglog -lpthread -lboost_serialization -lboost_filesystem -lboost_program_options -lhdf5 -lhdf5_cpp
LDFLAGS := $(LDFLAGS) $(BLASLIB)
#$(info $$LDFLAGS is [${LDFLAGS}])

SRCDIR := src
BUILDDIR := bin
MAIN_SRCDIR := aitextml
HDF5CONVERTER_SRCDIR := converter
HEADERS:=$(wildcard include/*.hpp)

SRCS:=$(wildcard $(SRCDIR)/*.cpp)
MAIN_SRCS:=$(wildcard $(MAIN_SRCDIR)/*.cpp)
HDF5CONVERTER_SRCS:=$(wildcard $(HDF5CONVERTER_SRCDIR)/*.cpp)

OBJS:=$(patsubst $(SRCDIR)/%.cpp,$(BUILDDIR)/%.o,$(SRCS))
MAIN_OBJS:=$(patsubst $(MAIN_SRCDIR)/%.cpp,$(BUILDDIR)/%.o,$(MAIN_SRCS))
HDF5CONVERTER_OBJS:=$(patsubst $(HDF5CONVERTER_SRCDIR)/%.cpp,$(BUILDDIR)/%.o,$(HDF5CONVERTER_SRCS))

MAIN_TARGET := $(BUILDDIR)/aitextml
HDF5CONVERTER_TARGET := $(BUILDDIR)/hdf5_converter

.DEFAULT: all

all: $(SRCS) $(MAIN_TARGET) $(HDF5CONVERTER_TARGET)

$(MAIN_TARGET): $(OBJS) $(MAIN_OBJS)
	${CC} $^ -o $(MAIN_TARGET) $(CC_OPTS) $(INC) $(LIB) $(LDFLAGS) $(DEBUG_INFO)

$(HDF5CONVERTER_TARGET): $(OBJS) $(HDF5CONVERTER_OBJS)
	${CC} $^ -o $(HDF5CONVERTER_TARGET) $(CC_OPTS) $(INC) $(LIB) $(LDFLAGS) $(DEBUG_INFO)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp $(HEADERS)
	$(CC) $< $(CC_OPTS) $(CFLAGS) -o $@ $(INC) $(DEBUG_INFO)

$(BUILDDIR)/%.o: $(MAIN_SRCDIR)/%.cpp $(HEADERS)
	$(CC) $< $(CC_OPTS) $(CFLAGS) -o $@ $(INC) $(DEBUG_INFO)

$(BUILDDIR)/%.o: $(HDF5CONVERTER_SRCDIR)/%.cpp $(HEADERS)
	$(CC) $< $(CC_OPTS) $(CFLAGS) -o $@ $(INC) $(DEBUG_INFO)

.PHONY: clean
clean:
	rm $(BUILDDIR)/*.o $(MAIN_TARGET) $(HDF5CONVERTER_TARGET)
