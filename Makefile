# Single-threaded, zero-dependency C++20 Gemma-style inference engine.
# Requires: a C++20 compiler (Apple clang 14+ / GCC 10+), Python 3 + numpy for tests.

CXX      := clang++
CXXFLAGS := -std=c++20 -O3 -Wall -Wextra -fno-exceptions

# NEON + FMA are part of the arm64 baseline, so nothing extra is needed there
# (and -march=native has known quirks on Apple clang). x86-64 gets the AVX2
# build through -march=native.
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
CXXFLAGS += -march=native
endif

SRC     := src/weights.cpp src/kernel.cpp src/ops.cpp \
           src/tokenizer.cpp src/sampler.cpp src/model.cpp
OBJ     := $(SRC:src/%.cpp=build/%.o)

all: gemma unittests integration

build:
	mkdir -p build

build/%.o: src/%.cpp src/%.h src/config.h src/tensor.h | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/gemma: src/main.cpp $(OBJ) | build
	$(CXX) $(CXXFLAGS) src/main.cpp $(OBJ) -o $@

build/unit_tests: tests/unit_tests.cpp $(OBJ) | build
	$(CXX) $(CXXFLAGS) -Isrc tests/unit_tests.cpp $(OBJ) -o $@

build/integration: tests/integration.cpp $(OBJ) | build
	$(CXX) $(CXXFLAGS) -Isrc tests/integration.cpp $(OBJ) -o $@

gemma: build/gemma
unittests: build/unit_tests
integration: build/integration

# Regenerate all test fixtures and golden data (deterministic, fixed seeds).
golden:
	tests/scripts/create_test_data.sh

test: gemma unittests integration golden
	tests/scripts/run_tests.sh
	tests/scripts/run_integration.sh
	@echo "ALL TESTS PASSED"

clean:
	rm -rf build weights tests/golden

.PHONY: all gemma unittests integration golden test clean
