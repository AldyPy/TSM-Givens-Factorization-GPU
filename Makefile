GCC		:=gcc
NVCC	:=nvcc --use_fast_math --cudart=static -ccbin gcc -Xcompiler -fopenmp
CFLAGS	:=-O3 

SRC_DIR			:=src
BIN_DIR			:=bin
INC_DIR			:=include
TEST_DATA_DIR 	:=tests

LIBS	:=-lcusolver -lcusolverMg -lcurand

ARCHES :=-gencode arch=compute_80,code=\"compute_80,sm_80\"

SOURCES :=qr_demo gen_matrix

all: $(SOURCES)
.PHONY: all

test_data: $(BIN_DIR)/gen_matrix
	$(BIN_DIR)/gen_matrix 100 100 > $(TEST_DATA_DIR)/test

qr_demo:
	$(NVCC) $(CFLAGS) -I$(INC_DIR) ${ARCHES} $(SRC_DIR)/qr_demo.cu -o $(BIN_DIR)/$@ $(LIBS)

gen_matrix:
	$(GCC) $(CFLAGS) -I$(INC_DIR) $(SRC_DIR)/gen_matrix.c -o $(BIN_DIR)/$@