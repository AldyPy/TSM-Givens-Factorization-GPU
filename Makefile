GCC		:=gcc
NVCC	:=nvcc -g -G -lineinfo  --cudart=static -ccbin gcc -Xcompiler -fopenmp
CFLAGS	:=-O0

SRC_DIR			:=src
BIN_DIR			:=bin
INC_DIR			:=include
TEST_DATA_DIR 	:=tests

LIBS	:=-lcusolver -lcusolverMg -lcurand

ARCHES :=-gencode arch=compute_75,code=\"compute_75,sm_75\"

SOURCES :=qr_demo gen_matrix

ifeq (run,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "run"
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(RUN_ARGS):;@:)
endif


all: $(SOURCES)
.PHONY: all

run:
	$(BIN_DIR)/qr_demo $(TEST_DATA_DIR)/$(filter-out $@,$(MAKECMDGOALS))

test_data: $(BIN_DIR)/gen_matrix
	$(BIN_DIR)/gen_matrix $(H) $(W) > $(TEST_DATA_DIR)/test

qr_demo:
	$(NVCC) $(CFLAGS) -I$(INC_DIR) ${ARCHES} $(SRC_DIR)/qr_demo.cu -o $(BIN_DIR)/$@ $(LIBS)

gen_matrix:
	$(GCC) $(CFLAGS) -I$(INC_DIR) $(SRC_DIR)/gen_matrix.c -o $(BIN_DIR)/$@

lls_demo:
	$(NVCC) $(CFLAGS) -I$(INC_DIR) ${ARCHES} $(SRC_DIR)/lls_demo.cu -o $(BIN_DIR)/$@ $(LIBS)