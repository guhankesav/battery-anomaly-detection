CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wdouble-promotion -Iinclude
SAN = -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -g

all: build/libcellmon.so build/test_cellmon

build/libcellmon.so: src/cellmon.c include/cellmon.h
	@mkdir -p build
	$(CC) $(CFLAGS) -fPIC -shared -o $@ src/cellmon.c -lm

build/test_cellmon: tests/c/test_cellmon.c src/cellmon.c include/cellmon.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tests/c/test_cellmon.c src/cellmon.c -lm

test: build/test_cellmon
	./build/test_cellmon

# AddressSanitizer + UndefinedBehaviorSanitizer. Run in CI on Linux: on macOS 27 with Apple clang 17
# the sanitizer runtimes intermittently hang at startup, even for an empty main().
test-sanitize:
	@mkdir -p build
	$(CC) $(CFLAGS) $(SAN) -o build/test_sanitize tests/c/test_cellmon.c src/cellmon.c -lm
	./build/test_sanitize

# Cortex-M4F footprint (needs arm-none-eabi-gcc; `make arm-docker` runs it in a container).
arm:
	arm-none-eabi-gcc -std=c11 -Os -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard \
	  -ffunction-sections -fdata-sections -Wall -Wextra -Iinclude -c src/cellmon.c -o build/cellmon_m4.o
	arm-none-eabi-size -A build/cellmon_m4.o

arm-docker:
	@mkdir -p build
	docker run --rm -v $(CURDIR):/w -w /w ubuntu:24.04 sh -c \
	  "apt-get update -qq >/dev/null && apt-get install -y -qq gcc-arm-none-eabi make >/dev/null && make arm"

clean:
	rm -rf build

.PHONY: all test test-sanitize arm arm-docker clean
