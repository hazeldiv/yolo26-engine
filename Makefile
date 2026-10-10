CC           := gcc
VULKAN_INC   := -I"$(VULKAN_SDK)/Include"

CFLAGS       := -O2 -Wall -Wextra -Iinclude $(VULKAN_INC)
LDFLAGS      := -L"$(VULKAN_SDK)/Lib" -lvulkan-1 -luser32 -lgdi32

TARGET       := test.exe
DLL          := yolo26.dll
SRC_DIR      := src
BUILD_DIR    := build
BIN_DIR      := bin
SHADER_DIR   := shader

CORE_SRCS    := $(SRC_DIR)/buffer.c $(SRC_DIR)/device.c $(SRC_DIR)/descriptor.c \
                $(SRC_DIR)/dispatch.c $(SRC_DIR)/fence.c $(SRC_DIR)/pipeline.c \
                $(SRC_DIR)/session.c $(SRC_DIR)/onnx.c $(SRC_DIR)/compile.c
ENGINE_SRCS  := $(CORE_SRCS) $(SRC_DIR)/graph.c $(SRC_DIR)/decode.c
CORE_OBJS    := $(ENGINE_SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
TEST_OBJS    := $(BUILD_DIR)/test_main.o $(BUILD_DIR)/yolo.o $(CORE_OBJS)
DLL_OBJS     := $(BUILD_DIR)/yolo.o $(CORE_OBJS)

rwildcard    = $(foreach d,$(wildcard $(1)*),$(call rwildcard,$(d)/,$(2)) $(filter $(subst *,%,$(2)),$(d)))
SHADERS      := $(call rwildcard,$(SHADER_DIR)/,*.comp)
SHADER_OUT   := $(BIN_DIR)/shader
SHADER_OUT_W := $(subst /,\,$(SHADER_OUT))
SHADERS_OBJS := $(addprefix $(SHADER_OUT)/,$(notdir $(SHADERS:.comp=.spv)))

all: ${SHADERS_OBJS} $(BIN_DIR)/$(TARGET) $(BIN_DIR)/$(DLL)

$(BIN_DIR)/$(TARGET): $(TEST_OBJS)
	@if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built successfully: $@"

$(BIN_DIR)/$(DLL): $(DLL_OBJS)
	@if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)
	@echo "Built successfully: $@"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

define COMPILE_SHADER
$(SHADER_OUT)/$(notdir $(basename $(1))).spv: $(1)
	@if not exist $(SHADER_OUT_W) mkdir $(SHADER_OUT_W)
	"$(VULKAN_SDK)/Bin/glslangValidator" -V --target-env vulkan1.1 "$(1)" -o $$@
endef
$(foreach f,$(SHADERS),$(eval $(call COMPILE_SHADER,$(f))))

run: all
	@cd bin && test.exe

clean:
	@if exist $(BIN_DIR) rmdir /S /Q $(BIN_DIR)
	@if exist $(BUILD_DIR) rmdir /S /Q $(BUILD_DIR)

.PHONY: all run clean
