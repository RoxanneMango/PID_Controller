include ../makefile.include

BUILD_DIR = build
OUTPUT_DIR = out
BIN_FILE = hello

OBJECTS_C =
OBJECTS_CXX = main.o
OBJECTS_ASM =

ARCHIVES = $(SAM_ARCHIVE) $(BUILD_DIR)/$(BIN_FILE).a

.PHONY: link clean flash bump putty

link: build-dir out-dir $(OBJECTS_C) $(OBJECTS_CXX) $(OBJECTS_ASM)
	$(CC) -mcpu=cortex-m3 -mthumb -Os -o $(OUTPUT_DIR)/$(BIN_FILE).elf $(LINKER_FLAGS)
	$(OBJC) -O binary $(OUTPUT_DIR)/$(BIN_FILE).elf $(OUTPUT_DIR)/$(BIN_FILE).bin
	$(SIZE) -A $(OUTPUT_DIR)/$(BIN_FILE).elf

$(OBJECTS_C): %.o: %.c
	$(CC) $(CFLAGS) $(SAM_INCLUDE) $< -o $(BUILD_DIR)/$@
	$(AR) rcs $(BUILD_DIR)/$(BIN_FILE).a $(BUILD_DIR)/$@
	
$(OBJECTS_CXX): %.o: %.cpp
	$(CXX) $(CXXFLAGS) $(SAM_INCLUDE) $< -o $(BUILD_DIR)/$@
	$(AR) rcs $(BUILD_DIR)/$(BIN_FILE).a $(BUILD_DIR)/$@

$(OBJECTS_ASM): %.o: %.S
	$(CXX) $(C_ASM_FLAGS) $(SAM_INCLUDE) $< -o $(BUILD_DIR)/$@
	$(AR) rcs $(BUILD_DIR)/$(BIN_FILE).a $(BUILD_DIR)/$@

ifeq ($(OS), Windows_NT)
build-dir:
	if not exist build mkdir build
out-dir:
	if not exist out mkdir out
else
build-dir:
	mkdir -p build
out-dir:
	mkdir -p out
endif

clean: 
	@echo "Initiating lemon pledge . . .";
	cd $(BUILD_DIR) && $(RM) *.o *.bin *.elf *.map *.d *.a
	cd $(OUTPUT_DIR) && $(RM) *.bin *.elf
	@echo "I clean . . .";

ifeq ($(OS),Windows_NT)
bump:
	@echo "Forcing reset with port-bump of 1200 bps . . ."
	$(PYTHON) $(PORT_BUMP) $(PORT)
flash:
	@echo "Forcing reset with port-bump of 1200 bps . . ."
	$(PYTHON) $(PORT_BUMP) $(PORT)
	@echo "Writing binary to the arduino due . . ."
	$(BOSSAC) -i --port=$(PORT) -U false -e -w -v -b $(OUTPUT_DIR)/$(BIN_FILE).bin -R
else
# Requires dark magic and some luck: what currently works (for me): connect to screen, then bump once, then flash twice
bump:
	python3 $(PORT_BUMP) /dev/$(PORT)
flash:
	@echo "Writing binary to the arduino due . . ."
	$(BOSSAC) -i --port=$(PORT) -U false -e -w -v -b $(OUTPUT_DIR)/$(BIN_FILE).bin -R
endif


putty:
	@putty -serial com3