# Deploying the HDC detector on the NUCLEO-F303K8

All Python scripts go next to `train_hdc.py` and are run from that folder.

## 1. Export the trained model (PC, about 1 minute)

    python export_model.py

Creates `device_export/` with `hdc_model.h`, `hdc_model.c`, `device_reference.npz`, `memory_report.txt`.
It prints the test F-beta of the exported seed-0 model; it should match
`results_select/no_fft_e20_d4096_s0.json` (binary model).

Optional, if `gcc` is installed: `python c_check.py --n 1000` compiles the C detector on the PC and
checks it against Python before you touch the board.

## 2. CubeIDE project (board)

1. New STM32 project, Board Selector: NUCLEO-F303K8. Accept the default peripherals.
2. Clock: 64 MHz from the internal oscillator and the PLL (the harness reports the core clock).
3. USART2 (the virtual COM port): baud rate 460800, 8N1. If transfers fail, use 115200 and pass
   `--baud 115200` to host_eval.py.
4. Copy `hdc_infer.h`, `hdc_infer.c`, `device_export/hdc_model.h`, `device_export/hdc_model.c`
   into `Core/Inc` and `Core/Src`.
5. Paste the marked parts of `hdc_main_snippet.c` into the matching USER CODE blocks of `main.c`.
6. Project > Properties > C/C++ Build > Settings > MCU GCC Compiler > Optimization: -O2
   FPU: hardware, single precision (the default).
7. Build. Record the Build Analyzer numbers: FLASH used and RAM used.

## 3. Measure (PC + board)

    pip install pyserial
    python host_eval.py --port COM5 --n 200      # quick check (find the port in Device Manager)
    python host_eval.py --port COM5              # all 5,625 test segments (about 15 minutes at 460800)

Reports: latency per segment (mean, median, p99, max), agreement with Python (predictions, levels,
Hamming distances) and the on-device test F-beta, sensitivity and specificity.

## 4. Second build: rebuilt levels

In `hdc_model.h` change `#define HDC_REBUILD_LEVELS 0` to `1`, rebuild, record FLASH and RAM again,
and rerun `host_eval.py --out ./device_export/device_eval_rebuilt`. Predictions must be identical; only
memory and latency change.

## 5. Baseline build (for the memory breakdown)

Comment out the `hdc_predict(seg, &r);` line, build, and record FLASH and RAM. The difference to the
builds above is the detector's own cost; the rest is HAL, UART and startup code.

## 6. Latency breakdown (optional profiling build)

In `hdc_infer.h` change `#define HDC_PROFILE 0` to `1` (and paste the updated USER CODE 3 block of
`hdc_main_snippet.c` into `main.c`), rebuild, flash, then:

    python host_eval.py --port COM5 --n 500 --profile --out ./device_export/device_eval_profile

It prints and saves the mean time of each stage: feature extraction, encoding (scaling, level lookup,
binding, bundling) and classification (Hamming distances). Predictions are identical to the normal build.
Set `HDC_PROFILE` back to 0 afterwards: flash, RAM and total latency are always reported from the normal build.

## Reading flash and RAM

After each build the console ends with a line like

       text    data     bss     dec     hex filename
      21500     120    6800   28420    6f04 project.elf

Flash = text + data. Static RAM = data + bss. Copy this line for every build (the Build Analyzer view,
Window > Show View > Build Analyzer, shows the same totals per memory region).

## Checking the optimization level

Project > Properties > C/C++ Build > Settings > Tool Settings > MCU GCC Compiler > Optimization.
Note the active configuration at the top of that dialog (Debug or Release). A fresh CubeIDE project
builds Debug with -O0. If yours is -O0, set -O2 and repeat every measurement, because latency and
flash both change.

## What to send back

- the console output of export_model.py
- the size line (text, data, bss) for the three builds: stored levels, rebuilt levels, baseline
- the optimization level and configuration name
- `device_export/device_eval.json` (stored levels) and `device_export/device_eval_rebuilt.json`
- `device_export/device_eval_profile.json`

The numbers in the paper were measured with Keil MDK and Arm Compiler 6 at -O2 with
`-ffp-mode=full -ffp-contract=off`. The same sources build with STM32CubeIDE and GCC.
