/* hdc_main_snippet.c -- paste these parts into the CubeIDE-generated Core/Src/main.c
 * (NUCLEO-F303K8, USART2 = the ST-LINK virtual COM port). Keep CubeMX's own code as it is.
 *
 * Protocol with host_eval.py:
 *   PC  -> board : 0xA5 0x5A, then 1250 little-endian float32 samples (5000 bytes)
 *   board -> PC  : 0x5A 0xA5, pred (1 B), h0 (int16), h1 (int16), cycles (uint32), core clock Hz (uint32),
 *                  15 level indices (1 B each)                                       = 30 bytes
 *                  with HDC_PROFILE = 1: + cycles of features, encoding, classification (3 x uint32) = 42 bytes
 * Latency = DWT cycle count of hdc_predict() alone (features + encoding + classification), not the UART.
 */

/* ---- USER CODE BEGIN Includes ---- */
#include <string.h>
#include "hdc_infer.h"
/* ---- USER CODE END Includes ---- */

/* ---- USER CODE BEGIN PV ---- */
static float seg[HDC_T];                 /* 5000 B: the only large buffer */
/* ---- USER CODE END PV ---- */

/* ---- USER CODE BEGIN 0 ---- */
static void dwt_init(void)
{
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static int wait_sync(void)
{
  uint8_t b = 0;
  for (;;) {
    if (HAL_UART_Receive(&huart2, &b, 1, HAL_MAX_DELAY) != HAL_OK) return 0;
    if (b != 0xA5) continue;
    if (HAL_UART_Receive(&huart2, &b, 1, HAL_MAX_DELAY) != HAL_OK) return 0;
    if (b == 0x5A) return 1;
  }
}
/* ---- USER CODE END 0 ---- */

/* ---- inside main(), after MX_USART2_UART_Init(): USER CODE BEGIN 2 ---- */
  dwt_init();
/* ---- USER CODE END 2 ---- */

/* ---- inside while (1): USER CODE BEGIN 3 ---- */
    if (!wait_sync()) continue;
    if (HAL_UART_Receive(&huart2, (uint8_t *)seg, sizeof(seg), HAL_MAX_DELAY) != HAL_OK) continue;

    hdc_result_t r;
    uint32_t t0 = DWT->CYCCNT;
    hdc_predict(seg, &r);
    uint32_t cycles = DWT->CYCCNT - t0;

#if HDC_PROFILE
    uint8_t out[42];
#else
    uint8_t out[30];
#endif
    uint32_t clk = SystemCoreClock;
    out[0] = 0x5A; out[1] = 0xA5; out[2] = r.pred;
    memcpy(&out[3], &r.h0, 2);
    memcpy(&out[5], &r.h1, 2);
    memcpy(&out[7], &cycles, 4);
    memcpy(&out[11], &clk, 4);
    memcpy(&out[15], r.levels, HDC_NFEAT);
#if HDC_PROFILE
    memcpy(&out[30], &r.cyc_feat, 4);
    memcpy(&out[34], &r.cyc_enc, 4);
    memcpy(&out[38], &r.cyc_cls, 4);
#endif
    HAL_UART_Transmit(&huart2, out, sizeof(out), HAL_MAX_DELAY);
/* ---- USER CODE END 3 ---- */
