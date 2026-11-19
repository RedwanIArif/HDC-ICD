## Encoder comparison at D = 4096 (integer prototypes, 20 epochs, all 30 features)

| Encoder | Input | Seeds | KiB | Val F-beta | Test F-beta | Sens (%) | Spec (%) |
|---|---|---|---|---|---|---|---|
| record | samples | 3 | 672.0 | 0.882 ± 0.006 | 0.820 ± 0.012 | 98.3 ± 1.1 | 25.4 ± 10.6 |
| ngram | samples | 3 | 47.0 | 0.871 ± 0.006 | 0.794 ± 0.008 | 99.7 ± 0.4 | 5.4 ± 6.2 |
| sinusoid | features | 3 | 139.0 | 0.951 ± 0.001 | 0.946 ± 0.004 | 97.6 ± 0.1 | 86.6 ± 1.6 |
| feature | features | 3 | 62.0 | 0.960 ± 0.003 | 0.960 ± 0.003 | 98.1 ± 0.1 | 90.5 ± 1.3 |
